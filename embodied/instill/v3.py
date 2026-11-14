"""v3 closed-loop body instillation. Usage: python v3.py named|nameless SEED STEPS [MODEL]
Per step t the LLM sees [proprio token][skin token] then emits/reads one motor token. Linear capped adapter (<=0.5x residual norm).
Losses: motor tokens on goal episodes (imitation of scripted experts), next-sensation forward model from the motor-token state,
one question per episode."""
import sys, os, time, json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import sim3 as sim
OUT = os.environ.get("OUTDIR", "v3")

DEV = "cuda" if torch.cuda.is_available() else "mps"
T = 10; CAP = 0.5; BS = 24
ACT_TOK = None  # filled from tokenizer: ' A'..' P'


class Adapter(nn.Module):
    def __init__(s, d):
        super().__init__(); s.j = nn.Linear(22, d); s.k = nn.Linear(16, d); s.register_buffer("cap", torch.tensor(1.0))
    def clip(s, v):
        return v * torch.clamp(s.cap / v.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
    def forward(s, x, kind):  # x [..., 38]
        vj, vk = s.clip(s.j(x[..., :22])), s.clip(s.k(x[..., 22:]))
        return vj * (kind == 1)[..., None] + vk * (kind == 2)[..., None]


class Body:
    def __init__(s, model, named, dtype=torch.float32):
        global ACT_TOK
        s.named = named; s.tok = AutoTokenizer.from_pretrained(model)
        ACT_TOK = [s.tok.encode(" " + c)[0] for c in "ABCDEFGHIJKLMNOP"]
        m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        for p in m.parameters(): p.requires_grad_(False)
        nL = m.config.num_hidden_layers; d = m.config.hidden_size
        s.L_IN = max(2, round(nL * 0.2)); s.L_FWD = nL // 2; s.nL = nL; s.d = d
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         layers_to_transform=list(range(s.L_IN, nL))))
        s.m = m.to(DEV); s.ad = Adapter(d).to(DEV); s.fwd = nn.Linear(d, sim.SENSOR_DIM).to(DEV)
        s.layers = s.m.get_base_model().model.layers
        s.inject = None; s.patch = None; s.measure = False
        s.layers[s.L_IN - 1].register_forward_hook(s._hook)
        s.PH = s.tok.encode(" ~") + s.tok.encode(" ^")

    def _hook(s, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if s.measure: s._rn = h[:, 1:].float().norm(dim=-1).median().item()
        if s.inject is not None: h = h + s.inject.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def prompt(s, ep, q, ans, label_actions):
        """token ids, labels, kinds, positions of the motor tokens."""
        tk = s.tok.encode; ids = []
        if ep["kind"] in ("reach", "find", "wall"): ids += tk("Goal: " + sim.goal_text(ep["kind"], ep["g"], s.named) + "\n")
        ids += tk("Body:"); lab = [-100] * len(ids); kind = [0] * len(ids); apos, kpos = [], []
        for t in range(len(ep["S"])):
            ids += s.PH; lab += [-100, -100]; kind += [1, 2]; kpos.append(len(ids) - 1)
            if t < T - 1 and ep["A"] is not None and t < len(ep["A"]):
                apos.append(len(ids)); ids.append(ACT_TOK[ep["A"][t]]); kind.append(0); lab.append(ACT_TOK[ep["A"][t]] if label_actions else -100)
        p = tk(f"\nQ: {q}\nA:"); ids += p; lab += [-100] * len(p); kind += [0] * len(p)
        if ans is not None: a = tk(" " + ans); ids += a; lab += a; kind += [0] * len(a)
        return ids, lab, kind, apos, kpos

    def collate(s, items, eps):
        n = max(len(x[0]) for x in items); pad = s.tok.pad_token_id; B = len(items)
        ids = torch.full((B, n), pad); lab = torch.full((B, n), -100); kind = torch.zeros(B, n, dtype=torch.long)
        att = torch.zeros(B, n, dtype=torch.long); X = torch.zeros(B, n, sim.SENSOR_DIM)
        for b, ((i, l, k, ap, kp), ep) in enumerate(zip(items, eps)):
            L = len(i); ids[b, :L] = torch.tensor(i); lab[b, :L] = torch.tensor(l); kind[b, :L] = torch.tensor(k); att[b, :L] = 1
            S = torch.tensor(ep["S"][:len(kp)]); X[b, [p - 1 for p in kp]] = S; X[b, kp] = S
        return [t.to(DEV) for t in (ids, lab, kind, att, X)]

    def run(s, ids, kind, att, X, hidden=False, vec_fn=None):
        vec = s.ad(X, kind)
        if vec_fn is not None: vec = vec_fn(vec)
        s.inject = vec * (kind > 0)[..., None]
        try: return s.m.get_base_model().model(input_ids=ids, attention_mask=att, output_hidden_states=hidden)
        finally: s.inject = None

    def lab_logp(s, out, lab):
        """Sum of log-probs of the labelled tokens per row, computing the vocab head only where labels exist."""
        tgt = lab[:, 1:]; m = tgt != -100; h = out.last_hidden_state[:, :-1][m]
        lp = torch.log_softmax(s.m.get_base_model().lm_head(h).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
        return torch.zeros(lab.shape[0], device=DEV).index_add(0, m.nonzero()[:, 0], lp), m.sum()

    @torch.no_grad()
    def calibrate(s):
        rng = np.random.default_rng(99); eps = [sim.make_episode(rng, T) for _ in range(8)]
        items = [s.prompt(e, "Where?", None, False) for e in eps]; ids, lab, kind, att, X = s.collate(items, eps)
        s.measure = True; s.m(input_ids=ids, attention_mask=att); s.measure = False
        s.ad.cap.fill_(CAP * s._rn); return s._rn

    @torch.no_grad()
    def cand_logp(s, eps, Q, C, chunk=48):
        seqs, own = [], []
        for i, (e, q, cs) in enumerate(zip(eps, Q, C)):
            for c in cs: seqs.append((s.prompt(e, q, c, False), e)); own.append(i)
        out = []
        for k in range(0, len(seqs), chunk):
            items, ee = zip(*seqs[k:k + chunk]); ids, lab, kind, att, X = s.collate(items, ee)
            out += s.lab_logp(s.run(ids, kind, att, X), lab)[0].cpu().tolist()
        res = [[] for _ in eps]
        for i, v in zip(own, out): res[i].append(v)
        return [np.array(r) for r in res]

    @torch.no_grad()
    def act(s, ep_partial):
        """Greedy motor token after the latest skin token."""
        items = [s.prompt(ep_partial, "", None, False)]
        ids, lab, kind, att, X = s.collate(items, [ep_partial])
        k = items[0][4][-1]
        lg = s.m.get_base_model().lm_head(s.run(ids, kind, att, X).last_hidden_state[0, k])[ACT_TOK]
        return int(lg.argmax())


def closed_loop(B, kind, n, seed):
    rng = np.random.default_rng(seed); succ, acts = [], []
    for _ in range(n):
        st = {}
        def pol(env, t, S):
            ep = dict(kind=kind, g=st["g"], S=np.stack(S), A=st["A"]); a = B.act(ep); st["A"].append(a); return a
        # draw the goal the same way make_episode does, then run with the model as policy
        r2 = np.random.default_rng(int(rng.integers(1 << 30)))
        probe = sim.make_episode(np.random.default_rng(r2.integers(1 << 30)), T, kind)  # expert-solvable start
        st["g"] = probe["g"]; st["A"] = []
        e = run_from(probe, pol)
        succ.append(sim.success(kind, e["info"])); acts += st["A"]
    return float(np.mean(succ)), np.bincount(acts, minlength=sim.NA).tolist()


def run_from(probe, pol):
    env0 = probe["env"]; rng = np.random.default_rng(0)
    env = sim.Env(probe["S0q"], rng, wall=env0.wall, obj=env0.obj); S = [env.sense()]
    for t in range(T - 1): a = pol(env, t, S); env.step(a); S.append(env.sense())
    return dict(S=np.stack(S), info=sim.summarize(env))


def evaluate(B, n, seed, named):
    rng = np.random.default_rng(seed); eps, Q, A, C, K = [], [], [], [], []
    for _ in range(n):
        e = sim.make_episode(rng, T, kind="babble")
        for k, q, a, c in sim.questions(e["info"], named, "babble"): eps.append(e); Q.append(q); A.append(a); C.append(c); K.append(k)
    LP = B.cand_logp(eps, Q, C); res = {}
    for e, k, a, c, lp in zip(eps, K, A, C, LP):
        g = "crossed" if e["info"]["crossed"] else "uncrossed"
        if c is sim.DIG:
            pr = np.exp(lp - lp.max()); pr /= pr.sum(); v = abs(float(pr @ np.arange(11)) - int(a))
        else: v = float(c[int(lp.argmax())] == a)
        res.setdefault(k, {}).setdefault(g, []).append(v)
    return {k: {g: round(float(np.mean(v)), 3) for g, v in d.items()} for k, d in res.items()}


if __name__ == "__main__":
    arm, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3-0.6B"; named = arm == "named"
    tag = f"{model.split('/')[-1]}_{arm}_s{seed}"; os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    B = Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
    print("resid norm", B.calibrate(), "L_IN", B.L_IN, "L_FWD", B.L_FWD, flush=True)
    opt = torch.optim.AdamW([{"params": B.ad.parameters(), "lr": 1e-3}, {"params": B.fwd.parameters(), "lr": 1e-3},
                             {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}], weight_decay=0.0)
    log = []; t0 = time.time()
    for step in range(steps):
        eps = [sim.make_episode(rng, T) for _ in range(BS)]; items = []
        for e in eps:
            qa = sim.questions(e["info"], named, e["kind"]); k, q, a, c = qa[rng.integers(len(qa))]
            items.append(B.prompt(e, q, a, e["act_label"]))
        ids, lab, kind, att, X = B.collate(items, eps)
        out = B.run(ids, kind, att, X, hidden=True)
        # motor tokens and answer tokens get equal total weight (motor labels would otherwise swamp the questions)
        tgt = lab[:, 1:]; m = tgt != -100; isact = torch.zeros_like(m)
        for b, it in enumerate(items):
            if it[3] and lab[b, it[3][0]] != -100: isact[b, [p - 1 for p in it[3]]] = True
        h_ = out.last_hidden_state[:, :-1][m]
        tl = -torch.log_softmax(B.m.get_base_model().lm_head(h_).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
        ia = isact[m]; la = tl[ia].mean() if ia.any() else tl.sum() * 0; lq = tl[~ia].mean()
        lm = la + 2 * lq
        H = out.hidden_states[B.L_FWD]; ap = torch.tensor([it[3] for it in items], device=DEV)  # [B, T-1]
        h = H[torch.arange(len(items), device=DEV)[:, None], ap].float()
        tgt = torch.tensor(np.stack([e["S"][1:] for e in eps]), device=DEV)
        fl = ((B.fwd(h) - tgt) ** 2).mean()
        loss = lm + 3 * fl; opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or os.environ.get("DBG"):
            log.append(dict(step=step, act=round(la.item(), 4), qa=round(lq.item(), 4), fwd=round(fl.item(), 5), t=round(time.time() - t0))); print(log[-1], flush=True)
    torch.save({"ad": B.ad.state_dict(), "fwd": B.fwd.state_dict(),
                "lora": {k: v for k, v in B.m.state_dict().items() if "lora" in k}}, f"{OUT}/ckpt_{tag}.pt")
    NE = int(os.environ.get("NE", 150)); NC = int(os.environ.get("NC", 40))
    ev = evaluate(B, NE, 123, named); print("eval", ev, flush=True)
    cl = {k: closed_loop(B, k, NC, 7 + i) for i, k in enumerate(["reach", "find", "wall"])}; print("closed loop", cl, flush=True)
    json.dump(dict(log=log, eval=ev, closed_loop=cl), open(f"{OUT}/train_{tag}.json", "w"), indent=1)
