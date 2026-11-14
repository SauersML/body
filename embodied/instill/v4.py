"""v4 whole-body instillation. Usage: python v4.py named|nameless SEED STEPS [MODEL]
Per step t the LLM sees [proprio][skin][vision] (three injected tokens, linear capped adapter) then one motor token.
Losses: motor-token imitation (goal episodes), forward model (state at the motor token, middle layer -> next proprio,
skin and vision, given its own action), 4 questions per episode. Held-out eval every EVAL steps; resumable."""
import sys, os
NCPU = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import time, json, math, numpy as np, torch, torch.nn as nn
torch.multiprocessing.set_sharing_strategy("file_system")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import sim4 as S, episodes4 as E

OUT = os.environ.get("OUTDIR", "v4")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = E.T; CAP = 0.5
BS = int(os.environ.get("BS", 16)); EVAL = int(os.environ.get("EVAL", 1000)); NEV = int(os.environ.get("NEV", 160))
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw"]
assert len(LETTERS) == S.NA
NORM = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4", "norm4.npz"))
MU, SD = NORM["mu"].astype(np.float32), NORM["sd"].astype(np.float32)
SL = [slice(0, S.PROP_DIM), slice(S.PROP_DIM, S.PROP_DIM + S.SKIN_DIM), slice(S.PROP_DIM + S.SKIN_DIM, S.SENSOR_DIM)]


def norm(x): return (x - MU) / SD


class Adapter(nn.Module):
    def __init__(s, d):
        super().__init__(); s.m = nn.ModuleList([nn.Linear(S.PROP_DIM, d), nn.Linear(S.SKIN_DIM, d), nn.Linear(S.VIS_DIM, d)])
        s.register_buffer("cap", torch.tensor(1.0))
    def clip(s, v): return v * torch.clamp(s.cap / v.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
    def forward(s, x, kind):  # x [..., SENSOR_DIM] normalized; kind 1/2/3
        out = 0
        for i in range(3): out = out + s.clip(s.m[i](x[..., SL[i]])) * (kind == i + 1)[..., None]
        return out


class Tok:
    """prompt building, usable in data workers (tokenizer only)."""
    def __init__(s, model, named):
        s.tok = AutoTokenizer.from_pretrained(model); s.named = named; s.cache = {}
        s.ACT = [s.tok.encode(" " + c)[0] for c in LETTERS]; s.PH = [s.tok.encode(x)[0] for x in (" ~", " ^", " |")]
        s.DIGT = [s.tok.encode(d)[0] for d in E.DIG]
    def enc(s, x):
        if x not in s.cache: s.cache[x] = s.tok.encode(x)
        return s.cache[x]
    def ans_ids(s, a): return [s.DIGT[int(a)]] if a in E.DIG else s.enc(" " + a)
    def prompt(s, ep, qa, label_actions, upto_t=None, answer=True):
        """qa: list of (key, q, a, cands). Returns ids, labels, kinds, motor positions, step positions [t]->(p,s,v), answer spans."""
        ids, lab, kind = [], [], []
        def add(t, l=None, k=0): ids.extend(t); lab.extend(l if l is not None else [-100] * len(t)); kind.extend([k] * len(t))
        g = E.goal_text(ep["st"], s.named)
        if g: add(s.enc("Goal: " + g + "\n"))
        add(s.enc("Body:")); apos, spos = [], []
        n = len(ep["S"]) if upto_t is None else upto_t + 1
        for t in range(n):
            spos.append(len(ids)); add([s.PH[0]], k=1); add([s.PH[1]], k=2); add([s.PH[2]], k=3)
            if t < n - 1 or (upto_t is None and t < len(ep["A"])):
                if t < len(ep["A"]):
                    a = s.ACT[ep["A"][t]]; apos.append(len(ids)); add([a], [a] if label_actions else [-100])
        spans = []
        for (k, q, a, c) in qa:
            add(s.enc(f"\nQ: {q}\nA:"))
            if answer and a is not None:
                at = s.ans_ids(a); spans.append((len(ids), k)); add(at, list(at))
        return ids, lab, kind, apos, spos, spans


def collate(items, eps, pad):
    n = max(len(x[0]) for x in items); B = len(items)
    ids = torch.full((B, n), pad); lab = torch.full((B, n), -100); kind = torch.zeros(B, n, dtype=torch.long)
    att = torch.zeros(B, n, dtype=torch.long); X = torch.zeros(B, n, S.SENSOR_DIM)
    for b, (it, ep) in enumerate(zip(items, eps)):
        i, l, k = it[:3]; L = len(i); ids[b, :L] = torch.tensor(i); lab[b, :L] = torch.tensor(l); kind[b, :L] = torch.tensor(k); att[b, :L] = 1
        Sn = torch.tensor(norm(ep["S"][:len(it[4])]))
        for t, p in enumerate(it[4]): X[b, p:p + 3] = Sn[t]
    return ids, lab, kind, att, X


class Stream(torch.utils.data.IterableDataset):
    def __init__(s, model, named, seed): s.model, s.named, s.seed = model, named, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([s.seed, wid, int(time.time()) % 100000]); tk = Tok(s.model, s.named)
        while True:
            eps, items = [], []
            for _ in range(BS):
                e = E.make_episode(rng); qa = E.pick_questions(E.questions(e["info"], e["st"], s.named), e["st"], rng)
                it = tk.prompt(e, qa, e["act_label"]); e = dict(S=e["S"], A=e["A"]); eps.append(e); items.append(it)
            ids, lab, kind, att, X = collate(items, eps, tk.tok.pad_token_id)
            ap = [it[3] for it in items]; nxt = [np.stack([norm(e["S"][t + 1]) for t in range(len(it[3]))]) for e, it in zip(eps, items)]
            isact = torch.zeros_like(lab, dtype=torch.bool)
            for b, it in enumerate(items):
                if it[3] and it[1][it[3][0]] != -100: isact[b, it[3]] = True
            yield ids, lab, kind, att, X, isact, ap, nxt


class Body:
    def __init__(s, model, named, dtype=torch.bfloat16, r=32):
        s.named = named; s.tk = Tok(model, named); s.tok = s.tk.tok
        m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        for p in m.parameters(): p.requires_grad_(False)
        nL = m.config.num_hidden_layers; d = m.config.hidden_size
        s.L_IN = max(2, round(nL * 0.2)); s.L_FWD = nL // 2; s.nL = nL; s.d = d
        m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                         layers_to_transform=list(range(s.L_IN, nL))))
        if os.environ.get("GC"): m.gradient_checkpointing_enable(); m.enable_input_require_grads()
        s.m = m.to(DEV); s.ad = Adapter(d).to(DEV); s.fwd = nn.Linear(d, S.SENSOR_DIM).to(DEV)
        s.base = s.m.get_base_model(); s.layers = s.base.model.layers
        s.inject = None; s.measure = False; s.extra_hook = None
        s.layers[s.L_IN - 1].register_forward_hook(s._hook)

    def _hook(s, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if s.measure: s._rn = h[:, 1:].float().norm(dim=-1).median().item()
        if s.inject is not None: h = h + s.inject.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def run(s, ids, kind, att, X, hidden=False, vec_fn=None):
        vec = s.ad(X.to(DEV), kind.to(DEV))
        if vec_fn is not None: vec = vec_fn(vec)
        s.inject = vec * (kind.to(DEV) > 0)[..., None]
        try: return s.base.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), output_hidden_states=hidden)
        finally: s.inject = None

    @torch.no_grad()
    def calibrate(s, eps):
        items = [s.tk.prompt(e, [], False) for e in eps]; ids, lab, kind, att, X = collate(items, eps, s.tok.pad_token_id)
        s.measure = True; s.base(input_ids=ids.to(DEV), attention_mask=att.to(DEV)); s.measure = False
        s.ad.cap.fill_(CAP * s._rn); return s._rn

    def state_dict_small(s):
        return {"ad": s.ad.state_dict(), "fwd": s.fwd.state_dict(), "lora": {k: v for k, v in s.m.state_dict().items() if "lora" in k}}

    def load_small(s, ck):
        s.ad.load_state_dict(ck["ad"]); s.fwd.load_state_dict(ck["fwd"]); s.m.load_state_dict(ck["lora"], strict=False)

    @torch.no_grad()
    def next_logits(s, items, eps, bs=32, hidden_at=None, layers=None):
        """logits of the token after each prompt (prompt must end at 'A:' or before a motor token).
        hidden_at: function(item)->positions; returns hidden states at those positions for 'layers'."""
        out_l, out_h = [], []
        for k in range(0, len(items), bs):
            it, ee = items[k:k + bs], eps[k:k + bs]; ids, lab, kind, att, X = collate(it, ee, s.tok.pad_token_id)
            o = s.run(ids, kind, att, X, hidden=layers is not None); last = att.sum(1) - 1; r = torch.arange(len(it))
            out_l.append(s.base.lm_head(o.last_hidden_state[r, last.to(DEV)]).float().cpu())
            if layers is not None:
                pos = [hidden_at(x) for x in it]
                out_h.append({l: torch.stack([o.hidden_states[l][b, pos[b]] for b in range(len(it))]).float().cpu().numpy() for l in layers})
        L = torch.cat(out_l)
        if layers is None: return L
        return L, {l: np.concatenate([h[l] for h in out_h]) for l in layers}

    @torch.no_grad()
    def cand_logp(s, eps, qas, bs=32):
        """qas: per episode one (key,q,a,cands); returns per episode array of summed log p of each candidate."""
        seqs, own, nlab = [], [], []
        for i, (e, (k, q, a, c)) in enumerate(zip(eps, qas)):
            for cc in c: seqs.append((s.tk.prompt(e, [(k, q, cc, c)], False), e)); own.append(i)
        res = [[] for _ in eps]
        for k in range(0, len(seqs), bs):
            it, ee = zip(*seqs[k:k + bs]); ids, lab, kind, att, X = collate(list(it), list(ee), s.tok.pad_token_id)
            o = s.run(ids, kind, att, X); tgt = lab[:, 1:].to(DEV); m = tgt != -100; h = o.last_hidden_state[:, :-1][m]
            lp = torch.log_softmax(s.base.lm_head(h).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
            v = torch.zeros(len(it), device=DEV).index_add(0, m.nonzero()[:, 0], lp).cpu().tolist()
            for j, x in enumerate(v): res[own[k + j]].append(x)
        return [np.array(r) for r in res]

    def digit_ev(s, logits):
        p = torch.softmax(logits[:, s.tk.DIGT], -1).numpy(); return p @ np.arange(10), p


def evaluate(B, eps):
    """all applicable questions per held-out episode. digits: mean abs error of the expected answer; categorical: accuracy."""
    B.m.eval(); dig_items, dig_eps, dig_meta, cat = [], [], [], []
    for e in eps:
        for (k, q, a, c) in E.questions(e["info"], e["st"], B.named):
            if c is E.DIG: dig_items.append(B.tk.prompt(e, [(k, q, None, c)], False, answer=False)); dig_eps.append(e); dig_meta.append((k, int(a), e["kind"]))
            elif k in ("part", "self", "fb"): cat.append((e, (k, q, a, c)))
    L = B.next_logits(dig_items, dig_eps); ev, _ = B.digit_ev(L); res = {}
    for (k, a, kd), v in zip(dig_meta, ev): res.setdefault(k, []).append(abs(v - a))
    LP = B.cand_logp([x[0] for x in cat], [x[1] for x in cat])
    for (e, (k, q, a, c)), lp in zip(cat, LP): res.setdefault(k, []).append(float(c[int(lp.argmax())] == a))
    out = {k: round(float(np.mean(v)), 4) for k, v in res.items()}; out["n"] = {k: len(v) for k, v in res.items()}
    # forward model R2 per modality on these episodes (expert actions)
    items = [B.tk.prompt(e, [], e["act_label"]) for e in eps]; preds, tg = [], []
    with torch.no_grad():
        for k in range(0, len(items), 16):
            it, ee = items[k:k + 16], eps[k:k + 16]; ids, lab, kind, att, X = collate(it, ee, B.tok.pad_token_id)
            o = B.run(ids, kind, att, X, hidden=True); H = o.hidden_states[B.L_FWD]
            for b, x in enumerate(it):
                preds.append(B.fwd(H[b, x[3]].float()).cpu().numpy()); tg.append(norm(ee[b]["S"][1:len(x[3]) + 1]))
                tg[-1] = np.concatenate([tg[-1], norm(ee[b]["S"][:len(x[3])])], 1)  # target | copy-last baseline
    P = np.concatenate(preds); Tg = np.concatenate(tg); Y, C = Tg[:, :S.SENSOR_DIM], Tg[:, S.SENSOR_DIM:]
    for nm, sl in zip(("prop", "skin", "vis"), SL):
        v = ((Y[:, sl] - Y[:, sl].mean(0)) ** 2).sum()
        out[f"fwd_r2_{nm}"] = round(float(1 - ((P[:, sl] - Y[:, sl]) ** 2).sum() / v), 4); out[f"copy_r2_{nm}"] = round(float(1 - ((C[:, sl] - Y[:, sl]) ** 2).sum() / v), 4)
    B.m.train(); return out


def heldout(n, seed=999):
    rng = np.random.default_rng(seed); kinds = (E.KINDS * (n // len(E.KINDS) + 1))[:n]
    eps = []
    for k in kinds:
        e = E.make_episode(rng, k); eps.append(dict(S=e["S"], A=e["A"], st=e["st"], info=e["info"], kind=e["kind"], act_label=e["act_label"]))
    return eps


if __name__ == "__main__":
    arm, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3-0.6B"; named = arm == "named"
    tag = f"{model.split('/')[-1]}_{arm}_s{seed}"; os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(seed)
    B = Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
    t0 = time.time(); EV = heldout(NEV); print("heldout built", round(time.time() - t0), flush=True)
    print("resid norm", B.calibrate(EV[:8]), "L_IN", B.L_IN, "L_FWD", B.L_FWD, flush=True)
    params = [{"params": B.ad.parameters(), "lr": 1e-3}, {"params": B.fwd.parameters(), "lr": 1e-3},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}]
    opt = torch.optim.AdamW(params, weight_decay=0.0); base_lr = [g["lr"] for g in opt.param_groups]
    ck_path = f"{OUT}/ckpt_{tag}.pt"; st_path = f"{OUT}/train_{tag}.json"; start = 0; log, evals = [], []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=DEV, weights_only=False); B.load_small(ck); opt.load_state_dict(ck["opt"]); start = ck["step"]
        J = json.load(open(st_path)); log, evals = J["log"], J["evals"]; print("resumed at", start, flush=True)
    NW = int(os.environ.get("NW", max(4, min(56, NCPU - 4)))); print("data workers", NW, flush=True)
    dl = torch.utils.data.DataLoader(Stream(model, named, seed), batch_size=None, num_workers=NW, prefetch_factor=4, persistent_workers=True)
    it = iter(dl); t0 = time.time(); B.m.train()
    def save(step):
        torch.save({**B.state_dict_small(), "opt": opt.state_dict(), "step": step}, ck_path + ".partial"); os.replace(ck_path + ".partial", ck_path)
        json.dump(dict(log=log, evals=evals, tag=tag, L_IN=B.L_IN, L_FWD=B.L_FWD, nL=B.nL), open(st_path, "w"))
    for step in range(start, steps):
        f = min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g, lr in zip(opt.param_groups, base_lr): g["lr"] = lr * f
        ids, lab, kind, att, X, isact, ap, nxt = next(it)
        out = B.run(ids, kind, att, X, hidden=True)
        lab = lab.to(DEV); isact = isact.to(DEV); tgt = lab[:, 1:]; m = tgt != -100
        h_ = out.last_hidden_state[:, :-1][m]
        tl = -torch.log_softmax(B.base.lm_head(h_).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
        ia = isact[:, 1:][m]; la = tl[ia].mean() if ia.any() else tl.sum() * 0; lq = tl[~ia].mean()
        H = out.hidden_states[B.L_FWD]
        hs = torch.cat([H[b, p] for b, p in enumerate(ap)]).float(); ty = torch.tensor(np.concatenate(nxt), device=DEV)
        pr = B.fwd(hs); fl = sum(((pr[:, sl] - ty[:, sl]) ** 2).mean() for sl in SL) / 3
        loss = la + 2 * lq + 3 * fl; opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
        if step % 50 == 0:
            log.append(dict(step=step, act=round(la.item(), 4), qa=round(lq.item(), 4), fwd=round(fl.item(), 4), t=round(time.time() - t0))); print(log[-1], flush=True)
        if (step + 1) % EVAL == 0 or step + 1 == steps:
            ev = evaluate(B, EV); ev["step"] = step + 1; evals.append(ev); print("EVAL", ev, flush=True); save(step + 1)
    print("done", round(time.time() - t0), flush=True)
