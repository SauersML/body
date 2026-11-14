"""3D body instillation. python train3d.py named|nameless SEED STEPS [MODEL]
Per step: [proprio][skin][vision] injected tokens (linear capped adapter, added after layer L_IN) then a motor token
whose last-layer state feeds an action head (41 joint targets). Forward model at L_FWD. Questions in words at the end."""
import sys, os
NCPU = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import time, json, math, numpy as np, torch, torch.nn as nn
torch.multiprocessing.set_sharing_strategy("file_system")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import env as E, episodes as P

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("OUTDIR", "out")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if os.environ.get("MPS") else "cpu")
T = P.T; CAP = 0.5; NJ = E.NJ
BS = int(os.environ.get("BS", 16)); EVAL = int(os.environ.get("EVAL", 500)); NEV = int(os.environ.get("NEV", 180))
NORM = np.load(os.path.join(HERE, "norm.npz")); MU, SD = NORM["mu"].astype(np.float32), NORM["sd"].astype(np.float32)
AMU, ASD = NORM["amu"].astype(np.float32), NORM["asd"].astype(np.float32)
SL = [slice(0, E.PROP_DIM), slice(E.PROP_DIM, E.PROP_DIM + E.SKIN_DIM), slice(E.PROP_DIM + E.SKIN_DIM, E.SENSOR_DIM)]
def norm(x): return (x - MU) / SD
def anorm(a): return (a - AMU) / ASD


class Adapter(nn.Module):
    def __init__(s, d):
        super().__init__(); s.m = nn.ModuleList([nn.Linear(sl.stop - sl.start, d) for sl in SL]); s.register_buffer("cap", torch.tensor(1.0))
    def clip(s, v): return v * torch.clamp(s.cap / v.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
    def forward(s, x, kind):
        out = 0
        for i in range(3): out = out + s.clip(s.m[i](x[..., SL[i]])) * (kind == i + 1)[..., None]
        return out


class Tok:
    def __init__(s, model, named):
        s.tok = AutoTokenizer.from_pretrained(model); s.named = named; s.cache = {}
        s.PH = [s.tok.encode(x)[0] for x in (" ~", " ^", " |")]; s.MOT = s.tok.encode(" >")[0]
    def enc(s, x):
        if x not in s.cache: s.cache[x] = s.tok.encode(x)
        return s.cache[x]
    def prompt(s, ep, qa, upto_t=None, answer=True, goal=True):
        """returns ids, labels, kinds, motor positions, sensor positions, answer spans. upto_t: stop after sensors of step upto_t."""
        ids, lab, kind = [], [], []
        def add(t, l=None, k=0): ids.extend(t); lab.extend(l if l is not None else [-100] * len(t)); kind.extend([k] * len(t))
        g = P.goal_text(ep, s.named) if goal else ""
        if g: add(s.enc("Goal: " + g + "\n"))
        add(s.enc("Body:")); apos, spos = [], []
        n = T + 1 if upto_t is None else upto_t + 1
        for t in range(n):
            spos.append(len(ids)); add([s.PH[0]], k=1); add([s.PH[1]], k=2); add([s.PH[2]], k=3)
            if t < T and (upto_t is None or t <= upto_t): apos.append(len(ids)); add([s.MOT])
        spans = []
        for (k, q, a, c) in qa:
            add(s.enc(f"\nQ: {q}\nA:"))
            if answer and a is not None: at = s.enc(" " + a); spans.append((len(ids), k)); add(at, list(at))
        return ids, lab, kind, apos, spos, spans


def collate(items, eps, pad):
    n = max(len(x[0]) for x in items); B = len(items)
    ids = torch.full((B, n), pad); lab = torch.full((B, n), -100); kind = torch.zeros(B, n, dtype=torch.long)
    att = torch.zeros(B, n, dtype=torch.long); X = torch.zeros(B, n, E.SENSOR_DIM)
    for b, (it, ep) in enumerate(zip(items, eps)):
        i, l, k = it[:3]; L = len(i); ids[b, :L] = torch.tensor(i); lab[b, :L] = torch.tensor(l); kind[b, :L] = torch.tensor(k); att[b, :L] = 1
        Sn = torch.tensor(norm(ep["S"][:len(it[4])]))
        for t, p in enumerate(it[4]): X[b, p:p + 3] = Sn[t]
    return ids, lab, kind, att, X


def slim(e): return dict(S=e["S"], A=e["A"], mask=e["mask"], kind=e["kind"], info=e["info"])


class Stream(torch.utils.data.IterableDataset):
    def __init__(s, model, named, seed): s.model, s.named, s.seed = model, named, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([s.seed, wid, int(time.time() * 1000) % 100000]); tk = Tok(s.model, s.named)
        while True:
            eps, items = [], []
            for _ in range(BS):
                e = slim(P.make_episode(rng)); Q = P.questions(e, s.named); qa = [Q[i] for i in rng.permutation(len(Q))]
                items.append(tk.prompt(e, qa, goal=rng.uniform() > 0.05)); eps.append(e)
            ids, lab, kind, att, X = collate(items, eps, tk.tok.pad_token_id)
            ap = [it[3] for it in items]
            nxt = [norm(e["S"][1:T + 1]) for e in eps]; act = [anorm(e["A"]) for e in eps]; msk = [e["mask"] for e in eps]
            yield ids, lab, kind, att, X, ap, np.stack(nxt), np.stack(act), np.stack(msk)


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
        s.m = m.to(DEV); s.ad = Adapter(d).to(DEV); s.fwd = nn.Linear(d, E.SENSOR_DIM).to(DEV)
        s.act = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, NJ)).to(DEV)
        s.base = s.m.get_base_model(); s.layers = s.base.model.layers
        s.inject = None; s.measure = False; s.steer = None
        s.layers[s.L_IN - 1].register_forward_hook(s._hook)

    def _hook(s, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if s.measure: s._rn = h[:, 1:].float().norm(dim=-1).median().item()
        if s.inject is not None: h = h + s.inject.to(h.dtype)
        if s.steer is not None: h = h + s.steer.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def run(s, ids, kind, att, X, hidden=False, vec_fn=None):
        vec = s.ad(X.to(DEV), kind.to(DEV))
        if vec_fn is not None: vec = vec_fn(vec)
        s.inject = vec * (kind.to(DEV) > 0)[..., None]
        try: return s.base.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), output_hidden_states=hidden)
        finally: s.inject = None

    @torch.no_grad()
    def calibrate(s, eps):
        items = [s.tk.prompt(e, []) for e in eps]; ids, lab, kind, att, X = collate(items, eps, s.tok.pad_token_id)
        s.measure = True; s.base(input_ids=ids.to(DEV), attention_mask=att.to(DEV)); s.measure = False
        s.ad.cap.fill_(CAP * s._rn); return s._rn

    def state_dict_small(s):
        return {"ad": s.ad.state_dict(), "fwd": s.fwd.state_dict(), "act": s.act.state_dict(), "lora": {k: v for k, v in s.m.state_dict().items() if "lora" in k}}

    def load_small(s, ck):
        s.ad.load_state_dict(ck["ad"]); s.fwd.load_state_dict(ck["fwd"]); s.act.load_state_dict(ck["act"]); s.m.load_state_dict(ck["lora"], strict=False)

    @torch.no_grad()
    def cand_logp(s, eps, qas, bs=48):
        seqs, own = [], []
        for i, (e, (k, q, a, c)) in enumerate(zip(eps, qas)):
            for cc in c: seqs.append((s.tk.prompt(e, [(k, q, cc, c)]), e)); own.append(i)
        res = [[] for _ in eps]
        for k in range(0, len(seqs), bs):
            it, ee = zip(*seqs[k:k + bs]); ids, lab, kind, att, X = collate(list(it), list(ee), s.tok.pad_token_id)
            o = s.run(ids, kind, att, X); tgt = lab[:, 1:].to(DEV); m = tgt != -100; h = o.last_hidden_state[:, :-1][m]
            lp = torch.log_softmax(s.base.lm_head(h).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
            v = torch.zeros(len(it), device=DEV).index_add(0, m.nonzero()[:, 0], lp).cpu().tolist()
            for j, x in enumerate(v): res[own[k + j]].append(x)
        return [np.array(r) for r in res]


def evaluate(B, eps):
    B.m.eval(); cat = [(e, q) for e in eps for q in P.questions(e, B.named)]; res = {}
    LP = B.cand_logp([x[0] for x in cat], [x[1] for x in cat])
    for (e, (k, q, a, c)), lp in zip(cat, LP): res.setdefault(k, []).append(float(c[int(lp.argmax())] == a))
    out = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    items = [B.tk.prompt(e, []) for e in eps]; pa, ta, ma, pf, tf = [], [], [], [], []
    with torch.no_grad():
        for k in range(0, len(items), 16):
            it, ee = items[k:k + 16], eps[k:k + 16]; ids, lab, kind, att, X = collate(it, ee, B.tok.pad_token_id)
            o = B.run(ids, kind, att, X, hidden=True); H = o.hidden_states[B.L_FWD]; Hl = o.last_hidden_state
            for b, x in enumerate(it):
                pa.append(B.act(Hl[b, x[3]].float()).cpu().numpy()); ta.append(anorm(ee[b]["A"])); ma.append(ee[b]["mask"])
                pf.append(B.fwd(H[b, x[3]].float()).cpu().numpy()); tf.append(norm(ee[b]["S"][1:]))
    pa, ta, ma = np.concatenate(pa), np.concatenate(ta), np.concatenate(ma) > 0
    out["act_r2"] = round(float(1 - ((pa - ta)[ma] ** 2).sum() / ((ta[ma] - ta[ma].mean(0)) ** 2).sum()), 4)
    pf, tf = np.concatenate(pf), np.concatenate(tf)
    for nm, sl in zip(("prop", "skin", "vis"), SL):
        out[f"fwd_r2_{nm}"] = round(float(1 - ((pf[:, sl] - tf[:, sl]) ** 2).sum() / ((tf[:, sl] - tf[:, sl].mean(0)) ** 2).sum()), 4)
    B.m.train(); return out


def heldout(n, seed=999):
    rng = np.random.default_rng(seed); return [slim(P.make_episode(rng, k)) for k in (P.KINDS * (n // len(P.KINDS) + 1))[:n]]


def train_step(B, opt, batch):
    ids, lab, kind, att, X, ap, nxt, act, msk = batch
    out = B.run(ids, kind, att, X, hidden=True)
    lab = lab.to(DEV); tgt = lab[:, 1:]; m = tgt != -100; h_ = out.last_hidden_state[:, :-1][m]
    lq = -torch.log_softmax(B.base.lm_head(h_).float(), -1).gather(-1, tgt[m][:, None])[:, 0].mean()
    ap_t = torch.tensor(np.stack(ap), device=DEV); r = torch.arange(len(ap))[:, None].to(DEV)
    Hl = out.last_hidden_state[r, ap_t].float(); Hf = out.hidden_states[B.L_FWD][r, ap_t].float()
    act = torch.tensor(act, device=DEV); msk = torch.tensor(msk, device=DEV); nxt = torch.tensor(nxt, device=DEV)
    la = (((B.act(Hl) - act) ** 2).mean(-1) * msk).sum() / msk.sum().clamp(min=1)
    pr = B.fwd(Hf); lf = sum(((pr[..., sl] - nxt[..., sl]) ** 2).mean() for sl in SL) / 3
    loss = la + lq + 2 * lf; opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
    return la.item(), lq.item(), lf.item()


if __name__ == "__main__":
    arm, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3-0.6B"; named = arm == "named"
    tag = f"{model.split('/')[-1]}_{arm}"; os.makedirs(OUT, exist_ok=True); torch.manual_seed(seed)
    B = Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
    t0 = time.time(); EV = heldout(NEV); print("heldout built", round(time.time() - t0), flush=True)
    print("resid norm", B.calibrate(EV[:8]), "L_IN", B.L_IN, "L_FWD", B.L_FWD, flush=True)
    params = [{"params": list(B.ad.parameters()) + list(B.fwd.parameters()) + list(B.act.parameters()), "lr": 1e-3},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}]
    opt = torch.optim.AdamW(params, weight_decay=0.0); base_lr = [g["lr"] for g in opt.param_groups]
    ck_path = f"{OUT}/ckpt_{tag}.pt"; st_path = f"{OUT}/train_{tag}.json"; start = 0; log, evals = [], []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=DEV, weights_only=False); B.load_small(ck); opt.load_state_dict(ck["opt"]); start = ck["step"]
        J = json.load(open(st_path)); log, evals = J["log"], J["evals"]; print("resumed at", start, flush=True)
    NW = int(os.environ.get("NW", max(2, min(56, NCPU - 4)))); print("data workers", NW, flush=True)
    dl = torch.utils.data.DataLoader(Stream(model, named, seed), batch_size=None, num_workers=NW, prefetch_factor=4, persistent_workers=True)
    it = iter(dl); t0 = time.time(); B.m.train()
    def save(step):
        torch.save({**B.state_dict_small(), "opt": opt.state_dict(), "step": step, "rn": B.ad.cap.item()}, ck_path + ".partial"); os.replace(ck_path + ".partial", ck_path)
        json.dump(dict(log=log, evals=evals, tag=tag, L_IN=B.L_IN, L_FWD=B.L_FWD, nL=B.nL), open(st_path, "w"))
    for step in range(start, steps):
        f = min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g, lr in zip(opt.param_groups, base_lr): g["lr"] = lr * f
        la, lq, lf = train_step(B, opt, next(it))
        if step % 50 == 0: log.append(dict(step=step, act=round(la, 4), qa=round(lq, 4), fwd=round(lf, 4), t=round(time.time() - t0))); print(log[-1], flush=True)
        if (step + 1) % EVAL == 0 or step + 1 == steps:
            ev = evaluate(B, EV); ev["step"] = step + 1; evals.append(ev); print("EVAL", ev, flush=True); save(step + 1)
    print("done", round(time.time() - t0), flush=True)
