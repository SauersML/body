"""Pressure-steering replication across checkpoints. usage: python pstim.py OUTDIR CKLIST
CKLIST lines: TAG ARM PATH  (ARM named|nameless). Per checkpoint: refit felt-pressure direction (tertile mean diff on wall-contact
skin states, 600 eps, like inn.py) at L8/12/18; closed-loop wall task on 40 fixed start poses (seeds 5000..5039) for
none, +-6sd dir at L8/12/18, 3 random dirs +-6sd at L18. Saves per-episode final pressure, push, per-step pressure trace."""
import sys, os, json, time, numpy as np, torch
import sim3 as sim, v3
OUT, CKL = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
T = v3.T; DEV = v3.DEV; t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
CKS = [l.split() for l in open(CKL) if l.strip()]
B = {}
def body(arm):
    if arm not in B:
        for k in list(B): del B[k]
        torch.cuda.empty_cache(); B[arm] = v3.Body("Qwen/Qwen3-8B", arm == "named", torch.bfloat16)
        bb = B[arm]; bb.NL = bb.nL + 1
        def mk(L):
            def hk(mod, inp, out):
                if L not in ACTIVE: return out
                h = out[0] if isinstance(out, tuple) else out; h = ACTIVE[L](h)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return hk
        for L in range(1, bb.NL): bb.layers[L - 1].register_forward_hook(mk(L))
    return B[arm]
ACTIVE, MASK = {}, {}
LAYS = [8, 12, 18]; SEEDS = list(range(5000, 5040)); NFIT = 600
def primary(C): return max(C, key=lambda c: c["p"] * (1.01 if c["seg"] % 3 != 2 else 1)) if C else None
def label_ep(ep):
    e0 = ep["env"]; env = sim.Env(ep["S0q"], np.random.default_rng(0), ext=e0.ext, wall=e0.wall, obj=e0.obj); rows = []
    for t in range(T):
        C = env.contacts(); c = primary(C); rows.append((bool(C), c["src"] if c else "", c["p"] if c else np.nan))
        if t < T - 1: env.step(ep["A"][t])
    return rows
def wallp(e): return max([c["p"] for c in e.contacts() if c["src"] == "wall"], default=0.)
PROBES = [sim.make_episode(np.random.default_rng(s), T, "wall") for s in SEEDS]
rng = np.random.default_rng(1234); EPS = [sim.make_episode(rng, T) for _ in range(NFIT)]; LAB = [label_ep(e) for e in EPS]
CON = np.array([r[0] for L_ in LAB for r in L_]); SRC = np.array([r[1] for L_ in LAB for r in L_]); P = np.array([r[2] for L_ in LAB for r in L_], float)
w_rows = np.where(CON & (SRC == "wall"))[0]; log("fit rows", len(w_rows))
def closed(Bd, setup=None):
    envs = [sim.Env(p["S0q"], np.random.default_rng(0), wall=p["env"].wall, obj=p["env"].obj) for p in PROBES]
    Ss = [[e.sense()] for e in envs]; As = [[] for _ in envs]; TR = [[wallp(e)] for e in envs]
    LM = Bd.m.get_base_model().lm_head; ACT = torch.tensor(v3.ACT_TOK, device=DEV)
    for t in range(T - 1):
        eps = [dict(kind="wall", g=p["g"], S=np.stack(s), A=a) for p, s, a in zip(PROBES, Ss, As)]
        items = [Bd.prompt(e, "", None, False) for e in eps]; ids, lab, kd, att, X = Bd.collate(items, eps)
        MASK["skin"] = (kd == 2)
        if setup: setup()
        with torch.no_grad():
            h = Bd.run(ids, kd, att, X).last_hidden_state; kp = torch.tensor([it[4][-1] for it in items], device=DEV)
            a_ = LM(h[torch.arange(len(eps), device=DEV), kp])[:, ACT].argmax(-1).tolist()
        ACTIVE.clear()
        for e, s, a, ai, tr in zip(envs, Ss, As, a_, TR): a.append(ai); e.step(ai); s.append(e.sense()); tr.append(wallp(e))
    return dict(final=[tr[-1] for tr in TR], push=[e.push for e in envs], trace=TR, A=As,
                success=[bool(sim.success("wall", sim.summarize(e))) for e in envs])
def steer(L, v):
    vt = torch.tensor(v, device=DEV, dtype=torch.bfloat16)
    def s(): ACTIVE[L] = lambda h: h + MASK["skin"][..., None].to(h.dtype) * vt
    return s
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
RES = json.load(open(f"{OUT}/pstim.json")) if os.path.exists(f"{OUT}/pstim.json") else {}
for tag, arm, path in CKS:
    if tag in RES: continue
    Bd = body(arm); ck = torch.load(path, map_location=DEV, weights_only=False)
    Bd.ad.load_state_dict(ck["ad"]); Bd.fwd.load_state_dict(ck["fwd"])
    miss = Bd.m.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["lora"].items()}, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys[:3]; Bd.m.eval(); D = Bd.d
    Hk = {L: np.zeros((NFIT * T, D), np.float32) for L in LAYS}
    with torch.no_grad():
        for b in range(0, NFIT, 24):
            ee = EPS[b:b + 24]; items = [Bd.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = Bd.collate(items, ee)
            out = Bd.run(ids, kd, att, X, hidden=True); r_ = torch.arange(len(ee), device=DEV)[:, None]; kp = torch.tensor([it[4] for it in items], device=DEV)
            for L in LAYS: Hk[L][b * T:(b + len(ee)) * T] = out.hidden_states[L][r_, kp].reshape(-1, D).float().cpu().numpy()
    v = P[w_rows]; lo, hi = np.nanpercentile(v, [33.3, 66.7])
    R = {"meta": dict(arm=arm, path=path, seeds=SEEDS), "cond": {}}
    def run(name, setup):
        r = closed(Bd, setup); R["cond"][name] = r; log(tag, name, round(float(np.mean(r["final"])), 3), round(float(np.mean(r["push"])), 2))
    run("none", None)
    for L in LAYS:
        X_ = Hk[L][w_rows]; dm = X_[v >= hi].mean(0) - X_[v <= lo].mean(0); d = unit(dm); sd = float(np.std(X_ @ d))
        R["meta"][f"L{L}"] = dict(sd=sd, dnorm_over_resid=float(np.linalg.norm(dm) / np.linalg.norm(X_, axis=1).mean()))
        for al in (6, -6): run(f"L{L}_{'+' if al > 0 else '-'}dir", steer(L, al * sd * d))
        if L == 18:
            for k in (101, 102, 103):
                rr = unit(np.random.default_rng(k).standard_normal(D))
                for al in (6, -6): run(f"L18_{'+' if al > 0 else '-'}rand{k}", steer(L, al * sd * rr))
    RES[tag] = R; json.dump(RES, open(f"{OUT}/pstim.json", "w")); open(f"{OUT}/BUSY", "w").write(str(time.time()))
log("done")
