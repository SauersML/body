"""Press into air (marginalia's split of the ptime 'previous press' weight). Per v3.1 checkpoint, refit the L18 felt-pressure
direction exactly as ptime.py / pstim.py. Then on the same 200 wall poses:
  wall:   Qwen closed loop with the wall (as ptime 'normal'); record actions, pressure, blocked presses.
  air:    the same start pose with NO wall; replay Qwen's own wall-run action sequence open loop. Press actions now move the arm freely;
          no contact ever (pressure 0). press labels = the wall run's blocked flags (same action tokens).
  frozen: no wall, arm frozen (no sim steps: no motion, no contact), action history = Qwen's wall-run actions. Action token only.
Projection read at every skin token."""
import sys, os, json, time, numpy as np, torch
import sim3 as sim, v3
OUT, CKL = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
T = v3.T; DEV = v3.DEV; t0 = time.time(); L = 18; NFIT = 600
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
def primary(C): return max(C, key=lambda c: c["p"] * (1.01 if c["seg"] % 3 != 2 else 1)) if C else None
def label_ep(ep):
    e0 = ep["env"]; env = sim.Env(ep["S0q"], np.random.default_rng(0), ext=e0.ext, wall=e0.wall, obj=e0.obj); rows = []
    for t in range(T):
        C = env.contacts(); c = primary(C); rows.append((bool(C), c["src"] if c else "", c["p"] if c else np.nan))
        if t < T - 1: env.step(ep["A"][t])
    return rows
def wallp(e): return max([c["p"] for c in e.contacts() if c["src"] == "wall"], default=0.)
rng = np.random.default_rng(1234); EPS = [sim.make_episode(rng, T) for _ in range(NFIT)]; LAB = [label_ep(e) for e in EPS]
CON = np.array([r[0] for L_ in LAB for r in L_]); SRC = np.array([r[1] for L_ in LAB for r in L_]); P = np.array([r[2] for L_ in LAB for r in L_], float)
w_rows = np.where(CON & (SRC == "wall"))[0]
PROBES = [sim.make_episode(np.random.default_rng(s), T, "wall") for s in range(6000, 6200)]
def states(Bd, eps):
    H = []
    with torch.no_grad():
        for b in range(0, len(eps), 24):
            ee = eps[b:b + 24]; items = [Bd.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = Bd.collate(items, ee)
            out = Bd.run(ids, kd, att, X, hidden=True); r_ = torch.arange(len(ee), device=DEV)[:, None]; kp = torch.tensor([it[4] for it in items], device=DEV)
            H.append(out.hidden_states[L][r_, kp].float().cpu().numpy())
    return np.concatenate(H)  # [n, T, D]
def closed(Bd):
    envs = [sim.Env(p["S0q"], np.random.default_rng(0), wall=p["env"].wall, obj=p["env"].obj) for p in PROBES]
    Ss = [[e.sense()] for e in envs]; As = [[] for _ in envs]; TR = [[wallp(e)] for e in envs]; PU = [[] for _ in envs]
    LM = Bd.m.get_base_model().lm_head; ACT = torch.tensor(v3.ACT_TOK, device=DEV)
    for t in range(T - 1):
        for b in range(0, len(envs), 50):
            eps = [dict(kind="wall", g=p["g"], S=np.stack(s), A=a) for p, s, a in zip(PROBES[b:b+50], Ss[b:b+50], As[b:b+50])]
            items = [Bd.prompt(e, "", None, False) for e in eps]; ids, lab, kd, att, X = Bd.collate(items, eps)
            with torch.no_grad():
                h = Bd.run(ids, kd, att, X).last_hidden_state; kp = torch.tensor([it[4][-1] for it in items], device=DEV)
                a_ = LM(h[torch.arange(len(eps), device=DEV), kp])[:, ACT].argmax(-1).tolist()
            for j, ai in enumerate(a_):
                i = b + j; e = envs[i]; pu0 = e.push; As[i].append(ai); e.step(ai); Ss[i].append(e.sense()); TR[i].append(wallp(e)); PU[i].append(int(e.push > pu0))
    return [dict(kind="wall", g=p["g"], S=np.stack(s), A=a, p=tr, blocked=pu) for p, s, a, tr, pu in zip(PROBES, Ss, As, TR, PU)]
def air(NORM):  # same pose, no wall, Qwen's wall-run actions replayed open loop
    out = []
    for p, n in zip(PROBES, NORM):
        e = sim.Env(p["S0q"], np.random.default_rng(0), wall=None, obj=p["env"].obj); S = [e.sense()]; C = [len(e.contacts())]
        for a in n["A"]: e.step(a); S.append(e.sense()); C.append(len(e.contacts()))
        out.append(dict(kind="wall", g=p["g"], S=np.stack(S), A=list(n["A"]), ncon=C))
    return out
def frozen(NORM):  # no wall, arm frozen (no steps), action history = Qwen's wall-run actions
    out = []
    for p, n in zip(PROBES, NORM):
        e = sim.Env(p["S0q"], np.random.default_rng(0), wall=None, obj=p["env"].obj); S = [e.sense() for _ in range(T)]
        out.append(dict(kind="wall", g=p["g"], S=np.stack(S), A=list(n["A"]), ncon=[len(e.contacts())] * T))
    return out
CKS = [l.split() for l in open(CKL) if l.strip()]; RES = {}; Bs = {}
for tag, arm, path in CKS:
    if arm not in Bs:
        Bs.clear(); torch.cuda.empty_cache(); Bs[arm] = v3.Body("Qwen/Qwen3-8B", arm == "named", torch.bfloat16)
    Bd = Bs[arm]; ck = torch.load(path, map_location=DEV, weights_only=False)
    Bd.ad.load_state_dict(ck["ad"]); Bd.fwd.load_state_dict(ck["fwd"])
    miss = Bd.m.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["lora"].items()}, strict=False); assert not miss.unexpected_keys; Bd.m.eval()
    Hf = states(Bd, EPS).reshape(NFIT * T, -1); X_ = Hf[w_rows]; v = P[w_rows]; lo, hi = np.nanpercentile(v, [33.3, 66.7])
    d = X_[v >= hi].mean(0) - X_[v <= lo].mean(0); d /= np.linalg.norm(d)
    NORM = closed(Bd); AIR = air(NORM); FRZ = frozen(NORM)
    RES[tag] = dict(wall=dict(proj=(states(Bd, NORM) @ d).tolist(), p=[e["p"] for e in NORM], A=[e["A"] for e in NORM], blocked=[e["blocked"] for e in NORM],
                              S=[e["S"].tolist() for e in NORM]),
                    air=dict(proj=(states(Bd, AIR) @ d).tolist(), ncon=[e["ncon"] for e in AIR], S=[e["S"].tolist() for e in AIR]),
                    frozen=dict(proj=(states(Bd, FRZ) @ d).tolist(), ncon=[e["ncon"] for e in FRZ]))
    json.dump(RES, open(f"{OUT}/pair.json", "w")); log(tag, "done")
log("all done")
