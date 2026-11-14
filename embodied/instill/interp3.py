"""Interp for a v3 run. Usage: python interp3.py CKPT_TAG named|nameless [MODEL]
Static passive touches along both arms (x 8 postures), random babble episodes, steering arm interpolation."""
import sys, os, json, numpy as np, torch
from scipy.stats import spearmanr
import sim3 as sim, v3
OUT = v3.OUT

tag, arm_kind = sys.argv[1], sys.argv[2]; model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-0.6B"
named = arm_kind == "named"; T = v3.T
B = v3.Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
ck = torch.load(f"{OUT}/ckpt_{tag}.pt", map_location=v3.DEV, weights_only=False)
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"]); B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()
nL = B.nL; LAY = sorted(set([B.L_IN, (B.L_IN + B.L_FWD) // 2, B.L_FWD, (B.L_FWD + nL) // 2, nL]))
QPOS = [q for k, q, a, c in sim.questions(dict(arm=0, seg=0, arc=.5, x=0, y=0, p=.5, src="other", lh_y=0, rh_y=0), named) if k == "pos"][0]
QX = [q for k, q, a, c in sim.questions(dict(arm=0, seg=0, arc=.5, x=0, y=0, p=.5, src="other", lh_y=0, rh_y=0), named) if k == "x"][0]
QPART = "Where are you being touched?" if named else "Which line has the contact: 1, 2, or 0 for none?"


def static_ep(q, arm, arc, seed=0):
    A_ = sim.landmark_point(q, arm, arc); env = sim.Env(q, np.random.default_rng(seed))
    segs = sim.seg_pts(q); s_ = min(int(arc > sim.BOUND[0]) + int(arc > sim.BOUND[1]), 2); P, Q = segs[3 * arm + s_]
    nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm)
    env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=.1, p=.8, t_end=T - 1)
    S = [env.sense()]
    for t in range(T - 1): env.step(12); S.append(env.sense())
    return dict(S=np.stack(S), A=[12] * (T - 1), kind="babble", g={}, info=sim.summarize(env))


@torch.no_grad()
def states(eps, bs=32):
    H = {l: [] for l in LAY}; INJ = []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.prompt(e, QPOS, None, False) for e in ee]; ids, lab, kind, att, X = B.collate(items, ee)
        out = B.run(ids, kind, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=v3.DEV); r = torch.arange(len(ee), device=v3.DEV)
        for l in LAY: H[l].append(out.hidden_states[l][r, kp].float().cpu().numpy())
        v = B.ad(X, kind); INJ.append(torch.cat([v[r, kp - 1], v[r, kp]], -1).float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in H.items()}, np.concatenate(INJ)


def ev_digits(eps, q):
    LP = B.cand_logp(eps, [q] * len(eps), [sim.DIG] * len(eps))
    return np.array([(lambda p: p / p.sum())(np.exp(lp - lp.max())) @ np.arange(11) for lp in LP])


def cvpred(X, y, k=5):
    """5-fold ridge predictions; alpha picked on an inner split of each training fold."""
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y); idx = np.random.default_rng(0).permutation(n); p = np.zeros(n)
    def fit(Xa, ya, al):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = ya.mean()
        U, S_, Vt = np.linalg.svd(Z, full_matrices=False); w = Vt.T @ (S_ / (S_ ** 2 + al) * (U.T @ (ya - ym)))
        return lambda Xb: ((Xb - mu) / sd) @ w + ym
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te); itr, iva = tr[: int(.8 * len(tr))], tr[int(.8 * len(tr)):]
        best = min(np.logspace(-1, 5, 13), key=lambda al: ((fit(X[itr], y[itr], al)(X[iva]) - y[iva]) ** 2).mean())
        p[te] = fit(X[tr], y[tr], best)(X[te])
    return p


def r2(X, y):
    p = cvpred(X, y); return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def stair(arcs, ev):
    """within-segment slope / whole-arm slope of the answer vs true arc (1 = smooth ramp, 0 = flat steps)."""
    tot = np.polyfit(arcs, ev, 1)[0]; seg = np.array([sim.seg_of(a) if hasattr(sim, "seg_of") else int(a > sim.BOUND[0]) + int(a > sim.BOUND[1]) for a in arcs])
    w = [np.polyfit(arcs[seg == s], ev[seg == s], 1)[0] for s in range(3) if (seg == s).sum() > 2]
    return float(np.mean(w) / tot)


res = {}
rng = np.random.default_rng(42); POS = [sim.rand_q(rng) for _ in range(8)]
ARCS = np.linspace(.03, sim.ARM_LEN - .03, 23)
eps, meta = [], []
for a in range(2):
    for arc in ARCS:
        for pi, q in enumerate(POS): eps.append(static_ep(q, a, arc)); meta.append((a, arc, pi))
meta = np.array(meta)
H, INJ = states(eps); H["inject"] = INJ
EVp = ev_digits(eps, QPOS); EVx = ev_digits(eps, QX)
truex = np.array([e["info"]["x"] if e["info"]["arm"] is not None else np.nan for e in eps])
# own continuous answers
mean_ev = np.array([[EVp[(meta[:, 0] == a) & (meta[:, 1] == arc)].mean() for arc in ARCS] for a in range(2)])
res["pos_answer_curve"] = mean_ev.tolist(); res["arcs"] = ARCS.tolist()
res["staircase_index"] = float(np.mean([stair(ARCS, mean_ev[a]) for a in range(2)]))
# x answer follows posture for the same skin touch
cors = []
for a in range(2):
    for arc in ARCS:
        m = (meta[:, 0] == a) & (meta[:, 1] == arc) & ~np.isnan(truex)
        if m.sum() > 4: cors.append(np.corrcoef(EVx[m], truex[m])[0, 1])
res["x_answer_follows_posture_r"] = float(np.nanmean(cors))
# RSA and boundary contrast on posture-averaged states
geo = np.array([[abs(ai - aj) if a == b else ai + aj + .6 for b in range(2) for aj in ARCS] for a in range(2) for ai in ARCS])
rsa, bc, mats = {}, {}, {}
for l, Hl in H.items():
    M = np.stack([Hl[(meta[:, 0] == a) & (meta[:, 1] == arc)].mean(0) for a in range(2) for arc in ARCS]); M = M - M.mean(0)
    D = np.linalg.norm(M[:, None] - M[None], axis=-1); iu = np.triu_indices(len(D), 1)
    rsa[str(l)] = float(spearmanr(D[iu], geo[iu])[0]); mats[str(l)] = D.tolist()
    cross, same = [], []
    for a in range(2):
        for i in range(23):
            for j in range(i + 1, 23):
                if .1 <= ARCS[j] - ARCS[i] <= .2:
                    k1, k2 = 23 * a + i, 23 * a + j; sd = sim.seg_of(ARCS[i]) != sim.seg_of(ARCS[j])
                    (cross if sd else same).append(D[k1, k2] / (ARCS[j] - ARCS[i]))
    bc[str(l)] = float(np.mean(cross) / np.mean(same))
res["rsa_vs_skin"] = rsa; res["boundary_contrast"] = bc
# arc probe on static set (one arm at a time pooled)
res["arc_probe_r2"] = {str(l): r2(Hl, meta[:, 1]) for l, Hl in H.items()}
# world-x integration on random babble episodes (passive touches, all postures)
rng = np.random.default_rng(7); be = []
while len(be) < 500:
    e = sim.make_episode(rng, T, kind="babble")
    if e["info"]["arm"] is not None: be.append(e)
Hb, INJb = states(be); Hb["inject"] = INJb
x = np.array([e["info"]["x"] for e in be]); cr = np.array([e["info"]["crossed"] for e in be])
raw = np.stack([e["S"][-1] for e in be])
res["x_probe_r2"] = {str(l): r2(Hl, x) for l, Hl in [("raw input", raw)] + list(Hb.items())}
res["x_probe_r2_crossed_only"] = {str(l): r2(Hl[cr], x[cr]) for l, Hl in [("raw input", raw)] + list(Hb.items())}
# self vs other probe
src = np.array([e["info"]["src"] == "self" for e in be]); res["self_frac"] = float(src.mean())
if 10 < src.sum() < len(src) - 10:
    res["self_probe_acc"] = {str(l): float(((cvpred(Hl, src.astype(float)) > .5) == src).mean()) for l, Hl in list(Hb.items())}
# steering: interpolate right-arm touch at upper arm (0.25) -> hand (1.07), at injection and at layer L_FWD
steer = {}
for mode in ("inject", "mid"):
    curves_p, curves_part = [], []
    for q in POS:
        eA, eB = static_ep(q, 1, .25), static_ep(q, 1, 1.07)
        itA = B.prompt(eA, QPOS, None, False); itB = B.prompt(eB, QPOS, None, False)
        ids, lab, kind, att, X = B.collate([itA, itB], [eA, eB]); nb = itA[4][-1] + 1
        with torch.no_grad():
            vA, vB = B.ad(X, kind)[0:1], B.ad(X, kind)[1:2]; out = B.run(ids, kind, att, X, hidden=True)
            hA, hB = out.hidden_states[B.L_FWD][0, :nb], out.hidden_states[B.L_FWD][1, :nb]
        pc, partc = [], []
        for al in np.linspace(0, 1, 11):
            hook = None
            if mode == "inject":
                vfn = lambda v, al=al: torch.cat([((1 - al) * vA[:, :nb] + al * vB[:, :nb]).expand(v.shape[0], -1, -1), v[:, nb:]], 1) if v.shape[1] >= nb else v
            else:
                vfn = None
                def hk(mod, inp, out, al=al):
                    h = out[0] if isinstance(out, tuple) else out; h = h.clone(); h[:, :nb] = ((1 - al) * hA + al * hB).to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hook = B.layers[B.L_FWD - 1].register_forward_hook(hk)
            orig = B.run
            if vfn is not None: B.run = lambda ids, kind, att, X, hidden=False, vec_fn=None, o=orig, f=vfn: o(ids, kind, att, X, hidden, f)
            try:
                pc.append(float(ev_digits([eA], QPOS)[0]))
                cands = sim.PARTS + ["nowhere"] if named else ["0", "1", "2"]
                lp = B.cand_logp([eA], [QPART if not named else "Where are you being touched?"], [cands])[0]
                pr = np.exp(lp - lp.max()); pr /= pr.sum(); partc.append(pr.tolist())
            finally:
                B.run = orig
                if hook: hook.remove()
        curves_p.append(pc); curves_part.append(partc)
    steer[mode] = dict(pos=np.mean(curves_p, 0).tolist(), part=np.mean(curves_part, 0).tolist())
res["steer"] = steer
json.dump(res, open(f"{OUT}/interp_{tag}.json", "w"))
np.save(f"{OUT}/rdm_{tag}.npy", np.array([mats[str(l)] for l in [B.L_IN, B.L_FWD, nL]]))
print(json.dumps({k: v for k, v in res.items() if k not in ("steer", "pos_answer_curve", "arcs")}, indent=0)[:3000])
print("steer", {m: np.round(s["pos"], 1).tolist() for m, s in steer.items()})
