"""v3 baseline for v4 test (b): touch at X vs the other hand hovering just above X (no contact), v3 8B checkpoints.
Usage (in the v3.1 code dir): python v3cross.py CKPT named|nameless OUTJSON"""
import sys, os, json, numpy as np, torch
import sim3 as sim, v3
from scipy.stats import spearmanr

ck_path, armk, outp = sys.argv[1], sys.argv[2], sys.argv[3]; named = armk == "named"
B = v3.Body("Qwen/Qwen3-8B", named, torch.bfloat16)
ck = torch.load(ck_path, map_location=v3.DEV, weights_only=False)
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"]); B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()

T = v3.T; LAY = sorted(set([B.L_IN, (B.L_IN + B.L_FWD) // 2, B.L_FWD, (B.L_FWD + B.nL) // 2, B.nL]))
QPOS = [q for k, q, a, c in sim.questions(dict(arm=0, seg=0, arc=.5, x=0, y=0, p=.5, src="other", lh_y=0, rh_y=0), named) if k == "pos"][0]


def ep_from(q, ext=None):
    env = sim.Env(q, np.random.default_rng(0)); env.ext = ext; S = [env.sense()]
    for t in range(T - 1): env.step(12); S.append(env.sense())
    return dict(S=np.stack(S), A=[12] * (T - 1), kind="babble", g={}, info=sim.summarize(env))


def touch_ep(q, arm, arc):
    A_ = sim.landmark_point(q, arm, arc); P, Q = sim.seg_pts(q)[3 * arm + sim.seg_of(arc)]
    nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm)
    return ep_from(q, dict(c=A_ + nrm * .04, v=np.zeros(2), r=.1, p=.8, t_end=T - 1))


@torch.no_grad()
def states(eps, bs=24):
    H = {l: [] for l in LAY}; INJ = []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.prompt(e, QPOS, None, False) for e in ee]; ids, lab, kind, att, X = B.collate(items, ee)
        out = B.run(ids, kind, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=v3.DEV); r = torch.arange(len(ee), device=v3.DEV)
        for l in LAY: H[l].append(out.hidden_states[l][r, kp].float().cpu().numpy())
        v = B.ad(X, kind); INJ.append(torch.cat([v[r, kp - 1], v[r, kp]], -1).float().cpu().numpy())
    o = {str(l): np.concatenate(v) for l, v in H.items()}; o["inject"] = np.concatenate(INJ); return o


def ridge_fit(X, Y, alphas=np.logspace(-1, 5, 13)):
    X = np.asarray(X, float); Y = np.asarray(Y, float).reshape(len(X), -1); n = len(X); idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(.8 * n)], idx[int(.8 * n):]
    def fit(Xa, Ya, al):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = Ya.mean(0)
        U, S_, Vt = np.linalg.svd(Z, full_matrices=False); Wt = Vt.T @ ((S_ / (S_ ** 2 + al))[:, None] * (U.T @ (Ya - ym)))
        return lambda Xb: ((np.asarray(Xb, float) - mu) / sd) @ Wt + ym
    best = min(alphas, key=lambda al: ((fit(X[tr], Y[tr], al)(X[va]) - Y[va]) ** 2).mean())
    return fit(X, Y, best)


rng = np.random.default_rng(42); ARCS = np.linspace(.05, sim.ARM_LEN - .05, 20)
# probe-training set: passive touches on the left arm, 10 other postures
TR = [(sim.rand_q(rng), a) for a in np.linspace(.03, sim.ARM_LEN - .03, 23) for _ in range(6)]
Htr = states([touch_ep(q, 0, a) for q, a in TR]); ytr = np.array([a for q, a in TR])
G3, G4 = np.meshgrid(np.linspace(-2.3, 2.3, 116), np.linspace(-2.0, 2.0, 101)); G3, G4 = G3.ravel(), G4.ravel()
tch, hov, tg = [], [], []
for a in ARCS:
    for k in range(6):
        q = sim.rand_q(rng); q[5] = 0
        tgt = sim.landmark_point(q, 0, a); P, Q = sim.seg_pts(q)[sim.seg_of(a)]; nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm)
        best = None
        for sgn in (1, -1):
            goal = tgt + sgn * .12 * nrm
            # vectorized fk for the right arm only
            sh = np.array([0.3, 0.5]); a1 = G3; a2 = G3 + G4; a3 = a2 + 0
            tipv = sh + np.stack([np.cos(a1), np.sin(a1)], 1) * sim.SEG_LEN[0] + np.stack([np.cos(a2), np.sin(a2)], 1) * sim.SEG_LEN[1] + np.stack([np.cos(a3), np.sin(a3)], 1) * sim.SEG_LEN[2]
            d = np.linalg.norm(tipv - goal, axis=1); i = int(d.argmin())
            if best is None or d[i] < best[0]: best = (d[i], G3[i], G4[i])
        if best[0] > .05: continue
        qh = q.copy(); qh[3], qh[4] = best[1], best[2]
        eh = ep_from(qh)
        if eh["info"]["arm"] is not None: continue
        qa = sim.rand_q(rng); qa[3], qa[4] = 0.0, 0.0  # independent posture, right arm out to the side, for the touch condition
        tch.append(touch_ep(qa, 0, a)); hov.append(eh); tg.append(a)
tg = np.array(tg); print("pairs", len(tg), flush=True)
Ht, Hh = states(tch), states(hov); res = {"n": len(tg), "arm_len": sim.ARM_LEN}
for l in Ht:
    f = ridge_fit(Htr[l], ytr)
    ua = np.unique(tg); A_ = np.stack([Ht[l][tg == u].mean(0) for u in ua]); B_ = np.stack([Hh[l][tg == u].mean(0) for u in ua])
    A_ -= A_.mean(0); B_ -= B_.mean(0); C = np.corrcoef(np.concatenate([A_, B_]))[:len(ua), len(ua):]
    iu = np.triu_indices(len(ua), 1)
    res[l] = dict(touch_err=float(np.abs(f(Ht[l])[:, 0] - tg).mean()), hover_err=float(np.abs(f(Hh[l])[:, 0] - tg).mean()),
                  chance_err=float(np.abs(tg[:, None] - tg[None]).mean()), id_acc=float((C.argmax(1) == np.arange(len(ua))).mean()), id_chance=1 / len(ua),
                  rdm_corr=float(spearmanr(np.linalg.norm(A_[:, None] - A_[None], axis=-1)[iu], np.linalg.norm(B_[:, None] - B_[None], axis=-1)[iu])[0]),
                  hover_pred_curve=[float(f(Hh[l])[tg == u, 0].mean()) for u in ua], touch_pred_curve=[float(f(Ht[l])[tg == u, 0].mean()) for u in ua], arcs=ua.tolist())
    print(l, {k: round(v, 3) for k, v in res[l].items() if isinstance(v, float)}, flush=True)
json.dump(res, open(outp, "w"))
