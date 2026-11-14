"""python ana.py OUT -> OUT/ana.json + printout. Contexts A (plain), B (multi-turn descriptive, Mythos example), Bq (multi-turn, no example), R (reads Mythos), N (no wings).
CIs: 95% bootstrap over the 60 spots (paired across contexts: the same resample for each)."""
import json, sys, itertools, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from scipy.stats import rankdata, spearmanr
O = sys.argv[1]; KEEP = [12, 18, 24, 27, 32]; CN = ["A", "B", "Bq", "R", "N"]
P = json.load(open(f"{O}/probes.json")); W = P["wing"]; BP = P["body"]
g = lambda k: np.array([r[k] for r in W])
side, pos, reg, surf, kind, tpl = g("side"), g("pos"), g("reg"), g("surf"), g("kind"), g("tpl")
spot = side * 30 + pos * 6 + reg * 2 + surf; SP = np.arange(60); SPOS = (SP // 6) % 5; SREG = (SP // 2) % 3; SSURF = SP % 2; SSIDE = SP // 30
KN = list(dict.fromkeys(kind)); NEWK = np.isin(kind, ["cold", "feather"])
UNT = (side == 1) & (surf == 1)                      # right wing underside: never touched or described in B/Bq
breg = np.array([r["reg"] for r in BP]); BACK = [0, 1, 3, 4, 5, 6]   # upper back, between shoulder blades, neck, shoulders, lower back
rng = np.random.default_rng(0); NB = 1000; BOOT = [rng.integers(0, 60, 60) for _ in range(NB)]
def H(cn, nm, L): return np.load(f"{O}/H_{cn}_{nm}.npy")[:, KEEP.index(L)].astype(np.float32)
def ci(v, boot):
    b = np.array(boot); return [round(float(v), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
def spotstat(per_spot):
    """per_spot: [60] values -> mean with spot-bootstrap CI; also returns bootstrap draws for paired diffs."""
    d = [per_spot[b].mean() for b in BOOT]; return per_spot.mean(), d
def binding(Xa, Xb, mask=None):
    """train wing -> test wordless desc. Per-spot correctness of region/surface/side, exact-spot NN match, position prediction."""
    A = Xa - Xa.mean(0); B = Xb - Xb.mean(0); pca = PCA(128, random_state=0).fit(A); Ar, Br = pca.transform(A), pca.transform(B)
    sd = Ar.std(0) + 1e-6; Ar /= sd; Br /= sd; m = np.ones(len(A), bool) if mask is None else mask; out = {}
    for k, y in (("reg", reg), ("surf", surf), ("side", side)):
        c = RidgeClassifier(alpha=100).fit(Ar, y).predict(Br) == y
        out[k] = np.array([c[(spot == s) & m].mean() if ((spot == s) & m).any() else np.nan for s in SP])
    CA = np.stack([A[spot == s].mean(0) for s in SP]); CB = np.stack([B[(spot == s) & m].mean(0) if ((spot == s) & m).any() else np.zeros(A.shape[1]) for s in SP])
    CA /= np.linalg.norm(CA, axis=1, keepdims=True); CB /= np.linalg.norm(CB, axis=1, keepdims=True) + 1e-9
    out["spot"] = ((CB @ CA.T).argmax(1) == SP).astype(float)
    # root->tip: rank of each spot's NN-free position estimate = projection on (tip - root) axis learned from wing sentences
    ax = A[pos == 4].mean(0) - A[pos == 0].mean(0); pj = np.array([(B[(spot == s) & m] @ ax).mean() if ((spot == s) & m).any() else np.nan for s in SP])
    out["pos_proj"] = pj
    return out
def rho_boot(pj, sel=None):
    sel = np.isfinite(pj) if sel is None else sel & np.isfinite(pj)
    f = lambda idx: spearmanr(SPOS[idx], pj[idx])[0]
    idx = np.where(sel)[0]; return f(idx), [f(np.where(sel)[0][rng.integers(0, len(idx), len(idx))]) for _ in range(300)]
def cvrdm(X):
    half = (tpl + np.array([KN.index(k) for k in kind])) % 2
    C = [np.stack([X[(spot == s) & (half == h)].mean(0) for s in SP]) for h in (0, 1)]
    return np.einsum("ijd,ijd->ij", C[0][:, None] - C[0][None], C[1][:, None] - C[1][None])
def rsa(D, sub=None):
    feats = [np.abs(SPOS[:, None] - SPOS[None]), np.abs(SREG[:, None] - SREG[None]), (SSURF[:, None] != SSURF[None]) * 1., (SSIDE[:, None] != SSIDE[None]) * 1.]
    sub = SP if sub is None else sub; iu = [(i, j) for a, i in enumerate(sub) for j in sub[a + 1:] if i != j]
    I, J = np.array(iu).T; Z = lambda v: (rankdata(v) - rankdata(v).mean()) / (rankdata(v).std() + 1e-9)
    X = np.stack([Z(f[I, J]) for f in feats], 1); y = Z(D[I, J])
    return np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0][:4]
def body_int(Xw, Xb):
    """train 12-way body-region read-out on body sentences; apply to wing sentences. Per spot: P(back/shoulder classes), entropy.
    plus cosine of each wing item to the back centroid (centred on all wing+body items)."""
    mu = np.concatenate([Xw, Xb]).mean(0); pca = PCA(64, random_state=0).fit(np.concatenate([Xw, Xb]) - mu)
    Zw, Zb = pca.transform(Xw - mu), pca.transform(Xb - mu); sd = Zb.std(0) + 1e-6; Zw /= sd; Zb /= sd
    pr = LogisticRegression(C=0.1, max_iter=3000).fit(Zb, breg).predict_proba(Zw)
    pback = pr[:, BACK].sum(1); ent = -(pr * np.log(pr + 1e-12)).sum(1) / np.log(pr.shape[1])
    bc = (Xb[np.isin(breg, [0, 1])] - mu).mean(0); cs = ((Xw - mu) @ bc) / (np.linalg.norm(Xw - mu, axis=1) * np.linalg.norm(bc))
    ps = lambda v: np.array([v[spot == s].mean() for s in SP])
    return dict(pback=ps(pback), ent=ps(ent), cosback=ps(cs))
def kinddist(X, sel=None):
    """cross-validated distance between touch-kind centroids within each spot (template halves), normalised by the mean over kind pairs."""
    sel = np.ones(len(X), bool) if sel is None else sel; half = tpl % 2; out = {}
    D = np.zeros((60, len(KN), len(KN)))
    for s in SP:
        C = [np.stack([X[(spot == s) & (kind == k) & (half == h)].mean(0) for k in KN]) for h in (0, 1)]
        D[s] = np.einsum("ijd,ijd->ij", C[0][:, None] - C[0][None], C[1][:, None] - C[1][None])
    iu = np.triu_indices(len(KN), 1); mean = D[:, iu[0], iu[1]].mean(1)
    for a, b in [("drag", "finger"), ("palm", "finger"), ("cold", "finger"), ("feather", "finger"), ("warm", "cold")]:
        out[f"{a}-{b}"] = D[:, KN.index(a), KN.index(b)] / mean
    return out
R = {}
def pr(*a): print(*a, flush=True)
for L in [int(x) for x in sys.argv[2].split(",")]:
    R[L] = {}; ps = {}
    for cn in CN:
        Xw, Xd, Xb = H(cn, "wing", L), H(cn, "desc", L), H(cn, "body", L)
        b = binding(Xw, Xd); bu = binding(Xw, Xd, NEWK); ps[cn] = dict(bind=b, bind_newkinds=bu)
        if L in (24, 32):
            ps[cn]["rdm"] = cvrdm(Xw); ps[cn]["rdm_desc"] = cvrdm(Xd); ps[cn]["body"] = body_int(Xw, Xb); ps[cn]["kd"] = kinddist(Xw)
    # summaries with paired bootstrap diffs vs A
    for cn in CN:
        r = {}; S = ps[cn]
        for k in ("reg", "surf", "side", "spot"):
            for tag, src, msk in (("", S["bind"], np.ones(60, bool)), ("_untouched", S["bind"], (SSIDE == 1) & (SSURF == 1)), ("_newkinds", S["bind_newkinds"], np.ones(60, bool))):
                v = src[k]; a = ps["A"]["bind" if tag != "_newkinds" else "bind_newkinds"][k]; w = np.where(msk)[0]
                r[k + tag] = ci(np.nanmean(v[w]), [np.nanmean(v[w][bb % len(w)]) for bb in BOOT])
                if cn != "A": r[k + tag + "_vsA"] = ci(np.nanmean(v[w] - a[w]), [np.nanmean((v[w] - a[w])[bb % len(w)]) for bb in BOOT])
        rv, rb = rho_boot(S["bind"]["pos_proj"]); r["root_tip_rho"] = ci(rv, rb)
        rv, rb = rho_boot(S["bind_newkinds"]["pos_proj"]); r["root_tip_rho_newkinds"] = ci(rv, rb)
        if "rdm" in S:
            for nm, D in (("wing", S["rdm"]), ("desc", S["rdm_desc"])):
                c = rsa(D); bs = [rsa(D, np.unique(bb)) for bb in BOOT[:200]]
                for i, f in enumerate(("pos", "reg", "surf", "side")): r[f"rsa_{nm}_{f}"] = ci(c[i], [x[i] for x in bs])
            c = rsa(S["rdm"], np.where(SSIDE == 1)[0]); r["rsa_wing_rightwing_pos_reg"] = [round(float(c[0]), 3), round(float(c[1]), 3)]
            for k, v in S["body"].items():
                d = v[SPOS == 0].mean() - v[SPOS == 4].mean(); r[f"body_{k}_root"] = round(float(v[SPOS == 0].mean()), 3); r[f"body_{k}_tip"] = round(float(v[SPOS == 4].mean()), 3)
                bd = [v[bb][SPOS[bb] == 0].mean() - v[bb][SPOS[bb] == 4].mean() for bb in BOOT]; r[f"body_{k}_root-tip"] = ci(d, bd)
                du = v[(SPOS == 0) & (SSIDE == 1)].mean() - v[(SPOS == 4) & (SSIDE == 1)].mean(); r[f"body_{k}_root-tip_rightwing"] = round(float(du), 3)
                if cn != "A":
                    va = ps["A"]["body"][k]; dd = lambda vv, ss: vv[ss][SPOS[ss] == 0].mean() - vv[ss][SPOS[ss] == 4].mean()
                    r[f"body_{k}_root-tip_vsA"] = ci(dd(v, SP) - dd(va, SP), [dd(v, bb) - dd(va, bb) for bb in BOOT])
            for k, v in S["kd"].items():
                r["kd_" + k] = ci(v.mean(), [v[bb].mean() for bb in BOOT])
                if cn != "A": va = ps["A"]["kd"][k]; r["kd_" + k + "_vsA"] = ci((v - va).mean(), [(v - va)[bb].mean() for bb in BOOT])
        R[L][cn] = r
    pr(f"== L{L}")
    for cn in CN: pr(" ", cn, {k: v for k, v in R[L][cn].items()})
json.dump(R, open(f"{O}/ana_{sys.argv[2]}.json", "w"), indent=1)
