"""python ana.py OUTDIR LAYER_INDEX -> OUTDIR/ana_J.json + printout
(A) ordinal geometry: leave-one-position-out (and leave-one-region-out) linear read-out; the held-out level's words never occur in training,
    so a bag of words can't place them; (B) cross-naming transfer: train on 'wing' sentences, test on sentences with different words;
(C) RSA: cross-validated spot distances regressed on structure + word-embedding distances + no-wings-chat distances."""
import json, sys, re, itertools, numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
O, J = sys.argv[1], int(sys.argv[2])
P = json.load(open(f"{O}/probes.json"))
f = lambda k: np.array([r[k] for r in P["wing"]])
LB = dict(side=f("side"), pos=f("pos"), t=f("t"), reg=f("reg"), surf=f("surf"), kind=f("kind"), tpl=f("tpl"))
LB["spot"] = LB["side"] * 30 + LB["pos"] * 6 + LB["reg"] * 2 + LB["surf"]
NT = LB["tpl"].max() + 1
SP = np.arange(60); SPOS = (SP // 6) % 5; SREG = (SP // 2) % 3; SSURF = SP % 2; SSIDE = SP // 30
def H(k): cn, nm = k.split("_"); return np.load(f"{O}/H_{cn}_{nm}.npy")[:, J].astype(np.float32)
def LEX(nm): return np.load(f"{O}/lex_{nm}.npy").astype(np.float32)
def bow(nm):
    W = [re.findall(r"[a-z']+", r["text"].lower()) for r in P[nm]]; V = sorted({w for x in W for w in x}); ix = {w: i for i, w in enumerate(V)}
    X = np.zeros((len(W), len(V)), np.float32)
    for i, x in enumerate(W):
        for w in x: X[i, ix[w]] += 1
    return X
def prep(X, k=128):
    X = X - X.mean(0)
    if X.shape[1] > k: X = PCA(k, random_state=0).fit_transform(X)
    return X / (X.std(0) + 1e-6)
from sklearn.linear_model import RidgeCV
def lolo(X, lev, y):
    """leave-one-level-out ridge (alpha by inner LOO): per held-out level, mean prediction minus the training-label mean
    (a read-out with no signal predicts the training mean, which is biased against the held-out level; this removes that)."""
    Xr = prep(X); out = {}
    for l in np.unique(lev):
        tr = lev != l; out[int(l)] = float(RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Xr[tr], y[tr]).predict(Xr[~tr]).mean() - y[tr].mean())
    return out
def ordinal(X):
    pp = lolo(X, LB["pos"], LB["t"]); rp = lolo(X, LB["reg"], LB["reg"] / 2.0)
    return dict(pos_shift=[round(pp[i], 3) for i in range(5)], pos_rho=float(spearmanr(range(5), [pp[i] for i in range(5)])[0]),
                reg_shift=[round(rp[i], 3) for i in range(3)])
def transfer(Xa, Xb):
    A = Xa - Xa.mean(0); Bm = Xb - Xb.mean(0); pca = PCA(128, random_state=0).fit(A); Ar, Br = pca.transform(A), pca.transform(Bm)
    sd = Ar.std(0) + 1e-6; Ar /= sd; Br /= sd
    pt = Ridge(alpha=100).fit(Ar, LB["t"]).predict(Br); pm = [float(pt[LB["pos"] == p].mean()) for p in range(5)]
    out = dict(pos_pred=[round(x, 3) for x in pm], pos_rho_spot=float(spearmanr([pt[LB["spot"] == s].mean() for s in SP], SPOS)[0]))
    for k, ch in (("reg", 1 / 3), ("surf", .5), ("side", .5)): out[k + "_acc"] = float((RidgeClassifier(alpha=100).fit(Ar, LB[k]).predict(Br) == LB[k]).mean())
    CA = np.stack([A[LB["spot"] == s].mean(0) for s in SP]); CB = np.stack([Bm[LB["spot"] == s].mean(0) for s in SP])
    CA /= np.linalg.norm(CA, axis=1, keepdims=True); CB /= np.linalg.norm(CB, axis=1, keepdims=True)
    nn = (CB @ CA.T).argmax(1); out["spot_match"] = float((nn == SP).mean()); out["pos_match"] = float((SPOS[nn] == SPOS).mean())
    return out, pt
def cvrdm(X):
    """cross-validated squared distances between the 60 spot centroids (template halves)."""
    ki = np.unique(LB["kind"], return_inverse=True)[1]; half = (LB["tpl"] + ki) % 2
    C = [np.stack([X[(LB["spot"] == s) & (half == h)].mean(0) for s in SP]) for h in (0, 1)]
    D = np.zeros((60, 60))
    for i, j in itertools.combinations(range(60), 2): D[i, j] = D[j, i] = (C[0][i] - C[0][j]) @ (C[1][i] - C[1][j])
    return D
IU = np.triu_indices(60, 1)
def rsa(D, controls):
    """rank-regress model distances on structure (+ controls); return partial coefficients (standardised ranks)."""
    feats = dict(pos=np.abs(SPOS[:, None] - SPOS[None]), reg=np.abs(SREG[:, None] - SREG[None]),
                 surf=(SSURF[:, None] != SSURF[None]) * 1., side=(SSIDE[:, None] != SSIDE[None]) * 1.)
    Z = lambda M: (rankdata(M[IU]) - rankdata(M[IU]).mean()) / rankdata(M[IU]).std()
    X = np.stack([Z(v) for v in feats.values()] + [Z(c) for c in controls.values()], 1); y = Z(D)
    b = np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0]
    names = list(feats) + ["ctl_" + k for k in controls]
    return {n: round(float(v), 3) for n, v in zip(names, b)}
def main():
    keys = ["W_wing", "W_desc", "W_glorp", "N_wing", "N_desc", "G_glorp", "G_desc", "F_wing"]
    S = {k: H(k) for k in keys}; L = {nm: LEX(nm) for nm in ("wing", "desc", "glorp")}; BW = {nm: bow(nm) for nm in L}
    res = {"ordinal": {}, "transfer": {}, "rsa": {}}
    for k in keys: res["ordinal"][k] = ordinal(S[k])
    for nm in L: res["ordinal"]["wordemb_" + nm] = ordinal(L[nm]); res["ordinal"]["bagofwords_" + nm] = ordinal(BW[nm])
    TP = {}
    for a, b in [("W_wing", "W_desc"), ("N_wing", "N_desc"), ("G_glorp", "G_desc"), ("W_wing", "W_glorp"), ("W_wing", "G_glorp"),
                 ("W_desc", "W_wing"), ("N_desc", "N_wing"), ("F_wing", "W_desc")]:
        res["transfer"][f"{a}>{b}"], TP[f"{a}>{b}"] = transfer(S[a], S[b])
    res["transfer"]["wordemb wing>desc"] = transfer(L["wing"], L["desc"])[0]
    np.save(f"{O}/transfer_pred_{J}.npy", np.stack([TP["W_wing>W_desc"], TP["N_wing>N_desc"]]))
    RD = {k: cvrdm(S[k]) for k in keys}; LD = {nm: cvrdm(L[nm]) for nm in L}
    for k in keys:
        nm = k.split("_")[1]; res["rsa"][k] = rsa(RD[k], {})
        res["rsa"][k + "|words"] = rsa(RD[k], {"wordemb": LD[nm]})
        if k[0] in "WG": res["rsa"][k + "|words+nowings"] = rsa(RD[k], {"wordemb": LD[nm], "nowings": RD["N_" + (nm if nm != "glorp" else "wing")]})
    res["rsa"]["W_desc|wingsentences"] = rsa(RD["W_desc"], {"wordemb": LD["desc"], "W_wing": RD["W_wing"]})
    np.save(f"{O}/rdm_{J}.npy", np.stack([RD["W_wing"], RD["N_wing"], RD["W_desc"], LD["wing"]]))
    json.dump(res, open(f"{O}/ana_{J}.json", "w"), indent=1)
    for sec, d in res.items():
        print("==", sec)
        for k, v in d.items(): print(f"  {k:24s}", " ".join(f"{a}={b}" if not isinstance(b, float) else f"{a}={b:.2f}" for a, b in v.items()))
if __name__ == "__main__": main()
