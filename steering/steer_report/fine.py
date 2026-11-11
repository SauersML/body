"""B: hidden clean axes? Held-out fine position inside forearm / hand / face, strips of spots left out (both sides together).
Readouts: ridge (alpha by inner strip CV), kNN, small MLP. Layer states (L22-27) vs input word embeddings (E)."""
import os
import sys, os, json, numpy as np
H0 = os.path.expandvars("$BODY_REPO/hires/v1"); sys.path.insert(0, H0)
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import stim as S, warnings; warnings.filterwarnings('ignore')
L = S.LOCS; n = len(L); P = np.array([l["xyz"] for l in L], float); ph = [l["phrase"] for l in L]
D = json.load(open(f"{H0}/rows.json")); rows = D["rows"]; KEEP = D["keep"]
kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows])
def spotmean(X):
    X = X.astype(np.float32).copy()
    for t in range(4): X[tp == t] -= X[(tp == t) & (kind == "loc")].mean(0)
    return np.stack([X[(kind == "loc") & (li == i)].mean(0) for i in range(n)])
AL = np.logspace(-1, 7, 33)
def ridge_all(Xtr, Ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd; ym = Ytr.mean(0)
    w, U = np.linalg.eigh(Xtr @ Xtr.T); w = np.clip(w, 0, None); UtY = U.T @ (Ytr - ym); KU = (Xte @ Xtr.T) @ U
    return np.stack([KU @ (UtY / (w + a)[:, None]) + ym for a in AL])
# ---- part definitions: spot index, target coordinates (cm), strip id
FW = ["just below the elbow", "a hand-width below the elbow", "halfway between the elbow and the wrist", "a hand-width above the wrist", "two finger-widths above the wrist", "just above the wrist"]
FA = {"inside": 0, "thumb side": 90, "back": 180, "little-finger side": 270}
fi = [i for i in range(n) if "forearm" in ph[i]]
f_row = np.array([[k for k, w in enumerate(FW) if ph[i].endswith(w)][0] for i in fi])
f_face = np.array([[k for k in FA if f"the {k} of" in ph[i]][0] for i in fi])
f_t = np.abs(P[fi, 0]) - 50  # cm from elbow
f_ang = np.deg2rad([FA[k] for k in f_face])
hi = [i for i in range(n) if L[i]["region"].startswith("hand")]
h_xy = np.c_[np.abs(P[hi, 0]) - 76, P[hi, 1] - 141]  # cm from wrist along the hand, cm across (thumb +)
km = KMeans(6, n_init=20, random_state=0).fit(h_xy); h_blk = km.labels_
def isface(i):
    p = ph[i]; return L[i]["region"] == "head" and not any(w in p for w in ["top of {p} head", "back of {p} head", "ear", "side of {p} head", "crown", "scalp"])
ei = [i for i in range(n) if isface(i)]
e_xyz = np.c_[np.abs(P[ei, 0]), P[ei, 1], P[ei, 2]]
e_blk = KMeans(6, n_init=20, random_state=0).fit(e_xyz[:, :2]).labels_
def shaft(word, alongf):
    idx = [i for i in range(n) if word(ph[i])]; a = np.array([alongf(P[i]) for i in idx]); r = np.unique(np.round(a, 1), return_inverse=True)[1]
    return idx, a, r
SH = {"upper arm": (lambda p: "upper arm," in p, lambda q: abs(q[0]) - 22), "forearm": (lambda p: "forearm," in p, lambda q: abs(q[0]) - 50),
      "thigh": (lambda p: "thigh," in p, lambda q: 88 - q[1]), "lower leg": (lambda p: any(w in p for w in ["shin,", "calf,", "lower leg,"]), lambda q: 50 - q[1])}
TASKS = {}
for nm, (w, af) in SH.items():
    idx, a, r = shaft(w, af); TASKS[f"{nm} along (cm)"] = (idx, a[:, None], r, "lin")
    if nm == "forearm":  # around-limb angle, same ring strips held out
        TASKS["forearm around (face angle)"] = (idx, np.c_[np.cos(f_ang), np.sin(f_ang)], r, "ang")
LEN = {"upper arm": 28, "forearm": 26, "thigh": 38, "lower leg": 42}; pi, pf, pj, pr_ = [], [], [], []
for k, (nm, (w, af)) in enumerate(SH.items()):
    idx, a, r = shaft(w, af); pi += idx; pf += list(a / LEN[nm]); pj += list(np.minimum(a, LEN[nm] - a) / LEN[nm]); pr_ += list(r + 10 * k)
TASKS["all shafts along (fraction)"] = (pi, np.array(pf)[:, None], np.array(pr_), "lin")
TASKS["all shafts joint distance (fraction)"] = (pi, np.array(pj)[:, None], np.array(pr_), "lin")
TASKS["hand (2-D position)"] = (hi, h_xy, h_blk, "lin"); TASKS["face (3-D position)"] = (ei, e_xyz, e_blk, "lin")
def ridge_cv(X, Y, G):
    pred = np.zeros_like(Y)
    for g in np.unique(G):
        te = G == g; tr = ~te; err = np.zeros(len(AL))
        for h in np.unique(G[tr]):
            itr = tr & (G != h); ite = G == h
            err += np.linalg.norm(ridge_all(X[itr], Y[itr], X[ite]) - Y[ite], axis=2).sum(1)
        pred[te] = ridge_all(X[tr], Y[tr], X[te])[int(np.argmin(err))]
    return pred
def other_cv(X, Y, G, how):
    pred = np.zeros_like(Y)
    for g in np.unique(G):
        te = G == g; tr = ~te
        Z = PCA(min(20, tr.sum() - 1)).fit(X[tr]); A, B = Z.transform(X[tr]), Z.transform(X[te]); s = A.std(0) + 1e-6; A, B = A / s, B / s
        if how == "knn": m = KNeighborsRegressor(3, metric="cosine")
        else: m = MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, max_iter=3000, random_state=0)
        ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        yt = (Y[tr] - ym) / ys; m.fit(A, yt.ravel() if yt.shape[1] == 1 else yt); pred[te] = m.predict(B).reshape(-1, Y.shape[1]) * ys + ym
    return pred
def score(Y, pr, G, k):
    if k == "ang":
        a = np.arctan2(pr[:, 1], pr[:, 0]); t = np.arctan2(Y[:, 1], Y[:, 0]); e = np.rad2deg(np.abs(np.angle(np.exp(1j * (a - t)))))
        return dict(err=float(np.mean(e)), base=90.0, rel=float(np.mean(e) / 90))
    base = np.zeros_like(Y)
    for g in np.unique(G): base[G == g] = Y[G != g].mean(0)
    e = np.linalg.norm(pr - Y, axis=1); b = np.linalg.norm(base - Y, axis=1)
    out = dict(err=float(np.median(e)), base=float(np.median(b)), rel=float(np.mean(e) / np.mean(b)),
               r2=float(1 - ((pr - Y) ** 2).sum() / ((base - Y) ** 2).sum()))
    if Y.shape[1] == 1: out["rho"] = float(spearmanr(pr[:, 0], Y[:, 0])[0])
    else:  # held-out geometry: rank corr of predicted vs true pairwise distances
        iu = np.triu_indices(len(Y), 1); dp = np.linalg.norm(pr[:, None] - pr[None], axis=-1)[iu]; dt = np.linalg.norm(Y[:, None] - Y[None], axis=-1)[iu]
        out["rho"] = float(spearmanr(dp, dt)[0])
    return out
if __name__ == "__main__":
    Hm = np.load(f"{H0}/H.npy", mmap_mode="r"); E = np.load(f"{H0}/E.npy")
    feats = {"words": spotmean(E)}
    for lay in (22, 24, 27): feats[f"L{lay}"] = spotmean(np.asarray(Hm[KEEP.index(lay)]))
    res = {}
    for t, (idx, Y, G, k) in TASKS.items():
        if os.environ.get('ONLY') and os.environ['ONLY'] not in t: continue
        for f, X in feats.items():
            Xs = X[idx]
            for how in ("ridge", "knn", "mlp"):
                pr = ridge_cv(Xs, Y, G) if how == "ridge" else other_cv(Xs, Y, G, how)
                r = score(Y, pr, G, k); res[f"{t}|{f}|{how}"] = r
                print(f"{t:32s} {f:6s} {how:6s}", {a: round(b, 3) for a, b in r.items()}, flush=True)
                if how == "ridge" and f in ("L24", "words"): np.save(f"pred_{t.replace(' ', '')[:12]}_{f}.npy", pr)
    json.dump(res, open(os.environ.get("FM", "fine_metrics.json"), "w"), indent=1)
    np.savez("fine_parts.npz", fi=fi, f_t=f_t, f_face=f_face, f_row=f_row, hi=hi, h_xy=h_xy, h_blk=h_blk, ei=ei, e_xyz=e_xyz, e_blk=e_blk)
