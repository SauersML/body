"""python ana.py TAG -> out/TAG_ana.json + short summary. Tests 1-4."""
import sys, json, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from bodies import *
TAG = sys.argv[1]; Z = np.load(f"out/{TAG}_states.npz", allow_pickle=True)
meta = Z["meta"]; LS = list(Z["layers"]); mb = np.array([m[0] for m in meta]); mp = np.array([int(m[1]) for m in meta]); mt = np.array([int(m[2]) for m in meta])
ALL = BODIES + ["ctrl"]; GEO = {b: xyz(b) for b in BODIES}; rng = np.random.default_rng(0); NB = 1000
def X(l): return Z[f"L{l}"].astype(np.float32)
def pm(A, b, tmask=None):
    m = (mb == b) if tmask is None else (mb == b) & tmask
    return np.stack([A[m & (mp == i)].mean(0) for i in range(16)])
def cosd(P): P = P - P.mean(0); P = P / np.linalg.norm(P, axis=1, keepdims=True); return 1 - P @ P.T
iu = np.triu_indices(16, 1)
def rsa(D, G, idx=None):
    if idx is None: return spearmanr(D[iu], G[iu])[0]
    a, b = np.meshgrid(idx, idx, indexing="ij"); m = np.triu(np.ones((16, 16), bool), 1) & (a != b)
    return spearmanr(D[a, b][m], G[a, b][m])[0]
GD = {b: np.linalg.norm(GEO[b][:, None] - GEO[b][None], axis=-1) for b in BODIES}
BOOT = [rng.integers(0, 16, 16) for _ in range(NB)]
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R = {"layers": [int(l) for l in LS], "rsa": {}, "rsa_ci": {}, "pr": {}, "lr": {}, "part": {}, "reuse": {}, "cent": {}}
for l in LS:
    A = X(l); Ds = {b: cosd(pm(A, b)) for b in BODIES}
    R["rsa"][l] = {b: float(rsa(Ds[b], GD[b])) for b in BODIES}
    bs = {b: np.array([rsa(Ds[b], GD[b], ix) for ix in BOOT]) for b in BODIES}
    R["rsa_ci"][l] = {b: ci(bs[b]) for b in BODIES}
    R["rsa_ci"][l]["human_desc-velk"] = ci(bs["human_desc"] - bs["velk"])
    for b in BODIES:  # participation ratio of part means
        P = pm(A, b); P = P - P.mean(0); s = np.linalg.svd(P, compute_uv=False) ** 2; R["pr"].setdefault(l, {})[b] = float(s.sum() ** 2 / (s ** 2).sum())
    # left/right: leave-one-pair-out on sentences of bilateral parts; part: leave-one-template-out
    for b in BODIES:
        x = GEO[b][:, 0]; nm = names(b); bil = [i for i in range(16) if abs(x[i]) > .04]
        key = lambda i: nm[i].replace("left", "_").replace("right", "_")
        groups = sorted({key(i) for i in bil}); acc = []
        if len(groups) >= 3:
            for g in groups:
                te = [i for i in bil if key(i) == g]; tr = [i for i in bil if key(i) != g]
                mtr = (mb == b) & np.isin(mp, tr); mte = (mb == b) & np.isin(mp, te)
                clf = LogisticRegression(C=.1, max_iter=2000).fit(A[mtr], x[mp[mtr]] > 0); acc.append(clf.score(A[mte], x[mp[mte]] > 0))
            R["lr"].setdefault(l, {})[b] = float(np.mean(acc))
        pa = []
        for t in range(8):
            mtr = (mb == b) & (mt != t); mte = (mb == b) & (mt == t)
            pa.append(LogisticRegression(C=.1, max_iter=2000).fit(A[mtr], mp[mtr]).score(A[mte], mp[mte]))
        R["part"].setdefault(l, {})[b] = float(np.mean(pa))
    # reuse: A's top-5 PCs (templates 0-3) explain B's held-out (templates 4-7) centred part means
    tr, te = mt < 4, mt >= 4; K = 5
    U = {b: np.linalg.svd(pm(A, b, tr) - pm(A, b, tr).mean(0), full_matrices=False)[2][:K] for b in ALL}
    H = {b: pm(A, b, te) - pm(A, b, te).mean(0) for b in ALL}
    R["reuse"][l] = {a: {b: float(((H[b] @ U[a].T) ** 2).sum() / (H[b] ** 2).sum()) for b in ALL} for a in ALL}
    # centroids: cosine distances between body centroids (relative to grand mean of the 8 non-duplicate bodies + ctrl)
    C = {b: A[mb == b].mean(0) for b in ALL}; g = np.mean([C[b] for b in ALL if b != "human_desc"], 0)
    cd = {a: {b: float(1 - np.dot(C[a] - g, C[b] - g) / np.linalg.norm(C[a] - g) / np.linalg.norm(C[b] - g)) for b in ALL} for a in ALL}
    R["cent"][l] = cd
    print(l, " ".join(f"{b[:5]}{R['rsa'][l][b]:.2f}" for b in BODIES), "| LR", " ".join(f"{v:.2f}" for v in R["lr"][l].values()), flush=True)
R["rsa"] = {str(k): v for k, v in R["rsa"].items()}
json.dump(R, open(f"out/{TAG}_ana.json", "w"), default=str)
