import os
import sys, json, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/hires/v1"))
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import RidgeCV
import stim as S
L = S.LOCS; n = len(L); P = np.array([l["xyz"] for l in L], float)
ph = [l["phrase"].replace("{p} ", "") for l in L]; reg = [l["region"] for l in L]
def fold(i):
    p, r = ph[i], reg[i]; s = r[-1] if r[-1] in "LR" else ""
    if r.startswith("arm"): return ("forearm" if "forearm" in p else "upper arm") + s
    if r.startswith("leg"): return ("lower leg" if any(w in p for w in ["shin", "calf", "lower leg"]) else "thigh") + s
    if r.startswith("hand") or r.startswith("foot"): return r
    if r == "head": return "scalp+ears" if any(w in p for w in ["top of head", "back of head", "ear", "side of head"]) else "face"
    if r == "neck": return "scalp+ears"
    if r == "torso": return "back" if any(w in p for w in ["spine", "tailbone", "back", "shoulder blade"]) else "front torso"
F = np.array([fold(i) for i in range(n)]); FOLDS = sorted(set(F))
D = json.load(open("rows.json")); rows = D["rows"]; KEEP = D["keep"]
kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows])
def spotmean(X):
    X = X.astype(np.float32).copy()
    for t in range(4): X[tp == t] -= X[(tp == t) & (kind == "loc")].mean(0)
    return np.stack([X[(kind == "loc") & (li == i)].mean(0) for i in range(n)])
AL = np.logspace(-1, 7, 33)
def ridge_all(Xtr, Ytr, Xte):  # predictions for every alpha via SVD
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd; ym = Ytr.mean(0)
    w, U = np.linalg.eigh(Xtr @ Xtr.T); w = np.clip(w, 0, None); UtY = U.T @ (Ytr - ym); KU = (Xte @ Xtr.T) @ U
    return np.stack([KU @ (UtY / (w + a)[:, None]) + ym for a in AL])
def cv(A, Y):  # outer: leave one body part out; alpha chosen by inner leave-part-out on the training parts only
    pred = np.zeros_like(Y); al = []
    for f in FOLDS:
        te = F == f; tr = ~te; err = np.zeros(len(AL))
        for g in FOLDS:
            if g == f: continue
            itr = tr & (F != g); ite = F == g
            err += (np.linalg.norm(ridge_all(A[itr], Y[itr], A[ite]) - Y[ite], axis=2)).sum(1)
        a = int(np.argmin(err)); al.append(AL[a]); pred[te] = ridge_all(A[tr], Y[tr], A[te])[a]
    return pred, al
DB = squareform(pdist(P)); np.fill_diagonal(DB, np.inf); NNB = np.argsort(DB, 1)[:, :10]
chains = {}
for i, p in enumerate(ph):
    if ", " in p and any(w in p for w in ["upper arm", "forearm", "thigh", "shin", "calf", "lower leg"]): chains.setdefault(p.split(", ")[0], []).append(i)
CH = [c for c in chains.values() if len(c) >= 4]
def metrics(Q):
    err = np.linalg.norm(Q - P, axis=1)
    r = []
    for c in CH:
        ax = P[c[-1]] - P[c[0]]; ax /= np.linalg.norm(ax)
        r.append(spearmanr((Q[c] - P[c[0]]) @ ax, (P[c] - P[c[0]]) @ ax)[0])
    ov = []; ch = []  # neighbours inside each held-out part: do the predicted 5 nearest match the true 5 nearest?
    for f in FOLDS:
        m = np.where(F == f)[0]; k = min(5, len(m) - 2)
        dq = squareform(pdist(Q[m])); db = DB[np.ix_(m, m)].copy(); np.fill_diagonal(dq, np.inf)
        nq = np.argsort(dq, 1)[:, :k]; nb = np.argsort(db, 1)[:, :k]
        ov += [len(set(nq[i]) & set(nb[i])) / k for i in range(len(m))]; ch += [k / (len(m) - 1)] * len(m)
    return dict(med_cm=float(np.median(err)), chain_rho=float(np.nanmean(r)), nn_overlap=float(np.mean(ov)), nn_chance=float(np.mean(ch)))
if __name__ == "__main__":
    H = np.load("H.npy", mmap_mode="r"); E = np.load("E.npy")
    res = {}; preds = {}
    base = np.zeros_like(P)
    for f in FOLDS: base[F == f] = P[F != f].mean(0)
    res["train-mean guess"] = metrics(base)
    AE = spotmean(E); q, a = cv(AE, P); res["word embeddings"] = metrics(q); preds["E"] = q
    rng = np.random.default_rng(0)
    for j, k in enumerate(KEEP):
        if k < 10: continue
        A = spotmean(np.asarray(H[j])); q, a = cv(A, P); res[f"L{k}"] = metrics(q); res[f"L{k}"]["alpha_med"] = float(np.median(a)); preds[f"L{k}"] = q
        if k in (27, 20):
            sh = [metrics(cv(A, P[rng.permutation(n)])[0]) for _ in range(3)]
            res[f"L{k} shuffled coords"] = {kk: float(np.mean([s[kk] for s in sh])) for kk in sh[0]}
        print(k, res[f"L{k}"], flush=True)
    for k, v in res.items(): print(f"{k:22s}", {a: round(b, 3) for a, b in v.items()})
    json.dump(dict(res=res, folds={f: int((F == f).sum()) for f in FOLDS}), open("probe/probe_metrics.json", "w"), indent=1)
    np.savez("probe/preds.npz", F=F, **preds)
