import numpy as np, json, collections, re
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
import fstim as S
SP = S.spots(); T = np.array([s["t"] for s in SP]); F = np.array([s["f"] for s in SP]); ARM = np.array([s["arm"] for s in SP])
JD = np.minimum(T, 1 - T)
def dmat(X, metric="correlation"):
    X = X - X.mean(0); return squareform(pdist(X, metric))
def bow(phrases_per_spot):
    vocab = {}; rows = []
    for ps in phrases_per_spot:
        c = collections.Counter(w for p in ps for w in re.findall(r"[a-z0-9\-]+", p.lower())); rows.append(c)
        for w in c: vocab.setdefault(w, len(vocab))
    M = np.zeros((len(rows), len(vocab)))
    for i, c in enumerate(rows):
        for w, n in c.items(): M[i, vocab[w]] = n
    return M
def metrics(D, idx):
    """D: n x n distances for spots idx (one arm)."""
    D = D.copy(); np.fill_diagonal(D, np.inf); t, f, jd = T[idx], F[idx], JD[idx]; n = len(idx)
    nn = np.argmin(D, 1); top3 = np.argsort(D, 1)[:, :3]
    o = {"same_face_nn1": np.mean(f[nn] == f), "same_face_nn3": np.mean(f[top3] == f[:, None])}
    # among same-face others: is the nearest at the opposite end of the arm (cross-end), for end spots
    ce = []; nnc = []
    for i in range(n):
        j = [k for k in range(n) if k != i and f[k] == f[i]]
        k = j[int(np.argmin(D[i, j]))]
        if t[i] <= .25 or t[i] >= .77: ce.append((t[k] <= .25 or t[k] >= .77) and ((t[k] < .5) != (t[i] < .5)))
    o["cross_end_nn"] = np.mean(ce)
    iu = np.triu_indices(n, 1); sf = (f[:, None] == f[None])[iu]; Dm = D.copy(); Dm[np.isinf(Dm)] = 0; d = Dm[iu]
    o["sf_rho_along"] = spearmanr(d[sf], np.abs(t[:, None] - t[None])[iu][sf])[0]
    o["sf_rho_jointdiff"] = spearmanr(d[sf], np.abs(jd[:, None] - jd[None])[iu][sf])[0]
    return o
def chance(idx, nperm=200, rng=np.random.default_rng(0)):
    """shuffled baseline: random distance matrix with labels permuted = metrics on a random symmetric matrix"""
    out = collections.defaultdict(list); n = len(idx)
    for _ in range(nperm):
        A = rng.random((n, n)); A = A + A.T; out_ = metrics(A, idx)
        for k, v in out_.items(): out[k].append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}
def rk(x): return rankdata(x)
def resid(y, Z):
    Z = np.column_stack([np.ones(len(y))] + [rk(z) for z in Z]); b = np.linalg.lstsq(Z, rk(y), rcond=None)[0]; return rk(y) - Z @ b
def partial(y, x, Z): 
    if not Z: return spearmanr(y, x)[0]
    return np.corrcoef(resid(y, Z), resid(x, Z))[0, 1]
def prsa(D, idx, ctrls, nperm=500, rng=np.random.default_rng(1)):
    """Partial Spearman RSA over all pairs. Predictors: side difference, along-arm distance, joint-distance difference.
    ctrls: list of n x n control matrices (lexical). Returns partial r (ctrl only) and partial r (ctrl + other predictors), perm p for the latter."""
    n = len(idx); iu = np.triu_indices(n, 1); t, f, jd = T[idx], F[idx], JD[idx]
    P = {"side": (f[:, None] != f[None]).astype(float), "along": np.abs(t[:, None] - t[None]), "jointdiff": np.abs(jd[:, None] - jd[None])}
    y = D[iu]; C = [c[iu] for c in ctrls]; out = {}
    for k, M in P.items():
        others = [P[o][iu] for o in P if o != k]
        r1 = partial(y, M[iu], C); r2 = partial(y, M[iu], C + others)
        null = []
        for _ in range(nperm):
            p = rng.permutation(n); Mp = M[np.ix_(p, p)][iu]; Op = [P[o][np.ix_(p, p)][iu] for o in P if o != k]
            null.append(partial(y, Mp, C + Op))   # permute the whole design together
        out[k] = dict(r_ctrl=round(float(r1), 3), r_full=round(float(r2), 3), p_full=float((np.sum(np.abs(null) >= abs(r2)) + 1) / (nperm + 1)))
    out["lex_alone"] = [round(float(spearmanr(y, c)[0]), 3) for c in C]
    return out
