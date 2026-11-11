"""Per-region local maps in the reliable subspace: framing count, local vs global subspace, dense vs v1 grid."""
import numpy as np, json, sys
import ana3 as a
from ana2 import *
from sklearn.manifold import Isomap
from scipy.linalg import orthogonal_procrustes
dirn, rd, Lyr, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
X, rows = a.load(dirn, rd, Lyr); Y, Q = a.prep(X, rows); del X
V, _, k = a.reliable(Y, a.splits(Y.shape[1])); U = Y @ V; del Y  # spots x framings x k
Ls = a.S.LOCS; ph = [l["phrase"] for l in Ls]; P = a.b.P; v1 = set(l["phrase"] for l in a.S1.LOCS)
def reg(side):
    S_ = side
    return {f"forearm {S_}": [i for i, p in enumerate(ph) if f"{S_} forearm" in p or f"{S_} wrist" in p],
            f"shin {S_}": [i for i, p in enumerate(ph) if any(f"{S_} {w}" in p for w in ["shin", "calf", "lower leg"]) or (f"{S_} ankle" in p and "bump" not in p)],
            f"hand {S_}": [i for i, l in enumerate(Ls) if l["region"] == "hand" + ("L" if S_ == "left" else "R") and "wrist" not in l["phrase"]]}
R = {**reg("left"), **reg("right"), "face": [i for i, l in enumerate(Ls) if l["region"] == "head" and l["n"][2] > 0.4 and 150 < l["xyz"][1] < 173]}
def flat(name, ix):  # real skin layout in 2-D, display only
    p = P[ix]
    if "forearm" in name or "shin" in name:
        ax = 0 if "forearm" in name else 1; c = p.mean(0); q = p - c; along = q[:, ax]
        o = [d for d in range(3) if d != ax]; ang = np.arctan2(q[:, o[1]], q[:, o[0]]); return np.stack([along, ang * 5], 1)
    if "hand" in name: return np.stack([p[:, 0] + np.where(np.array([Ls[i]["n"][2] for i in ix]) < 0, 0, 0), p[:, 1] + 12 * (np.array([Ls[i]["n"][2] for i in ix]) < 0)], 1)
    return p[:, :2]
def fid(Zr, ix):
    Ds = squareform(pdist(P[ix])); Dz = squareform(pdist(Zr)); iu = np.triu_indices(len(ix), 1); np.fill_diagonal(Dz, np.inf)
    nn = np.argsort(Dz, 1)[:, :3]; return spearmanr(pdist(Zr), Ds[iu])[0], np.mean([Ds[i, nn[i]].mean() for i in range(len(ix))]), Ds[iu].mean()
def localmap(M, kk=5): Mc = M - M.mean(0); return np.linalg.svd(Mc, full_matrices=False)[0][:, :kk] * np.linalg.svd(Mc, full_matrices=False)[1][:kk]
rng = np.random.default_rng(1); T = U.shape[1]; out = {}; maps = {}
G = U.mean(1)
for name, ix in R.items():
    o = {"n": len(ix)}
    o["global"] = fid(G[ix], ix); o["local_pca5"] = fid(localmap(G[ix]), ix); o["local_pca2"] = fid(localmap(G[ix], 2), ix)
    try: o["local_isomap2"] = fid(Isomap(n_neighbors=6, n_components=2).fit_transform(G[ix]), ix)
    except Exception as e: pass
    for nf in [2, 4, 8, 13, 26]:
        o[f"nf{nf}"] = np.mean([fid(localmap(U[ix][:, rng.permutation(T)[:nf]].mean(1)), ix)[0] for _ in range(6)], 0).tolist()
    sub = [j for j, i in enumerate(ix) if ph[i] in v1]; ixs = [ix[j] for j in sub]
    o["v1grid_local_pca5"] = fid(localmap(G[ixs]), ixs); o["v1grid_n"] = len(ixs)
    # split-half reliability of the local distance pattern (ceiling): framings halves
    p = rng.permutation(T); o["local_split_half_r"] = spearmanr(pdist(G[ix] * 0 + U[ix][:, p[:T // 2]].mean(1)), pdist(U[ix][:, p[T // 2:]].mean(1)))[0]
    F = flat(name, ix); Z2 = localmap(G[ix], 2); Z2 = (Z2 - Z2.mean(0)); Fc = F - F.mean(0)
    Rm, _ = orthogonal_procrustes(Z2, Fc); Z2 = Z2 @ Rm; Z2 *= np.linalg.norm(Fc) / np.linalg.norm(Z2)
    maps[name] = dict(ix=ix, flat=F.tolist(), model=(Z2 + F.mean(0)).tolist())
    out[name] = {kk: (np.round(v, 3).tolist() if isinstance(v, (tuple, list, np.ndarray)) else v) for kk, v in o.items()}
    print(name, {kk: out[name][kk] for kk in ["n", "global", "local_pca5", "local_isomap2", "nf2", "nf26", "v1grid_local_pca5", "local_split_half_r"] if kk in out[name]}, flush=True)
json.dump(dict(k=k, out=out, maps=maps), open(f"local_{tag}.json", "w"))
