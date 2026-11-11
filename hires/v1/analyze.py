import json, numpy as np, sys
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding
import stim as S
L = S.LOCS; n = len(L)
P = np.array([l["xyz"] for l in L]); NRM = np.array([l["n"] for l in L])
DB = squareform(pdist(P)); iu = np.triu_indices(n, 1)
D = json.load(open("rows.json")); rows = D["rows"]; KEEP = D["keep"]
H = np.load("H.npy").astype(np.float32); E = np.load("E.npy")
kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows])
def locmean(X):
    X = X.copy()
    for t in range(4): X[tp == t] -= X[(tp == t) & (kind == "loc")].mean(0)
    A = np.stack([X[(kind == "loc") & (li == i)].mean(0) for i in range(n)])
    W = np.stack([X[(kind == "wf") & (li == i)].mean(0) for i in range(len(S.wordfree()))])
    return A, W
def prep(A, W, k=30):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    p = PCA(k).fit((A - mu) / sd); return p.transform((A - mu) / sd), p.transform((W - mu) / sd), p
def knn_metrics(Z, k=5):
    DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf)
    nnz = np.argsort(DZ, 1)[:, :k]; DBn = DB.copy(); np.fill_diagonal(DBn, np.inf); nnb = np.argsort(DBn, 1)[:, :10]
    cm = np.mean([DB[i, nnz[i]].mean() for i in range(n)])
    ov = np.mean([len(set(nnz[i]) & set(nnb[i])) / k for i in range(n)])
    return cm, ov
def graph_geo(Z, k=6):
    DZ = squareform(pdist(Z)); G = np.zeros_like(DZ); nn = np.argsort(DZ, 1)[:, 1:k+1]
    for i in range(n): G[i, nn[i]] = DZ[i, nn[i]]
    G = np.maximum(G, G.T); return shortest_path(G, directed=False), connected_components(G > 0)[0]
def chains():
    out = {}
    for i, l in enumerate(L):
        ph = l["phrase"]
        if ", " in ph and any(w in ph for w in ["upper arm", "forearm", "thigh", "shin", "calf", "lower leg"]):
            out.setdefault(ph.split(", ")[0], []).append(i)
    return list(out.values())
CH = chains()
def chain_order(Z):  # within each along-limb chain: does the chain's own 1st PC order the points like the body does?
    r = []
    for c in CH:
        Zc = Z[c] - Z[c].mean(0); u = np.linalg.svd(Zc, full_matrices=False)[2][0]
        along = np.linalg.norm(P[c] - P[c][0], axis=1); r.append(abs(spearmanr(Zc @ u, along)[0]))
    return np.mean(r)
def mirror(Z):
    ph = [l["phrase"] for l in L]; pairs = []
    for i, p in enumerate(ph):
        if "left" in p:
            q = p.replace("left", "right")
            if q in ph: pairs.append((i, ph.index(q)))
    pairs = np.array(pairs); d = Z[pairs[:, 0]] - Z[pairs[:, 1]]; md = d.mean(0)
    fr = 1 - ((d - md) ** 2).sum() / (d ** 2).sum()  # share of L-R difference that is one shared shift
    Zs = Z.copy(); Zs[pairs[:, 0]] -= md / 2; Zs[pairs[:, 1]] += md / 2
    DZ = squareform(pdist(Zs)); np.fill_diagonal(DZ, np.inf)
    nnp = np.mean(np.argmin(DZ[pairs[:, 0]], 1) == pairs[:, 1])
    DZ0 = squareform(pdist(Z)); np.fill_diagonal(DZ0, np.inf)
    sameside = np.mean([np.sign(L[j]["side"]) == np.sign(L[i]["side"]) for i in range(n) if L[i]["side"] for j in [np.argmin(DZ0[i])]])
    return fr, nnp, sameside, len(pairs)
def report(name, Z):
    rho = spearmanr(pdist(Z), DB[iu])[0]; cm, ov = knn_metrics(Z); gg, ncc = graph_geo(Z)
    rg = spearmanr(gg[iu], DB[iu])[0] if ncc == 1 else np.nan
    fr, nnp, ss, npair = mirror(Z)
    return dict(name=name, rho=round(rho, 3), nn_cm=round(cm, 1), nn_overlap=round(ov, 3), graph_rho=round(rg, 3), ncc=int(ncc),
                chain=round(chain_order(Z), 3), mirror_shift=round(fr, 3), partner_nn=round(nnp, 3), sameside_nn=round(ss, 3))
if __name__ == "__main__":
    DBn = DB.copy(); np.fill_diagonal(DBn, np.inf)
    print("random-neighbour mean body cm:", DB[iu].mean().round(1), " true 5-NN cm:", np.sort(DBn, 1)[:, :5].mean().round(1), " chance overlap:", round(10 / (n - 1), 3))
    res = []
    A, W = locmean(np.repeat(E, 1, 0)); Z, _, _ = prep(A, W); res.append(report("lexical (input embedding)", Z))
    for j, k in enumerate(KEEP):
        A, W = locmean(H[j]); Z, _, _ = prep(A, W); res.append(report(f"L{k}", Z))
    for r in res: print(r)
    json.dump(res, open("layers.json", "w"))
