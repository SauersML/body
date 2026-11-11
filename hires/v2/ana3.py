"""Label-free reliable subspace (split-half cross-covariance over framings), crossnobis distances, MDS / Isomap / diffusion map; local patches."""
import numpy as np, json, sys
from ana2 import *
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rng = np.random.default_rng(0)
S = load_stim("stim.py"); b = Body(S); probes = S.probes(); n = b.n
S1 = load_stim("../hires/stim.py"); b1 = Body(S1); p2 = [l["phrase"] for l in S.LOCS]; IDX = [p2.index(l["phrase"]) for l in S1.LOCS]; pr1 = [dict(p, kind="wf") for p in S1.wordfree()]
def load(dirn, rd, L):
    D = json.load(open(f"{dirn}/rows.json")); rows = D["rows"]
    X = np.asarray(np.load(f"{dirn}/{rd}.npy", mmap_mode="r")[D["keep"].index(L)], dtype=np.float32); return X, rows
def prep(X, rows):
    kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows]); T = tp.max() + 1
    X = X.copy()
    for t in range(T): X[tp == t] -= X[(tp == t) & (kind == "loc")].mean(0)
    Y = np.zeros((n, T, X.shape[1]), np.float32); Y[li[kind == "loc"], tp[kind == "loc"]] = X[kind == "loc"]
    npb = len(probes); Q = np.zeros((npb, T, X.shape[1]), np.float32); Q[li[kind != "loc"], tp[kind != "loc"]] = X[kind != "loc"]
    mu = Y.mean((0, 1)); Y = Y - mu; Q = Q - mu
    from sklearn.utils.extmath import randomized_svd
    Vt = randomized_svd(Y.reshape(-1, Y.shape[2]), 800, random_state=0)[2]  # lossless enough: rows' top-800 basis, no labels
    return Y @ Vt.T, Q @ Vt.T  # spots x framings x 800
def splits(T, k=20): return [rng.permutation(T) for _ in range(k)]
def reliable(Y, sp, nperm=20):
    T = Y.shape[1]; d = Y.shape[2]; C = np.zeros((d, d))
    for p in sp:
        a, c = Y[:, p[:T // 2]].mean(1), Y[:, p[T // 2:]].mean(1); C += (a.T @ c + c.T @ a) / 2
    C /= len(sp); ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
    null = []
    for i in range(nperm):
        p = sp[i % len(sp)]; a, c = Y[:, p[:T // 2]].mean(1), Y[rng.permutation(n)][:, p[T // 2:]].mean(1)
        null.append(np.linalg.eigvalsh((a.T @ c + c.T @ a) / 2).max())
    k = int((ev > np.percentile(null, 95)).sum())
    # held-out choice of k: fit directions on half the framings, keep components whose split-half r on the other half > 0.5
    F = rng.permutation(T); F1, F2 = F[:T // 2], F[T // 2:]; C1 = np.zeros((d, d))
    for _ in range(20):
        q = F1[rng.permutation(len(F1))]; a, c = Y[:, q[:len(q) // 2]].mean(1), Y[:, q[len(q) // 2:]].mean(1); C1 += (a.T @ c + c.T @ a) / 2
    V1 = np.linalg.eigh(C1)[1][:, ::-1]; r = []
    for j in range(300):
        q = F2[rng.permutation(len(F2))]; r.append(np.corrcoef(Y[:, q[:len(q) // 2]].mean(1) @ V1[:, j], Y[:, q[len(q) // 2:]].mean(1) @ V1[:, j])[0, 1])
    r = np.array(r); kh = int((r > .5).sum())
    print("reliable dims: null-max test", k, "held-out r>0.5", kh, flush=True); return V[:, :kh], ev, kh
def crossnobis(Y, sp, V):
    U = Y @ V  # n x T x k
    R = (U - U.mean(1, keepdims=True)).reshape(-1, U.shape[2]); Sg = np.cov(R.T); Sg = 0.8 * Sg + 0.2 * np.eye(len(Sg)) * np.trace(Sg) / len(Sg)
    w, E = np.linalg.eigh(Sg); Wh = E @ np.diag(w ** -.5) @ E.T; U = U @ Wh; T = U.shape[1]; D = np.zeros((n, n))
    for p in sp:
        a, c = U[:, p[:T // 2]].mean(1), U[:, p[T // 2:]].mean(1); G = a @ c.T; g = np.diag(G); D += g[:, None] + g[None] - G - G.T
    return D / len(sp), Wh
def cmds(D, k=30):
    J = np.eye(len(D)) - 1 / len(D); B = -0.5 * J @ np.clip(D, 0, None) @ J; w, E = np.linalg.eigh(B); o = np.argsort(-w)[:k]; return E[:, o] * np.sqrt(np.clip(w[o], 0, None))
def score(Z, ZW=None):
    m = b.metrics(Z, ZW, probes) if ZW is not None else b.metrics(Z)
    m.update({"v1set_" + k: v for k, v in (b1.metrics(Z[IDX], ZW[:45], pr1) if ZW is not None else b1.metrics(Z[IDX])).items()}); return m
def patches():
    L = S.LOCS; P = {}
    P["forearm"] = [i for i, l in enumerate(L) if "left forearm" in l["phrase"] or "left wrist" in l["phrase"]]
    P["palm"] = [i for i, l in enumerate(L) if l["region"] == "handL" and l["n"][2] > 0.5]
    P["face"] = [i for i, l in enumerate(L) if l["region"] == "head" and l["n"][2] > 0.5 and 150 < l["xyz"][1] < 173]
    return P
def local(Dm, name):
    out = {}
    for pn, ix in patches().items():
        Dp = Dm[np.ix_(ix, ix)]; Ds = squareform(pdist(b.P[ix])); iu = np.triu_indices(len(ix), 1)
        near = Ds[iu] < np.percentile(Ds[iu], 25)
        Zp = cmds(Dp, 10); sil = max(silhouette_score(Zp, KMeans(k, n_init=10, random_state=0).fit_predict(Zp)) for k in range(2, 7))
        nn = np.argsort(Dp + np.eye(len(ix)) * 1e9, 1)[:, :3]; nncm = np.mean([Ds[i, nn[i]].mean() for i in range(len(ix))]); rnd = Ds[iu].mean()
        out[pn] = dict(n=len(ix), rho=spearmanr(Dp[iu], Ds[iu])[0], rho_near=spearmanr(Dp[iu][near], Ds[iu][near])[0], nn3_cm=nncm, rand_cm=rnd, best_silhouette=sil)
    return out
if __name__ == "__main__":
    dirn, rd, L = sys.argv[1], sys.argv[2], int(sys.argv[3]); tag = sys.argv[4]
    X, rows = load(dirn, rd, L); Y, Q = prep(X, rows); del X
    sp = splits(Y.shape[1]); V, ev, k = reliable(Y, sp)
    A = Y.mean(1); W = Q.mean(1)
    rel = [0]  # replaced by held-out test
    _ = [np.corrcoef((Y[:, p[:13]].mean(1) @ V[:, j]), (Y[:, p[13:]].mean(1) @ V[:, j]))[0, 1] for j in range(min(k, 40)) for p in sp[:1]]
    print(tag, "reliable dims", k, "split-half r of first 10:", np.round(rel[:10], 2), "last:", np.round(rel[-3:], 2), flush=True)
    R = {}; LOC = {}
    Dplain = squareform(pdist(A)) ** 2
    Z, ZW = embed(A, W); R["PCA-30 (v1 pipeline)"] = score(Z, ZW); LOC["plain"] = local(Dplain, "plain")
    ZR, ZWR = A @ V, W @ V; R["reliable subspace"] = score(ZR, ZWR); LOC["reliable"] = local(squareform(pdist(ZR)) ** 2, "rel")
    Dx, Wh = crossnobis(Y, sp, V); Zx = cmds(Dx, 30); R["crossnobis + MDS"] = score(Zx); LOC["crossnobis"] = local(Dx, "x")
    # probe ranks in noise-normalised reliable space
    ZRw, ZWRw = A @ V @ Wh, W @ V @ Wh; R["reliable, noise-whitened"] = score(ZRw, ZWRw)
    Dfull, _ = crossnobis(Y, sp, np.linalg.svd(A.T, full_matrices=False)[0][:, :min(200, n - 1)]); R["crossnobis, top-200 PCs, no subspace"] = score(cmds(Dfull, 30))
    for nb in [10]:
        iso = Isomap(n_neighbors=nb, n_components=10).fit(ZRw); R[f"Isomap k={nb}"] = score(iso.transform(ZRw), iso.transform(ZWRw))
        R[f"diffusion map k={nb}"] = score(SpectralEmbedding(n_components=10, n_neighbors=nb, random_state=0).fit_transform(ZRw))
    json.dump(dict(k=int(k), ev=ev[:100].tolist(), rel=rel, R=R, LOC=LOC), open(f"ana3_{tag}.json", "w"), indent=0)
    np.savez(f"emb3_{tag}.npz", ZRu=ZR, ZWu=ZWR, ZR=ZRw, ZW=ZWRw, Zx=Zx, Dx=Dx)
    C = ["v1set_nn5_cm", "v1set_chain_rho", "v1set_shaft_pair_rho", "v1set_partner_nn", "v1set_probe_wf_rank", "nn5_cm", "chain_rho", "shaft_pair_rho", "probe_alt_rank"]
    print("method".ljust(38), " ".join(c.replace("v1set_", "1:")[:8].rjust(8) for c in C))
    for kk, v in R.items(): print(kk.ljust(38), " ".join(f"{v.get(c, float('nan')):8.3g}" for c in C))
    for kk, v in LOC.items(): print(kk, {p: {a: round(float(x), 2) for a, x in d.items()} for p, d in v.items()})
