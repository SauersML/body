import numpy as np, collections
from analyze import *
from sklearn.cluster import AgglomerativeClustering
Pf = P.copy(); Pf[:, 0] = np.abs(Pf[:, 0]); DF = squareform(pdist(Pf))
reg = np.array([l["region"].rstrip("LR") if l["region"] not in ("head",) else "head" for l in L])
side = np.array([l["side"] for l in L]); distal = np.linalg.norm(P - [0, 120, 0], axis=1)
feats = dict(left_right=P[:, 0], lateral=np.abs(P[:, 0]), up_down=P[:, 1], front_back=P[:, 2], normal_front=NRM[:, 2], distal=distal)
LAY = int(sys.argv[1]) if len(sys.argv) > 1 else 27
j = KEEP.index(LAY); A, Wf = locmean(H[j])
mu = A.mean(0); An = (A - mu); An /= np.linalg.norm(An, axis=1, keepdims=True)
Wn = Wf - mu; Wn /= np.linalg.norm(Wn, axis=1, keepdims=True)
pca = PCA(30).fit(An); Z = pca.transform(An); ZW = pca.transform(Wn)
print("PCA var top10", pca.explained_variance_ratio_[:10].round(3))
def axes(Y, nm):
    for d in range(Y.shape[1]):
        c = {k: round(spearmanr(Y[:, d], v)[0], 2) for k, v in feats.items()}
        best = max(c, key=lambda k: abs(c[k])); print(nm, d, "best:", best, c[best], c)
axes(Z[:, :8], "PC")
iso = Isomap(n_neighbors=10, n_components=3).fit(Z); Y = iso.transform(Z); axes(Y, "ISO")
sp = SpectralEmbedding(n_components=3, n_neighbors=10, random_state=0).fit_transform(Z); axes(sp, "DIFF")
for nm, Q in [("pca30", Z), ("iso3", Y), ("spec3", sp), ("pca3", Z[:, :3])]:
    DZ = squareform(pdist(Q)); np.fill_diagonal(DZ, np.inf); nn = np.argsort(DZ, 1)[:, :5]
    print(nm, "rho full", round(spearmanr(pdist(Q), DB[iu])[0], 3), "folded", round(spearmanr(pdist(Q), DF[iu])[0], 3),
          "nn cm full", round(np.mean([DB[i, nn[i]].mean() for i in range(n)]), 1), "folded", round(np.mean([DF[i, nn[i]].mean() for i in range(n)]), 1))
# topology: model kNN graph geodesic vs folded body distance
gg, ncc = graph_geo(Z, 8); print("graph comps", ncc, "graph-geo rho folded", round(spearmanr(gg[iu], DF[iu])[0], 3) if ncc == 1 else None)
# nearest neighbour region confusion
DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf); nn1 = DZ.argmin(1)
print("NN same region", np.mean(reg[nn1] == reg), "NN same side", np.mean(side[nn1] == side), "NN mirror-or-same within 10cm folded", np.mean(DF[np.arange(n), nn1] < 10))
# clusters
for k in [8, 16]:
    cl = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    print("k", k); [print("  ", c, collections.Counter(reg[cl == c]).most_common(3), [L[i]["phrase"][:40] for i in np.where(cl == c)[0][:3]]) for c in range(k)]
# mirror
print("mirror (shift frac, partner NN after shift, same-side NN)", mirror(Z))
# word-free probes
wf = S.wordfree(); r = []
for i, w in enumerate(wf):
    d = np.linalg.norm(Z - ZW[i], axis=1); o = np.argsort(d); t = w["target"]
    r.append((np.where(o == t)[0][0], DB[o[0], t], DF[o[0], t]))
    print(f"{w['phrase'][:60]:60s} -> {L[o[0]]['phrase'][:55]:55s} rank {r[-1][0]}  {DF[o[0], t]:.0f}cm")
r = np.array(r); print("wordfree median rank", np.median(r[:, 0]), "median folded cm", np.median(r[:, 2]), "full", np.median(r[:, 1]), "top10 frac", np.mean(r[:, 0] < 10))
np.savez(f"emb_L{LAY}.npz", Z=Z, Y=Y, sp=sp, ZW=ZW)
