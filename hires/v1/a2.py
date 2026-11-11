import numpy as np
from analyze import *
Pf = P.copy(); Pf[:, 0] = np.abs(Pf[:, 0]); DF = squareform(pdist(Pf))
def m(Z, DBx):
    DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf); nn = np.argsort(DZ, 1)[:, :5]
    return round(spearmanr(pdist(Z), DBx[iu])[0], 3), round(np.mean([DBx[i, nn[i]].mean() for i in range(n)]), 1)
def variants(A, W):
    out = {}
    Ac = A - A.mean(0)
    out["raw"] = PCA(30).fit_transform(Ac)
    out["z"] = PCA(30).fit_transform(Ac / (A.std(0) + 1e-6))
    An = Ac / np.linalg.norm(Ac, axis=1, keepdims=True); out["cos"] = PCA(30).fit_transform(An)
    out["raw-full"] = Ac
    return out
print("folded random cm", DF[iu].mean().round(1))
A, W = locmean(E)
for k, Z in variants(A, W).items(): print("lex", k, "full", m(Z, DB), "folded", m(Z, DF))
for L_ in [10, 16, 20, 23, 25, 27, 28]:
    j = KEEP.index(L_); A, W = locmean(H[j])
    for k, Z in variants(A, W).items(): print(L_, k, "full", m(Z, DB), "folded", m(Z, DF))
