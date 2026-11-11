import numpy as np, json
from analyze import *; import skel
Pf = P.copy(); Pf[:, 0] = abs(Pf[:, 0]); DF = squareform(pdist(Pf)); side = np.array([l["side"] for l in L])
G = skel.groups()
def cosZ(A):
    Ac = A - A.mean(0); Ac /= np.linalg.norm(Ac, axis=1, keepdims=True); return PCA(30).fit_transform(Ac)
def mets(Z):
    DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf); nn = np.argsort(DZ, 1)[:, :5]
    C = {k: Z[v].mean(0) for k, v in G.items()}
    ch = []
    for s in "LR":
        for seq in [["shoulder", "ua1", "ua2", "ua3", "ua4", "ua5", "elbow", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "wrist", "palm"],
                    ["hip", "th1", "th2", "th3", "th4", "th5", "knee", "sh1", "sh2", "sh3", "sh4", "sh5", "ankle", "foot", "toes"]]:
            d = [np.linalg.norm(C[k + s] - C[seq[0] + s]) for k in seq]; ch.append(spearmanr(d, range(len(seq)))[0])
    return dict(nn5_cm_folded=round(np.mean([DF[i, nn[i]].mean() for i in range(n)]), 1), nn5_cm=round(np.mean([DB[i, nn[i]].mean() for i in range(n)]), 1),
                rho_folded=round(spearmanr(pdist(Z), DF[iu])[0], 3), other_side_nn=round(np.mean([np.mean(side[nn[i]] == -side[i]) for i in np.where(side != 0)[0]]), 3),
                limb_chain_rho=round(np.mean(ch), 3))
out = {"lexical": mets(cosZ(locmean(E)[0]))}
for j, k in enumerate(KEEP): out[f"L{k}"] = mets(cosZ(locmean(H[j])[0]))
for k, v in out.items(): print(k, v)
print("random cm folded", DF[iu].mean().round(1), "full", DB[iu].mean().round(1))
json.dump(out, open("final_metrics.json", "w"), indent=1)
