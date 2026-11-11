import numpy as np, collections
from analyze import *
e = np.load("emb_L27.npz"); Z = e["Z"]; ZW = e["ZW"]
reg = np.array([l["region"].rstrip("LR") for l in L])
for d in range(5): print("PC", d, {r: round(Z[reg == r, d].mean(), 2) for r in ["head", "neck", "torso", "arm", "hand", "leg", "foot"]})
DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf); nn = np.argsort(DZ, 1)[:, :5]
R = ["head", "neck", "torso", "arm", "hand", "leg", "foot"]
print("5-NN region confusion (row: point, col: neighbour), fraction")
for r in R:
    c = collections.Counter(reg[nn[reg == r]].ravel()); tot = sum(c.values()); print(f"{r:6s}", {k: round(c[k] / tot, 2) for k in R if c[k]})
# homology: arm<->leg, hand<->foot cross-limb neighbour rate vs chance
lim = np.isin(reg, ["arm", "hand", "leg", "foot"]); isup = np.isin(reg, ["arm", "hand"])
cross = np.mean([np.mean(isup[nn[i]] != isup[i]) for i in np.where(lim)[0]]); print("limb points: 5-NN from the other limb pair (arm<->leg)", round(cross, 3))
side = np.array([l["side"] for l in L])
print("5-NN on the other side (sided points)", round(np.mean([np.mean(side[nn[i]] == -side[i]) for i in np.where(side != 0)[0]]), 3))
# word-free with style offset removed
ZWc = ZW - ZW.mean(0) + Z.mean(0)
Pf = P.copy(); Pf[:, 0] = abs(Pf[:, 0]); DF = squareform(pdist(Pf)); r = []
for i, w in enumerate(S.wordfree()):
    d = np.linalg.norm(Z - ZWc[i], axis=1); o = np.argsort(d); t = w["target"]; r.append((np.where(o == t)[0][0], DF[o[:5], t].mean()))
r = np.array(r); print("wordfree centered: median rank", np.median(r[:, 0]), "median folded cm of 5-NN", np.median(r[:, 1]).round(1), "rand", DF[iu].mean().round(1))
