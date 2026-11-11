import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import *; import skel
e = np.load("emb_L27.npz"); Z = e["Z"]; iso = np.load("iso2.npz")
G = skel.groups(); E = skel.edges()
views = [("PC0/PC1", Z[:, [0, 1]]), ("PC1/PC2", Z[:, [1, 2]]), ("PC0/PC2", Z[:, [0, 2]]), ("iso10", iso["k10"]), ("iso20", iso["k20"]), ("iso6", iso["k6"])]
fig, ax = plt.subplots(2, 3, figsize=(24, 16))
for a, (t, V) in zip(ax.ravel(), views):
    C = {k: V[v].mean(0) for k, v in G.items()}
    a.scatter(*V.T, c="0.85", s=8)
    for u, w in E:
        c = "tab:blue" if u.endswith("L") or w.endswith("L") else "tab:red" if u.endswith("R") or w.endswith("R") else "k"
        a.plot([C[u][0], C[w][0]], [C[u][1], C[w][1]], c=c, lw=2, alpha=.7)
    for k in ["crown", "mouth", "chest", "pelvis", "palmL", "palmR", "toesL", "toesR", "kneeL", "elbowL", "shoulderL", "hipL", "indexL"]: a.text(*C[k], k, fontsize=11)
    a.set_title(t)
plt.savefig("explore3.png", dpi=50)
