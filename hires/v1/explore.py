import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import *
e = np.load("emb_L27.npz"); Z = e["Z"]; Y = e["Y"]
reg = np.array([l["region"].rstrip("LR") for l in L]); side = np.array([l["side"] for l in L])
cols = dict(head="k", neck="gray", torso="tab:brown", arm="tab:blue", hand="tab:cyan", leg="tab:red", foot="tab:orange")
DBn = DB.copy(); np.fill_diagonal(DBn, np.inf); nb = np.argsort(DBn, 1)[:, :3]
views = [("PC0", "PC1", Z[:, 0], Z[:, 1]), ("PC1", "PC2", Z[:, 1], Z[:, 2]), ("PC0", "PC2", Z[:, 0], Z[:, 2]), ("iso0", "iso1", Y[:, 0], Y[:, 1]), ("iso1", "iso2", Y[:, 1], Y[:, 2]), ("PC1", "PC4", Z[:, 1], Z[:, 4])]
fig, ax = plt.subplots(2, 3, figsize=(21, 14))
for a, (nx_, ny_, x, y) in zip(ax.ravel(), views):
    for i in range(n):
        if side[i] < 0: continue
        for j in nb[i]:
            if side[j] >= 0: a.plot([x[i], x[j]], [y[i], y[j]], c="0.7", lw=.6)
    for r, c in cols.items():
        m = (reg == r) & (side >= 0); a.scatter(x[m], y[m], c=c, s=14, label=r)
    a.set_title(f"{nx_} vs {ny_}")
ax[0, 0].legend(); plt.savefig("explore.png", dpi=60)
