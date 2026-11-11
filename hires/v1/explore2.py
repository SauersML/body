import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import *
e = np.load("emb_L27.npz"); Z = e["Z"]
reg = np.array([l["region"].rstrip("LR") for l in L]); side = np.array([l["side"] for l in L])
cols = dict(head="k", neck="gray", torso="tab:brown", arm="tab:blue", hand="tab:cyan", leg="tab:red", foot="tab:orange")
Dm = DB.copy(); Dm[(NRM @ NRM.T) < 0.3] = np.inf; Dm[side[:, None] * side[None, :] < 0] = np.inf; np.fill_diagonal(Dm, np.inf)
nb = np.argsort(Dm, 1)[:, :2]
isos = {k: Isomap(n_neighbors=k, n_components=2).fit_transform(Z) for k in [6, 10, 20]}
views = [("PC1 vs PC2", Z[:, 1], Z[:, 2]), ("iso k6", *isos[6].T), ("iso k10", *isos[10].T), ("iso k20", *isos[20].T)]
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
for a, (t, x, y) in zip(ax.ravel(), views):
    for i in range(n):
        if side[i] < 0: continue
        for j in nb[i]:
            if np.isfinite(Dm[i, j]) and side[j] >= 0: a.plot([x[i], x[j]], [y[i], y[j]], c="0.6", lw=.8)
    for r, c in cols.items():
        m = (reg == r) & (side >= 0); a.scatter(x[m], y[m], c=c, s=18, label=r)
    a.set_title(t)
ax[0, 0].legend(); plt.savefig("explore2.png", dpi=55)
np.savez("iso2.npz", **{f"k{k}": v for k, v in isos.items()})
