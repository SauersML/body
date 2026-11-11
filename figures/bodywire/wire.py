import os
from mesh import *
E = edges(k=4)
dm = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
CUT = np.percentile(dm, 90); E = E[dm < CUT]
side = np.array([l["side"] for l in L])
def wire(ax, elev, azim, sides=(-1, 0, 1), fs=30, zoom=1.1, lw=1.6, al=.55, dots=10, ylab=True):
    keep = [(a, b) for a, b in E if side[a] in sides and side[b] in sides]
    for a, b in keep:
        ax.plot(*np.c_[X[a], X[b]], color=col[a], lw=lw, alpha=al, solid_capstyle="round")
    m = np.isin(side, sides)
    ax.scatter(*X[m].T, c=col[m], s=dots, alpha=.8, edgecolors="none", depthshade=False)
    ax.view_init(elev=elev, azim=azim); ax.set_proj_type("ortho")
    ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_zlim(*LIM[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("PC2: trunk → fingertip / toe", fontsize=fs, labelpad=-8)
    ax.set_zlabel("PC3: feet → head", fontsize=fs, labelpad=-8)
    if ylab: ax.set_ylabel("PC5: back → front", fontsize=fs, labelpad=-8)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis): p.pane.set_facecolor((.965, .955, .94, 1)); p.pane.set_edgecolor("#ccc")
    ax.set_box_aspect([h - l for l, h in LIM], zoom=zoom)
TITLE = "Qwen3-8B's states for 407 touched body spots, joined to their skin neighbours,\nkeep the body's layout but crumple the skin grid inside each body part"
if __name__ == "__main__":
    fig = plt.figure(figsize=(28, 25), dpi=100)
    fig.text(.5, .985, TITLE, ha="center", va="top", fontsize=48, weight="bold")
    fig.text(.5, .905, "layer 27, PCA with no body labels  ·  each line joins two spots that are neighbours on the real skin", ha="center", va="top", fontsize=30, color="#222")
    for i, (sd, nm) in enumerate([((0, 1), "left half\n+ midline"), ((-1, 0), "right half\n+ midline")]):
        y0 = .47 - i * .40
        ax = fig.add_axes([.08, y0, .44, .42], projection="3d"); wire(ax, 0, -90, sd, fs=26, zoom=1.25, ylab=False)
        ax = fig.add_axes([.53, y0, .47, .42], projection="3d"); wire(ax, 22, -55, sd, fs=26, zoom=1.0)
        fig.text(.045, y0 + .21, nm, fontsize=38, ha="center", va="center", weight="bold", color="#444", rotation=90)
    fig.text(.30, .855, "front view", fontsize=38, ha="center", weight="bold", color="#444")
    fig.text(.765, .855, "turned view", fontsize=38, ha="center", weight="bold", color="#444")
    legend(fig, .035, 36)
    fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_wire.png"), facecolor="white")
