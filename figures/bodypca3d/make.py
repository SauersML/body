import os
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HR = os.path.expandvars("$BODY_REPO/hires/v1"); sys.path.insert(0, HR)
import stim as S, skel
plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]
L = S.LOCS; Z = np.load(HR + "/emb_L27.npz")["Z"]
X = np.c_[Z[:, 1], -Z[:, 4], Z[:, 2]]  # PC2 trunk->tip, PC5 sign-flipped so + = front, PC3 up
COL = {"head": ("#7b4fa8", "#7b4fa8"), "torso": ("#555555", "#555555"), "arm": ("#f5a862", "#c4570c"),
       "hand": ("#f290ab", "#b8204d"), "leg": ("#80c8ee", "#1f6fae"), "foot": ("#92d68c", "#2a7d34")}
def cat(r): return "head" if r in ("head", "neck") else "torso" if r == "torso" else r[:-1]
col = np.array([COL[cat(l["region"])][0 if l["side"] > 0 else 1] for l in L])
# joints: skel.py groups, with each limb shaft merged into a near half and a far half
MER = {"ua1": "uaA", "ua2": "uaA", "ua3": "uaB", "ua4": "uaB", "ua5": "uaB", "fa1": "faA", "fa2": "faA", "fa3": "faA", "fa4": "faB", "fa5": "faB", "fa6": "faB",
       "th1": "thA", "th2": "thA", "th3": "thB", "th4": "thB", "th5": "thB", "sh1": "shA", "sh2": "shA", "sh3": "shA", "sh4": "shB", "sh5": "shB"}
def f(k): return MER[k[:-1]] + k[-1] if k[-1] in "LR" and k[:-1] in MER else k
G = {}
for k, ix in skel.groups().items(): G.setdefault(f(k), []).extend(ix)
C = {k: X[ix].mean(0) for k, ix in G.items()}
E = []
for a, b in skel.edges():
    a, b = f(a), f(b)
    if a != b and (a, b) not in E: E.append((a, b))
def ecol(a, b): return col[G[b if b[-1] in "LR" else a][0]]
LIM = [(X[:, i].min() - .02, X[:, i].max() + .02) for i in range(3)]
def draw(ax, elev, azim, fs=34, ylab=True, dots=40, zoom=1.0):
    ax.scatter(*X.T, c=col, s=dots, alpha=.5, edgecolors="none", depthshade=False)
    for a, b in E:
        ax.plot(*np.c_[C[a], C[b]], color=ecol(a, b), lw=4.5, solid_capstyle="round")
    ax.scatter(*np.array([C[k] for k in C]).T, c=[ecol(k, k) for k in C], s=dots * 2.2, edgecolors="white", linewidths=1.5, depthshade=False, zorder=5)
    ax.view_init(elev=elev, azim=azim); ax.set_proj_type("ortho")
    ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_zlim(*LIM[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("PC2: trunk → fingertip / toe", fontsize=fs, labelpad=-8)
    ax.set_zlabel("PC3: feet → head", fontsize=fs, labelpad=-8)
    if ylab: ax.set_ylabel("PC5: back → front", fontsize=fs, labelpad=-8)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis): p.pane.set_facecolor((.965, .955, .94, 1)); p.pane.set_edgecolor("#ccc")
    ax.set_box_aspect([h - l for l, h in LIM], zoom=zoom)
LEG = [("head / face", COL["head"]), ("torso", COL["torso"]), ("arms", COL["arm"]), ("hands", COL["hand"]), ("legs", COL["leg"]), ("feet", COL["foot"])]
def legend(fig, y, fs, XS=(.03, .215, .35, .49, .645, .78), note_x=.92):
    for x, (n, (a, b)) in zip(XS, LEG):
        if a == b: fig.text(x + .011, y, "●", color=a, fontsize=fs * 1.3, ha="center", va="center")
        else:
            fig.text(x, y, "●", color=a, fontsize=fs * 1.3, ha="center", va="center"); fig.text(x + .022, y, "●", color=b, fontsize=fs * 1.3, ha="center", va="center")
        fig.text(x + .038, y, n, fontsize=fs, va="center")
    fig.text(note_x, y, "light = left\ndark = right", fontsize=fs * .85, va="center", ha="center")
if __name__ == "__main__":
    fig = plt.figure(figsize=(26, 19), dpi=100)
    fig.text(.5, .975, "Qwen3-8B's states for 407 touched body spots lay out a body:\nhead and arms above, legs and feet below, hands at the far end", ha="center", va="top", fontsize=50, weight="bold")
    fig.text(.5, .855, "layer 27, PCA with no body labels  ·  lines join neighbouring joints  ·  left and right land on top of each other", ha="center", va="top", fontsize=31, color="#222")
    ax = fig.add_axes([-.04, .07, .58, .77], projection="3d"); draw(ax, 0, -90, ylab=False, zoom=1.12)
    fig.text(.25, .79, "front view", fontsize=40, ha="center", weight="bold", color="#444")
    ax = fig.add_axes([.415, .05, .56, .79], projection="3d"); draw(ax, 22, -55)
    fig.text(.74, .79, "turned view", fontsize=40, ha="center", weight="bold", color="#444")
    legend(fig, .045, 38)
    fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_pca3d.png"), facecolor="white")
