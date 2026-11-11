import os
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, umap
HR = os.path.expandvars("$BODY_REPO/hires/v1"); sys.path.insert(0, HR)
import stim as S
from matplotlib import font_manager as fm
for f in __import__('glob').glob(os.path.expandvars('$BODY_WORK/fonts/inter/**/*.ttf'), recursive=True): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
L = S.LOCS; Z = np.load(HR + "/emb_L27.npz")["Z"]
COL = {"head": ("#7b4fa8", "#7b4fa8"), "torso": ("#555555", "#555555"), "arm": ("#f5a862", "#c4570c"),
       "hand": ("#f290ab", "#b8204d"), "leg": ("#80c8ee", "#1f6fae"), "foot": ("#92d68c", "#2a7d34")}
def cat(r): return "head" if r in ("head", "neck") else "torso" if r == "torso" else r[:-1]
col = np.array([COL[cat(l["region"])][0 if l["side"] > 0 else 1] for l in L])
fig, axs = plt.subplots(1, 2, figsize=(26, 13), dpi=100, facecolor="#FBF9F5")
for ax, nn, md in zip(axs, (8, 20), (0.15, 0.4)):
    Y = umap.UMAP(n_neighbors=nn, min_dist=md, metric="cosine", random_state=0).fit_transform(Z)
    ax.set_facecolor("#FBF9F5"); ax.scatter(Y[:, 0], Y[:, 1], c=col, s=160, alpha=.85, edgecolors="white", linewidths=.8)
    # label region centroids
    for c in COL:
        ix = [i for i, l in enumerate(L) if cat(l["region"]) == c]
        m = np.median(Y[ix], 0)
        ax.text(m[0], m[1] + .6, {"foot":"feet"}.get(c, c + ("s" if c not in ("head", "torso") else "")), fontsize=34, weight="bold", ha="center", va="bottom", color=COL[c][1],
                bbox=dict(boxstyle="round,pad=.2", fc="#FBF9F5", ec="none", alpha=.8))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"local detail (neighbours {nn})" if nn == 8 else f"overall layout (neighbours {nn})", fontsize=36, loc="left", color="#1d2433")
fig.suptitle("Qwen3-8B's inner body, unrolled with UMAP: body regions form their own islands", fontsize=40, x=.02, ha="left", y=.98, weight="bold", color="#1d2433")
fig.text(.02, .02, "407 touched spots, layer 27, no labels used to build the map; colour = true region, added afterwards.", fontsize=28, color="#555")
plt.tight_layout(rect=(0, .04, 1, .93)); fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_umap.png"), facecolor=fig.get_facecolor())
