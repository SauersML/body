import os
import sys, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/bodywire2"))
import render as R
from render import *
from still_final import legend, px
from PIL import Image
import json
Pt = np.array([l["xyz"] for l in L], float)
Z = np.load(os.path.expandvars("$BODY_REPO/hires/v1/probe/pairout_preds.npz"))
M = json.load(open(os.path.expandvars("$BODY_REPO/hires/v1/probe/pairout_metrics.json")))
def to3(Q): return np.c_[Q[:, 0], -Q[:, 2], Q[:, 1]] / 100.0
C0 = to3(Pt); ctr = (C0.max(0) + C0.min(0)) / 2
SS = 2; W, H = 3300, 2000
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * SS, facecolor="white")
fig.text(.02, .975, "A straight-line read-out of Qwen3-8B's activations puts held-out body parts near their true place;", fontsize=px(66), weight="bold", va="top")
fig.text(.02, .915, "the same read-out from word embeddings scrambles the body", fontsize=px(66), weight="bold", va="top")
fig.text(.02, .845, "Straight-line read-out: activation → 3-D body position (cm).  Each body part (hands, feet, face, forearms …) is left out of fitting, then placed.  Left/right comes from the word.",
         fontsize=px(38), va="top", color="#111")
panels = [("true body", C0, None), ("read out from Qwen3-8B, layer 22", to3(Z["L22f"]), M["L22 folded"]["med_cm"]),
          ("read out from word embeddings", to3(Z["Ef"]), M["E folded"]["med_cm"])]
xl = np.array([-1.05, 1.05]); yl = np.array([-.05, 2.0])
for k, (nm, X3, med) in enumerate(panels):
    R.Xc = X3 - ctr + np.array([0, 0, ctr[2]])
    ax = fig.add_axes([.01 + k * .33, .17, .32, .575]); draw(ax, el=6, az=14, lw=2.4, dot=26, lims=True, fade=.35)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    fig.text(.17 + k * .33, .775, nm, fontsize=px(48), weight="bold", ha="center", va="center", color="#222")
    if med is not None:
        fig.text(.17 + k * .33, .16, f"typical miss  {med:.0f} cm", fontsize=px(56), weight="bold", ha="center", va="center", color="#222")
legend(fig, .045, .02, .105, 46, nx=.0)
fig.savefig("/tmp/bp_ss.png", facecolor="white")
im = Image.open("/tmp/bp_ss.png").convert("RGB").resize((W, H), Image.LANCZOS)
im.save(os.path.expandvars("$BODY_REPO/figures/out/body_probe_map.png"))
