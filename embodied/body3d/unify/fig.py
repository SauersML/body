import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
S = json.load(open("$BODY_WORK/body3d/unify/r1/stats.json"))
plt.rcParams.update({"font.size": 22, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(20, 8.5)); a0 = fig.add_axes([0.0, 0.0, 0.42, 1]); a0.axis("off"); a0.set_xlim(0, 10); a0.set_ylim(-2, 10)
a0.text(5, 9.3, "Train both to land on one map", ha="center", fontsize=28, weight="bold")
a0.add_patch(FancyBboxPatch((0.2, 6.3), 4.3, 1.7, boxstyle="round,pad=0.1", fc="#fde8c8", ec="#c77800", lw=2))
a0.text(2.35, 7.15, "Words:\n\"Someone taps\nyour right shin.\"", ha="center", va="center", fontsize=19)
a0.add_patch(FancyBboxPatch((5.5, 6.3), 4.3, 1.7, boxstyle="round,pad=0.1", fc="#d6ecfa", ec="#1f6fb2", lw=2))
a0.text(7.65, 7.15, "Skin sensors:\nprobe presses\nthe right shin", ha="center", va="center", fontsize=19)
a0.add_patch(Ellipse((5, 3.2), 6.5, 3.0, fc="#eeeeee", ec="black", lw=2)); a0.text(5, 1.0, "Qwen's body map (layer 18)", ha="center", fontsize=22, weight="bold")
a0.text(5, 3.0, "right shin", ha="center", fontsize=22, color="#7a1fa2", weight="bold")
for x0 in (2.35, 7.65): a0.annotate("", xy=(5 + (x0 - 5) * 0.2, 3.5), xytext=(x0, 6.2), arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))
a0.text(5, -1.0, "Then: push the map with a direction\nmade from words only, while Qwen moves", ha="center", fontsize=19)
ax = fig.add_axes([0.5, 0.17, 0.48, 0.68])
arms = [("ctrl", "no words"), ("pair", "words +\nsensors\ntogether"), ("align", "words +\nmatching\nloss"), ("both", "together\n+ loss")]
x = np.arange(len(arms)); w = 0.38
for k, (key, lab, col) in enumerate((("trained", "places seen in words", "#7a1fa2"), ("held", "places never seen in words", "#b07cc6"))):
    v = np.array([S[a][f"b1_word_{key}_gain_cm"] for a, _ in arms]); m = v[:, 0]
    ax.bar(x + (k - 0.5) * w, m, w, color=col, yerr=[m - v[:, 1], v[:, 2] - m], capsize=6, error_kw=dict(lw=2))
    ax.text(x[-1] + (k - 0.5) * w, m[-1] + (v[-1, 2] - m[-1]) + 0.8, lab.replace("places ", "").replace(" in words", "\nin words"), ha="center", fontsize=16, color=col, weight="bold")
ax.axhline(0, color="black", lw=1.5); ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms], fontsize=19)
ax.set_ylabel("hand ends closer to the named place\nthan with a random push (cm)", fontsize=21); ax.set_ylim(-6, 36)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("A word-built push moves the real hand\nonly when trained with the matching loss", fontsize=24, weight="bold")
fig.savefig("$BODY_WORK/fig/body_unify_v1.png", dpi=110, bbox_inches="tight")
