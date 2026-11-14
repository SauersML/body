import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 30})
S = json.load(open("summary3.json")); S4 = json.load(open("summary4.json"))
GHOST = {m: S4["Qwen3-" + m]["ts"]["self"] for m in ("8B", "14B", "32B")}
MODELS = [m for m in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"] if f"Qwen3-{m}" in S]
V, F = [], []
for l in open("$BODY_WORK/fig/human/neutral.obj"):
    if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
V = np.array(V); F = np.array(F); V[:, 1] -= V[:, 1].min(); Hm = V[:, 1].max()
BLUE, MAN, WOM = "#34557F", "#2E8B6E", "#C2552E"
LIGHT = "#E3A04F"
def body(ax, x0, h, col=BLUE, z=5):
    P = V[:, :2] * (h / Hm); P[:, 0] += x0
    ax.add_collection(PolyCollection(P[F], facecolors=col, edgecolors=col, linewidths=0.3, zorder=z))
W, H = 28, 17.5; fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor("white")
axm = fig.add_axes([0.03, 0.72, 0.94, 0.27]); axm.axis("off"); axm.set_xlim(0, 100); axm.set_ylim(0, 20)
def bubble(x, y, w, txt, fc):
    axm.add_patch(FancyBboxPatch((x, y), w, 4.6, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="none"))
    axm.text(x + 1, y + 2.3, txt, va="center", fontsize=34)
axm.text(0, 18.5, "Each Qwen3 chat model, after a short standing-up exercise, is asked two ways:", fontsize=38, va="center")
bubble(0, 11.6, 52, "How tall are you? Answer with just a number of centimeters.", "#E6ECF4")
axm.text(53.5, 13.9, "= probability of each answer 100 to 230 cm, averaged", fontsize=32, va="center", color=BLUE, weight="bold")
bubble(0, 3.2, 52, "Are you taller or shorter than 150 cm? Answer with one word.", "#FBEBD8")
axm.text(53.5, 6.6, "= P(taller) summed over marks 0.3 to 3.0 m, read on the\n   scale the same question gives a dog, a child, a\n   basketball player and a giraffe", fontsize=30, va="center", color="#B86E1E", weight="bold", linespacing=1.3)
# main axes, equal aspect in meters
ax = fig.add_axes([0.07, 0.08, 0.91, 0.52]); ax.set_facecolor("white")
step = 0.82; door_w = 0.82; xs = door_w + 0.6 + np.arange(len(MODELS)) * step
ax.set_xlim(-0.05, xs[-1] + 1.75); ax.set_ylim(0, 2.25); ax.set_aspect("equal", adjustable="box", anchor="NW")
ax.add_patch(Rectangle((0, 0), door_w, 2.03, fc="#ECE5D9", ec="#9C8F7A", lw=3, zorder=2))
ax.add_patch(Circle((door_w - 0.1, 1.0), 0.035, fc="#9C8F7A", zorder=3))
ax.text(door_w / 2, 1.5, "door\n2.03 m", ha="center", va="center", fontsize=34, color="#6E6352")
ax.text(door_w - 0.16, 1.0, "knob\n1.0 m", ha="right", va="center", fontsize=28, color="#6E6352")
for y, lab in [(2.03, None), (1.2, "light switch 1.2 m"), (0.91, "kitchen counter 0.91 m")]:
    ax.plot([0, xs[-1] + 1.7], [y, y], color="#BEBEBE", lw=2, ls=(0, (5, 5)), zorder=1)
    if lab: ax.text(xs[-1] + 1.7, y - 0.015, lab, ha="right", va="top", fontsize=30, color="#7A7A7A")
for x, m in zip(xs, MODELS):
    d = S[f"Qwen3-{m}"]["direct"]; body(ax, x, d["self"])
    if m in GHOST: body(ax, x + 0.30, GHOST[m], LIGHT, 6)
ax.set_xticks(xs); ax.set_xticklabels([f"{m}" for m in MODELS], fontsize=38)
for x, m in zip(xs, MODELS):
    ax.text(x, S['Qwen3-'+m]['direct']['self'] + 0.03, f"{S['Qwen3-'+m]['direct']['self']*100:.0f} cm", ha="center", va="bottom", fontsize=32, color=BLUE, weight="bold")
    if m in GHOST: ax.text(x + 0.30, GHOST[m] + 0.03, f"{GHOST[m]*100:.0f} cm", ha="center", va="bottom", fontsize=32, color="#B86E1E", weight="bold", zorder=8, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
ax.set_yticks(np.arange(0, 2.01, 0.5)); ax.set_yticklabels([f"{v:.1f} m" for v in np.arange(0, 2.01, 0.5)], fontsize=32)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("Qwen3 chat model size", fontsize=36, labelpad=14)
fig.text(0.03, 0.625, open("header.txt").read().strip(), fontsize=44, weight="bold", linespacing=1.3)
fig.savefig("$BODY_WORK/fig/body_height.png", dpi=100, facecolor="white")
