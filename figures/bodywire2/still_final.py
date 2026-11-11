import os
import sys
from render import *
from PIL import Image
SS = 2; W, H = 3200, 2340
px = lambda p: p * 72 / 100
HERO = dict(el=22, az=30)
TITLE = "Qwen3-8B's inner body: the skin's grid, as the model arranges it"
SUB = "each dot: one of 407 touched body spots, placed by the model's layer-27 activations (PCA)   ·   each line: two spots that are neighbours on real skin"
LEG = [("head", "head"), ("torso", "torso"), ("arms", "arm"), ("hands", "hand"), ("legs", "leg"), ("feet", "foot")]
def legend(fig, y, x0, step, fs, r=.0075, gap=.019, note=True, nx=-.03):
    la = fig.add_axes([0, 0, 1, 1], facecolor="none"); la.set_xlim(0, 1); la.set_ylim(0, 1); la.axis("off")
    asp = fig.get_figwidth() / fig.get_figheight(); x = x0
    for nm, k in LEG:
        for j, h in enumerate(PAL[k]):
            la.add_patch(matplotlib.patches.Ellipse((x + r + j * gap, y - .001), 2 * r, 2 * r * asp, color=h, lw=0))
        la.text(x + 2 * r + gap + .008, y, nm, fontsize=px(fs), va="center_baseline"); x += step
    if note: la.text(x + nx, y, "light = left side,  deep = right side", fontsize=px(fs * .92), va="center_baseline", color="#111")
    return la
def inset_triad(fig, rect, R, fs=40, L=.5, lw=3):
    a = fig.add_axes(rect); a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_aspect("equal"); a.axis("off")
    triad(a, R, np.array([0, -.2]), L=L, fs=px(fs), lw=lw)
if __name__ == "__main__":
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * SS, facecolor="white")
    fig.text(.025, .972, TITLE, fontsize=px(84), weight="bold", va="top")
    fig.text(.025, .921, SUB, fontsize=px(42), va="top", color="#111")
    ax = fig.add_axes([.015, .09, .65, .80]); R = draw(ax, **HERO, lw=2.8, dot=40)
    inset_triad(fig, [.035, .735, .15, .19], R)
    for k, (nm, el, az, y0) in enumerate([("seen straight on", 0, 0, .50), ("seen from the side", 0, -90, .09)]):
        ax = fig.add_axes([.715, y0 + .015, .27, .33]); draw(ax, el, az, lw=2.2, dot=20)
        fig.text(.685, y0 + .375, nm, fontsize=px(52), weight="bold", va="top", color="#222")
        inset_triad(fig, [.625, y0 - .045, .12, .15], rot(el, az), fs=36, lw=2.6, L=.42)
    legend(fig, .038, .025, .108, 50, nx=.0)
    fig.savefig("/tmp/bw_ss.png", facecolor="white")
    im = Image.open("/tmp/bw_ss.png").convert("RGB"); im = im.resize((W, H), Image.LANCZOS)
    im.save(sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$BODY_REPO/figures/out/body_wire_v2.png"))
