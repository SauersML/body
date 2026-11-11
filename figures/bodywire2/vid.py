import sys, os
from still import *
W, H = 1920, 1080; SS = 2; NF = 600
a0, a1 = int(sys.argv[1]), int(sys.argv[2])
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * SS, facecolor="white")
fig.text(.03, .955, TITLE, fontsize=px(50), weight="bold", va="top")
fig.text(.03, .885, SUB.replace("   ·   ", "  ·  "), fontsize=px(25), va="top", color="#111", linespacing=1.4)
legend(fig, .05, .03, .1, 30, r=.006, gap=.016, nx=.0)
ax = fig.add_axes([.18, .08, .64, .78]); ax.set_adjustable("box"); ta = fig.add_axes([.07, .12, .16, .28])
ang = lambda i: (18 + 10 * np.sin(2 * np.pi * i / NF), 30 + 360 * i / NF)
ext = np.array([np.abs((Xc @ rot(*ang(i)).T)[:, [0, 2]]).max(0) for i in range(NF)]).max(0)
asp = (.64 * W) / (.78 * H); hy = max(ext[1], ext[0] / asp) * 1.01; hx = hy * asp
os.makedirs("/tmp/bwf", exist_ok=True)
for i in range(a0, a1):
    ph = i / NF
    el, az = ang(i)
    ax.cla(); R = draw(ax, el, az, lw=2.1, dot=16, lims=False)
    ax.set_xlim(-hx, hx); ax.set_ylim(-hy, hy)
    ta.cla(); ta.set_xlim(-1, 1); ta.set_ylim(-1, 1); ta.set_aspect("equal"); ta.axis("off")
    triad(ta, R, np.array([0, -.1]), L=.4, fs=px(26), lw=2.2)
    fig.savefig(f"/tmp/bwf/ss{i:04d}.png", facecolor="white")
    Image.open(f"/tmp/bwf/ss{i:04d}.png").convert("RGB").resize((W, H), Image.LANCZOS).save(f"/tmp/bwf/f{i:04d}.png")
    os.remove(f"/tmp/bwf/ss{i:04d}.png")
