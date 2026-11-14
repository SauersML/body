"""Body inside Qwen3-32B, layer 1 -> 64. Per layer: top-12 PCs of the 550 body-spot reply states,
Procrustes-aligned layer to layer (anchored at 64). ONE fixed linear readout, fitted once at layer 64 on all
550 spots: vertical = height, horizontal = distance from the body's midline. Each part drawn at the centroid
of its own spots; left-side parts placed on screen-right, right-side parts on screen-left (each from its own data)."""
import sys, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb as c2r
from multiprocessing import Pool
import matplotlib.patheffects as pe
from parts5 import *
fm.fontManager.addfont('/System/Library/Fonts/Avenir Next.ttc'); plt.rcParams['font.family'] = 'Avenir Next'
INK = '#1D1D1F'; MIDC = '#33407A'; ARM = ('#F08A4B', '#B8325A'); LEG = ('#3FB8A3', '#1F5F7A')
AL = np.load('AL12.npy'); NL = len(AL)
ax_ = np.abs(SP[:, 0]); Y = AL[-1]
w = np.linalg.lstsq(Y, ax_ - ax_.mean(), rcond=None)[0]; v = np.linalg.lstsq(Y, SP[:, 1] - SP[:, 1].mean(), rcond=None)[0]
sd = np.array([1 if s == 'L' else -1 if s == 'R' else 0 for s in S])
def lerp(a, b, t): a, b = np.array(c2r(a)), np.array(c2r(b)); return a + (b - a) * t
GK = list(GROUPS); GI = [np.array(GROUPS[k]) for k in GK]
def centroids(l):  # l: 1..NL
    Yl = AL[l - 1]; ox = Yl @ w; oy = Yl @ v
    mid0 = ox[sd == 0].mean()
    C = {}
    for k, ii in zip(GK, GI):
        sgn = 1 if k[1] == 'L' else -1 if k[1] == 'R' else 0
        x = (ox[ii] - mid0).mean() * sgn if sgn else 0.0
        C[k] = (x, oy[ii].mean())
    return np.array([C[k] for k in GK])
def spots(l):
    Yl = AL[l - 1]; ox = Yl @ w; oy = Yl @ v; mid0 = ox[sd == 0].mean()
    return np.c_[np.where(sd == 0, 0, (ox - mid0) * sd), oy]
KEYC = {l: centroids(l) for l in range(1, NL + 1)}; KEYS = {l: spots(l) for l in range(1, NL + 1)}
def interp(KEY, x):
    i = int(np.floor(x)); f = x - i
    if i >= NL: return KEY[NL]
    p0, p1, p2, p3 = (KEY[min(max(j, 1), NL)] for j in (i - 1, i, i + 1, i + 2))
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
gi = {k: j for j, k in enumerate(GK)}
CHAINS = []
for s, sg in (('L', 1), ('R', -1)):
    CHAINS.append(([('chest', '')] + [(p, s) for p in ARMC], ARM))
    CHAINS.append(([('belly', '')] + [(p, s) for p in LEGC], LEG))
SPINE = [('head', ''), ('neck', ''), ('chest', ''), ('belly', '')]
LABELS = [(('head', ''), MIDC, (0, 78), 'center'), (('chest', ''), MIDC, (-34, 8), 'right'), (('belly', ''), MIDC, (-34, 0), 'right'),
          (('hand', 'L'), ARM[1], (30, 34), 'left'), (('knee', 'R'), LEG[0], (-34, 0), 'right'), (('toes', 'L'), LEG[1], (30, -24), 'left')]
SPC = np.array([MIDC if r in ('head', 'neck', 'torso') else ARM[1] if r.startswith(('arm', 'hand')) else LEG[1] for r in SR])
SPC = np.array([c2r(c) for c in SPC])
FPS = 30; T0, T1, DUR = 0.5, 20.0, 24.0
NF = int(FPS * DUR)
def layer_at(k):
    t = k / FPS; u = np.clip((t - T0) / (T1 - T0), 0, 1)
    u = u * u * (3 - 2 * u) * 0.3 + u * 0.7
    return 1 + u * (NL - 1)
ALLC = np.vstack([KEYC[l] for l in range(1, NL + 1)]); C64 = KEYC[NL]
cy = (ALLC[:, 1].max() + ALLC[:, 1].min()) / 2 + 2; hh = (ALLC[:, 1].max() - ALLC[:, 1].min()) / 2 * 1.11
OUTD = os.environ.get('FR', 'v5')
def frame(k, out=None):
    x = layer_at(k); C = interp(KEYC, x); Sx = interp(KEYS, x)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(Sx[:, 0], Sx[:, 1], s=40, c=SPC, alpha=0.18, lw=0, zorder=1)
    for ch, (c0, c1) in CHAINS:
        P = np.array([C[gi[g]] for g in ch]); n = len(P)
        for j in range(n - 1):
            ax.plot(P[j:j + 2, 0], P[j:j + 2, 1], c=lerp(c0, c1, j / (n - 2)), lw=18, solid_capstyle='round', zorder=3)
        ax.scatter(P[1:, 0], P[1:, 1], s=420, c=[lerp(c0, c1, j / (n - 2)) for j in range(n - 1)], zorder=4, lw=0)
    P = np.array([C[gi[g]] for g in SPINE]); ax.plot(P[:, 0], P[:, 1], c=MIDC, lw=20, solid_capstyle='round', zorder=5)
    ax.scatter(P[1:, 0], P[1:, 1], s=480, c=MIDC, zorder=5, lw=0); ax.scatter(P[0, 0], P[0, 1], s=4200, c=MIDC, zorder=6, lw=0)
    for g, col, off, ha in LABELS:
        p = C[gi[g]]
        ax.annotate('foot' if g[0]=='toes' else g[0], p, xytext=off, textcoords='offset points', fontsize=50, color=col, weight='demibold',
                    ha=ha, va='center', zorder=7, path_effects=[pe.withStroke(linewidth=10, foreground='white')])
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_ylim(cy - hh, cy + hh); ax.set_xlim(-hh * 16 / 9, hh * 16 / 9)
    fig.text(0.04, 0.88, f'layer {int(round(x))}', fontsize=84, color=INK, weight='demibold')
    fig.text(0.04, 0.82, f'of {NL}', fontsize=48, color=INK)
    fig.savefig(out or f'{OUTD}/f{k:04d}.png', facecolor='white'); plt.close(fig)
if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True)
    ks = range(NF) if len(sys.argv) < 2 else [int(q) for q in sys.argv[1].split(',')]
    with Pool(int(os.environ.get('NP', 10))) as p: p.map(frame, ks)
    print('frames', len(ks))
