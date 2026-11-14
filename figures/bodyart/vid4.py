"""Body inside Qwen3-32B, layer 1 -> 64, every layer in order.
Per layer: top-K PCs of the 550 spot-mean reply states. Consecutive layers Procrustes-aligned
(orthogonal, K-dim, anchored at the last layer) so nothing spins. One fixed orthogonal 2-D VIEW
of that K-dim space, fitted once at the last layer: horizontal = the direction along which matched
left/right spots differ most; vertical = the direction (orthogonal to it) best tracking anatomical
height. The data are never mirrored or folded; symmetry is only how the camera looks."""
import sys, os, numpy as np
from scipy.linalg import orthogonal_procrustes
from symdraw4 import *
from multiprocessing import Pool
K = 12
A = np.load(sys.argv[1], mmap_mode='r')
NL = A.shape[0] - 1
Y = {}
for l in range(1, NL + 1):
    Z = np.asarray(A[l], np.float64); Z = Z - Z.mean(0)
    u, s, vt = np.linalg.svd(Z, full_matrices=False); y = u[:, :K] * s[:K]
    Y[l] = y / np.sqrt((y ** 2).sum())
AL = {NL: Y[NL]}
for l in range(NL - 1, 0, -1): R_, _ = orthogonal_procrustes(Y[l], AL[l + 1]); AL[l] = Y[l] @ R_
# left/right pairs by phrase
sd = np.array([l_['side'] for l_ in L]); key = {}
for i, p in enumerate(PH):
    if sd[i] != 0: key.setdefault(p.replace('left', 'X').replace('right', 'X'), {})[int(sd[i])] = i
pairs = np.array([(d[1], d[-1]) for d in key.values() if 1 in d and -1 in d])
VL = int(sys.argv[2]) if len(sys.argv) > 2 else NL
Yv = AL[VL]
D = Yv[pairs[:, 0]] - Yv[pairs[:, 1]]
dx = SP[pairs[:, 0], 0] - SP[pairs[:, 1], 0]
h = np.linalg.lstsq(D, dx, rcond=None)[0] if os.environ.get('HX', '0') == '1' else np.linalg.svd(D, full_matrices=False)[2][0]; h /= np.linalg.norm(h)
if np.mean(D @ h * np.sign(SP[pairs[:, 0], 0] - SP[pairs[:, 1], 0])) < 0: h = -h
Mid = (Yv[pairs[:, 0]] + Yv[pairs[:, 1]]) / 2; Mo = Mid - Mid @ h[:, None] * h[None]
ht = SP[pairs[:, 0], 1] - SP[pairs[:, 0], 1].mean()
v = np.linalg.lstsq(Mo, ht, rcond=None)[0]; v -= (v @ h) * h; v /= np.linalg.norm(v)
P = np.c_[h, v]
print('pairs', len(pairs), 'view layer', VL, 'height corr', round(np.corrcoef(Yv @ v, SP[:, 1])[0, 1], 2),
      'side corr', round(np.corrcoef(Yv @ h, SP[:, 0])[0, 1], 2), 'sym err', round(np.abs((Yv[pairs[:, 0]] @ P)[:, 1] - (Yv[pairs[:, 1]] @ P)[:, 1]).mean() / np.std(Yv @ v), 2))
T = SP[:, :2] - SP[:, :2].mean(0); TS = np.sqrt((T ** 2).sum())
KEY = {}
for l in AL:
    V = AL[l] @ P; V = V - V.mean(0); KEY[l] = V * (TS / np.sqrt((V ** 2).sum()))
FPS = 30; T0, T1, DUR = 0.5, 20.0, 24.0   # hold layer 1, sweep to 64 by T1, hold
NF = int(FPS * DUR)
def layer_at(k):
    t = k / FPS; u = np.clip((t - T0) / (T1 - T0), 0, 1)
    u = u * u * (3 - 2 * u) * 0.3 + u * 0.7
    return 1 + u * (NL - 1)
def pos(x):
    i = int(np.floor(x)); f = x - i
    if i >= NL: return KEY[NL]
    p0, p1, p2, p3 = (KEY[min(max(j, 1), NL)] for j in (i - 1, i, i + 1, i + 2))
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
allv = np.vstack([KEY[l] for l in range(max(1, NL - 20), NL + 1)])
cy = (np.percentile(allv[:, 1], 99.5) + np.percentile(allv[:, 1], 0.5)) / 2
hh = max((np.percentile(allv[:, 1], 99.5) - np.percentile(allv[:, 1], 0.5)) / 2 * 1.30,
         np.percentile(np.abs(allv[:, 0]), 99.5) * 1.25 * 9 / 16)
cy -= hh * 0.06
LAB = [('head', SPN[0], TR)] + [(('hand' if nm == 'arm' else 'foot'), ch[-1], (ARM if nm == 'arm' else LEG)[1]) for nm, ch in CH]
OUTD = os.environ.get('FR', 'frames3')
def frame(k):
    x = layer_at(k); V = pos(x)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, V, lw=9, dot=60); ax.set_aspect('equal')
    ax.set_ylim(cy - hh, cy + hh); ax.set_xlim(-hh * 16 / 9, hh * 16 / 9)
    ctr = V.mean(0)
    for name, g, col in LAB:
        p = V[g].mean(0); d = p - ctr; d = d / (np.linalg.norm(d) + 1e-9)
        q = p + d * hh * 0.11
        ax.text(q[0], q[1], name, fontsize=50, color=col, ha='center', va='center', weight='demibold', zorder=6)
    L_ = int(round(x))
    fig.text(0.035, 0.90, f'layer {L_}', fontsize=78, color=INK, weight='demibold')
    fig.text(0.035, 0.84, f'of {NL}', fontsize=44, color=INK)
    fig.savefig(f'{OUTD}/f{k:04d}.png', facecolor='white'); plt.close(fig)
if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True)
    ks = range(NF) if len(sys.argv) < 4 else [int(q) for q in sys.argv[3].split(',')]
    with Pool(int(os.environ.get('NP', 8))) as p: p.map(frame, ks)
    print('frames', len(ks))
