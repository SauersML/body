"""Body inside Qwen3-32B, layer 1 -> 64. Per layer: top-K PCs of the 550 spot means.
Consecutive layers Procrustes-aligned (orthogonal, K-dim) so nothing spins; one fixed view
(an orthogonal K->3 projection fitted once, at the anchor layer, to the anatomical template)
makes left-right horizontal. Data never mirrored or folded."""
import sys, numpy as np
from scipy.linalg import orthogonal_procrustes
from symdraw import *
from multiprocessing import Pool
K = 12
A = np.load(sys.argv[1] if len(sys.argv) > 1 else '$BODY_RUNS/reply_all_layers.npy', mmap_mode='r')
NL = A.shape[0] - 1                      # hidden_states: 0 = embeddings, k = after layer k
T = SP - SP.mean(0); TS = np.sqrt((T ** 2).sum())
Y = {}
for l in range(1, NL + 1):
    Z = np.asarray(A[l], np.float64); Z = Z - Z.mean(0)
    u, s, vt = np.linalg.svd(Z, full_matrices=False); y = u[:, :K] * s[:K]
    Y[l] = y / np.sqrt((y ** 2).sum())
def fit(y):
    P, _ = orthogonal_procrustes(y, np.c_[T, np.zeros((len(T), K - 3))]); return P
score = {l: np.mean([np.corrcoef((Y[l] @ fit(Y[l]))[:, j], T[:, j])[0, 1] for j in range(3)]) for l in Y}
ANC = max(Y, key=lambda l: score[l]) if len(sys.argv) < 3 else int(sys.argv[2])
print('anchor', ANC, 'fit', round(score[ANC], 2), {l: round(v, 2) for l, v in score.items() if l % 8 == 0 or l == NL})
AL = {ANC: Y[ANC]}
for l in range(ANC - 1, 0, -1): R_, _ = orthogonal_procrustes(Y[l], AL[l + 1]); AL[l] = Y[l] @ R_
for l in range(ANC + 1, NL + 1): R_, _ = orthogonal_procrustes(Y[l], AL[l - 1]); AL[l] = Y[l] @ R_
P = fit(Y[ANC])[:, :3]
KEY = {}
for l in AL:
    V = AL[l] @ P; V = V - V.mean(0); KEY[l] = V * (TS / np.sqrt((V ** 2).sum()))
np.savez('/tmp/bodyvid_keys.npz', **{str(l): KEY[l] for l in KEY})
FPS, DUR, HOLD = 30, 20.0, 3.0
NF = int(FPS * DUR)
def layer_at(k):
    t = k / FPS; u = min(t / (DUR - HOLD), 1.0)
    u = u * u * (3 - 2 * u) * 0.35 + u * 0.65          # gentle ease in/out, mostly constant pace
    return 1 + u * (NL - 1)
def pos(x):
    i = int(np.floor(x)); f = x - i
    if i >= NL: return KEY[NL]
    p0, p1, p2, p3 = (KEY[min(max(j, 1), NL)] for j in (i - 1, i, i + 1, i + 2))
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
fin = KEY[NL]; cy = (fin[:, 1].max() + fin[:, 1].min()) / 2; hh = (fin[:, 1].max() - fin[:, 1].min()) / 2 * 1.18
def frame(k):
    x = layer_at(k); V = pos(x)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, V[:, :2], lw=8, dot=55); ax.set_aspect('equal')
    ax.set_ylim(cy - hh, cy + hh); ax.set_xlim(-hh * 16 / 9, hh * 16 / 9)
    fig.text(0.035, 0.92, f'layer {int(round(x))}', fontsize=40, color=INK)
    fig.savefig(f'frames2/f{k:04d}.png', facecolor='white'); plt.close(fig)
if __name__ == '__main__':
    import os; os.makedirs('frames2', exist_ok=True)
    ks = range(NF) if len(sys.argv) < 4 else [int(v) for v in sys.argv[3].split(',')]
    with Pool(12) as p: p.map(frame, ks)
    print('frames', len(ks))
