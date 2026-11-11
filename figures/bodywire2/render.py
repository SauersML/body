import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from base import *
FD = os.path.expandvars("$BODY_WORK/fonts/inter/")
import glob
for f in glob.glob(FD + "extras/ttf/Inter-*.ttf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"
# palette: (left = light, right = deep); midline gets the midpoint
PAL = {"head": ("#a98bd3", "#5b3a8c"), "torso": ("#9aa3ad", "#3f4954"), "arm": ("#f0a35a", "#b35a12"),
       "hand": ("#ef8aa6", "#a3214a"), "leg": ("#7fb8e3", "#1d5c99"), "foot": ("#86cf8a", "#2c7a3a")}
def rgb(h): return np.array(to_rgb(h))
VC = np.array([rgb(PAL[c][0]) if s > 0 else rgb(PAL[c][1]) if s < 0 else (rgb(PAL[c][0]) + rgb(PAL[c][1])) / 2 for c, s in zip(C6, side)])
Xc = X - (X.max(0) + X.min(0)) / 2
R0 = np.linalg.norm(Xc, axis=1).max()
def rot(el, az):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz
WHITE = np.ones(3)
def draw(ax, el, az, lw=2.2, face=True, dot=14, fade=.5, scale=1.0, shrink=.62, falpha=.07, pad=.03, lims=True):
    R = rot(el, az); P3 = Xc @ R.T          # screen x = P3[:,0], up = P3[:,2], toward viewer = P3[:,1]
    S2 = P3[:, [0, 2]] * scale; dep = P3[:, 1]
    lo, hi = np.percentile(dep, [3, 97]); t = lambda d: np.clip((d - lo) / (hi - lo), 0, 1)   # 0 near .. 1 far
    if face:
        tri = P3[T]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lite = np.array([-.35, .75, .55]); lite /= np.linalg.norm(lite)
        sh = np.abs(n @ lite)
        tt = t(dep[T].mean(1)); o = np.argsort(-tt)
        fc = VC[T].mean(1)
        fc = fc * (.80 + .20 * sh[:, None]) ; fc = fc + (WHITE - fc) * (fade * tt[:, None])
        ax.add_collection(PolyCollection(S2[T][o], facecolors=np.c_[fc[o], np.full(len(o), falpha)], edgecolors="none", antialiased=True, zorder=1))
    tt = t(dep[E].mean(1)); o = np.argsort(-tt)
    c = VC[E].mean(1) * 0 + np.where((side[E[:, 0]] == side[E[:, 1]])[:, None], VC[E[:, 0]], VC[E].mean(1))
    c = c + (WHITE - c) * (fade * tt[:, None])
    ax.add_collection(LineCollection(S2[E][o], colors=c[o], linewidths=lw * (1 - shrink * tt[o]), capstyle="round", antialiased=True, zorder=2))
    tv = t(dep); o = np.argsort(-tv); cv = VC + (WHITE - VC) * (fade * tv[:, None])
    ax.scatter(S2[o, 0], S2[o, 1], s=dot * (1 - shrink * tv[o]) ** 2, c=cv[o], edgecolors="none", zorder=3)
    ax.set_aspect("equal", adjustable="datalim" if lims else "box"); ax.axis("off")
    if not lims: return R
    (x0, y0), (x1, y1) = S2.min(0), S2.max(0); ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    return R
AX = [("PC2", "trunk → tip", 0), ("PC5", "back → front", 1), ("PC3", "feet → head", 2)]
def triad(ax, R, origin, L=.16, fs=20, lw=2.4, which=(0, 1, 2), col="#222", sep=.2):
    items = []
    for nm, lab, i in AX:
        if i not in which: continue
        v = R[:, i][[0, 2]]; n = np.linalg.norm(v)
        a = float(np.clip((n - .2) / .25, 0, 1))
        if a <= 0: continue
        u = v / n; items.append([lab, v, u, origin + v * L + u * .07, a])
    for _ in range(50):   # push labels apart vertically
        for p in items:
            for q in items:
                if p is q: continue
                d = p[3] - q[3]
                if abs(d[1]) < sep and abs(d[0]) < .9: p[3] = p[3] + np.array([0, np.sign(d[1] + 1e-6) * .01])
    for lab, v, u, p, a in items:
        ax.annotate("", xy=origin + v * L, xytext=origin, arrowprops=dict(arrowstyle="-|>,head_length=.6,head_width=.3", lw=lw, color=col, alpha=a, shrinkA=0, shrinkB=0), zorder=5)
        ha = "left" if u[0] > .35 else "right" if u[0] < -.35 else "center"
        ax.text(*p, lab, fontsize=fs, ha=ha, va="center", color=col, alpha=a, zorder=6)
