exec(open(os.path.expandvars("$BODY_REPO/figures/bodychain/data_part.py")).read()) if False else None
import os
exec(open(os.path.expandvars("$BODY_REPO/figures/bodychain/data_part.py")).read())
from matplotlib.patches import FancyArrowPatch
WPX, HPX = 2400, 1800; PT = 72 / 100
fig = plt.figure(figsize=(WPX / 100, HPX / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WPX); ax.set_ylim(HPX, 0); ax.axis("off"); ax.set_aspect("equal")
K = "#111111"; SKIN = "#E6EAF0"
def T(x, y, s, px=50, **k):
    k.setdefault("ha", "left"); k.setdefault("va", "center"); return ax.text(x, y, s, fontsize=px * PT, color=k.pop("color", K), **k)
def cmix(c0, c1, t):
    c0 = np.array(matplotlib.colors.to_rgb(c0)); c1 = np.array(matplotlib.colors.to_rgb(c1)); return tuple(c0 + (c1 - c0) * t)
ARMC = lambda d: cmix("#93C4F5", "#0A2A85", d / 6)
LEGC = lambda d: cmix("#F9B98F", "#9E1F0B", d / 6)
WORDC = "#A87400"; MODC = "#0A2A85"
def arrow(p0, p1, c=K, lw=5, ms=34):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={ms/30:.2f},head_width={ms/50:.2f}", mutation_scale=30, color=c, lw=lw, zorder=3))
fig.canvas.draw(); RND = fig.canvas.get_renderer()
def width(t): return t.get_window_extent(RND).width

# ---------------- header + method ----------------
T(40, 58, "Inside Qwen3-8B the arm lies in order, fingers to shoulder, and places", 60, weight="bold")
T(40, 130, "near on the body sit near inside the model, beyond word similarity", 60, weight="bold")
ax.add_patch(FancyBboxPatch((40, 196), 1120, 92, boxstyle="round,pad=0,rounding_size=46", fc="#EDF2FA", ec="none"))
t1 = T(78, 242, "A feather brushes slowly across your left ", 48); T(78 + width(t1), 242, "thumb.", 48, weight="bold", color=ARMC(6))
arrow((1185, 242), (1265, 242))
T(1290, 242, "Qwen3-8B, layer 25: 54 places × 8 wordings", 48)

# ---------------- body ----------------
def capsule(p0, p1, r0, r1, n=28, fc=SKIN, z=1):
    p0, p1 = np.array(p0, float), np.array(p1, float); d = p1 - p0; u = d / np.hypot(*d); a0 = np.arctan2(u[0], -u[1])
    pts = [p1 + r1 * np.array([np.cos(a0 - t), np.sin(a0 - t)]) for t in np.linspace(0, np.pi, n)]
    pts += [p0 + r0 * np.array([np.cos(a0 + np.pi - t), np.sin(a0 + np.pi - t)]) for t in np.linspace(0, np.pi, n)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", zorder=z))
cx = 300
ax.add_patch(Ellipse((cx, 440), 128, 156, fc=SKIN, ec="none"))
capsule((cx, 500), (cx, 580), 36, 38)
torso = [(cx - 118, 588), (cx + 118, 588), (cx + 130, 640), (cx + 108, 780), (cx + 86, 880), (cx + 112, 990), (cx + 118, 1060),
         (cx - 118, 1060), (cx - 112, 990), (cx - 86, 880), (cx - 108, 780), (cx - 130, 640)]
ax.add_patch(Polygon(torso, closed=True, fc=SKIN, ec="none"))
J = {}
for s in (1, -1):
    sh, el, wr = (cx + s * 134, 628), (cx + s * 196, 855), (cx + s * 240, 1055)
    capsule(sh, el, 42, 31); capsule(el, wr, 33, 24)
    hp, kn, an = (cx + s * 70, 1040), (cx + s * 76, 1390), (cx + s * 70, 1690)
    capsule(hp, kn, 58, 42); capsule(kn, an, 42, 26)
    ax.add_patch(Ellipse((cx + s * 88, 1724), 112, 46, angle=-s * 6, fc=SKIN, ec="none"))
    hc = np.array((cx + s * 248, 1122.)); ax.add_patch(Ellipse(hc, 78, 100, angle=-s * 5, fc=SKIN, ec="none"))
    capsule(wr, hc + (0, -30), 24, 34)
    fx = [-26, -9, 8, 24]; fl = [62, 76, 70, 58]
    tips = [hc + (s * 52, 52)]; capsule(hc + (s * 26, -10), tips[0], 13, 10)
    for o, l in zip(fx[::-1] if s == 1 else fx, (fl[::-1] if s == 1 else fl)):
        b = hc + (s * o, 40); t = b + (s * o * 0.25, l); capsule(b, t, 11, 9); tips.append(t)
    J[s] = dict(sh=np.array(sh, float), el=np.array(el, float), wr=np.array(wr, float), hc=hc, tips=tips,
                hp=np.array(hp, float), kn=np.array(kn, float), an=np.array(an, float))
j = J[1]
bodypos = {"shoulder": j["sh"], "upperarm": (j["sh"] + j["el"]) / 2, "elbow": j["el"], "forearm": (j["el"] + j["wr"]) / 2, "wrist": j["wr"],
           "palm": j["hc"] + (-8, 4), "backhand": j["hc"] + (0, 52)}
# tips order: thumb, then fingers from thumb side outward? thumb is lateral (+x); fingers listed index..little going medial
for t, p in zip(["thumb", "index", "middle", "ring", "little"], [j["tips"][0]] + j["tips"][1:][::-1] if False else j["tips"]): bodypos[t] = p
legpos = {"hip": j["hp"], "thigh": (j["hp"] + j["kn"]) / 2, "knee": j["kn"], "shin": (j["kn"] + j["an"]) / 2, "ankle": j["an"],
          "heel": j["an"] + (-10, 44), "sole": j["an"] + (26, 50), "toes": j["an"] + (70, 38)}
def node(p, c, r=19, z=5, ring=False):
    ax.add_patch(Circle(p, r + 4, fc="white", ec="none", zorder=z))
    ax.add_patch(Circle(p, r - 2.5 if ring else r, fc="white" if ring else c, ec=c, lw=5 if ring else 0, zorder=z + 0.1))
for t, p in bodypos.items(): node(p, ARMC(dep[t]), r=11 if t in ARM[:5] else 18, ring=False)
for t, p in legpos.items(): node(p, LEGC(dep[t]), r=14 if t in ("heel", "sole", "toes") else 18, ring=False)

# ---------------- model maps ----------------
NAMES = dict(palm="palm", backhand="knuckles", wrist="wrist", forearm="forearm", elbow="elbow", upperarm="upper arm", shoulder="shoulder",
             toes="toes", sole="sole", heel="heel", ankle="ankle", shin="shin", knee="knee", thigh="thigh", hip="hip")
def draw_map(m, sel, col, cxm, top, scale):
    ymax = max(m[t][0] for t in sel)
    P = {t: np.array([cxm + scale * m[t][1], top + scale * (ymax - m[t][0])]) for t in sel}
    groups = sorted(set(dep[t] for t in sel)); cent = {d: np.mean([P[t] for t in sel if dep[t] == d], 0) for d in groups}
    ch = [cent[d] for d in groups]
    xs = np.array([c[0] for c in ch]); ys = np.array([c[1] for c in ch])
    ax.plot(xs, ys, color=col(3), lw=9, solid_capstyle="round", solid_joinstyle="round", zorder=2, alpha=.45)
    for t in sel:
        c = cent[dep[t]]
        if np.hypot(*(P[t] - c)) > 1: ax.plot([c[0], P[t][0]], [c[1], P[t][1]], color=col(dep[t]), lw=3.5, zorder=2, alpha=.45)
    for t in sel: node(P[t], col(dep[t]), r=21, ring=False)
    return P
MX = 925
T(MX, 378, "Same places, laid out by", 48, ha="center"); T(MX, 438, "distances inside Qwen3-8B", 48, ha="center")
PA = draw_map(maps["arm"], ARM, ARMC, MX, 510, 52)
PL = draw_map(maps["leg"], LEG, LEGC, MX, 1150, 50)
LBL = {}
OFF = dict(wrist=(1, 46), elbow=(-1, -6), knee=(1, -26), ankle=(1, 30), heel=(1, 0))
for P in (PA, PL):
    for t, p in P.items(): print(t, p.round())
for P in (PA, PL):
    for t, p in P.items():
        if t in ARM[:5]: continue
        side = OFF.get(t, (1 if p[0] >= MX else -1, 0))
        LBL[t] = T(p[0] + side[0] * 34, p[1] + side[1], NAMES[t], 48, ha="left" if side[0] > 0 else "right")
fc_ = np.array([PA[t] for t in ARM[:5]]); T(fc_[:, 0].max() + 34, fc_[:, 1].mean(), "fingers", 48)

# ---------------- right: all pairs ----------------
rng = np.random.default_rng(0)
def panel(x0, y0, w, h, xs, ys, col, xmax=15.5, alpha=.38, r=7, med=True):
    X = lambda v: x0 + w * v / xmax; Y = lambda v: y0 + h * (1 - v)
    ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=K, lw=2.5); ax.plot([x0, x0], [y0, y0 + h], color=K, lw=2.5)
    jit = rng.uniform(-.17, .17, len(xs))
    ax.scatter(X(xs + jit), Y(ys), s=r * r, color=col, alpha=alpha, lw=0, zorder=3)
    if med:
        g = np.array(sorted(set(xs))); m = np.array([np.median(ys[np.abs(xs - v) <= 1.0]) for v in g])
        ax.plot(X(g), Y(m), color=col, lw=9, solid_capstyle="round", zorder=4)
    return X, Y
SX0, SY0, SW, SH = 1500, 490, 860, 600
X, Y = panel(SX0, SY0, SW, SH, x, y, MODC)
for v in (0, 5, 10, 15): T(X(v), SY0 + SH + 38, str(v), 48, ha="center")
T(SX0 + SW / 2, SY0 + SH + 104, "distance along the body (steps) \u2192", 48, ha="center")
T(SX0 - 30, SY0 + SH / 2, "distance inside model \u2192", 48, ha="center", rotation=90)
T(SX0 - 30 + 0, SY0 + SH / 2, "", 10)
T(SX0 - 70, 366, f"Rank correlation {rho_main:.2f}", 66, weight="bold", color=MODC)
T(SX0 - 70, 436, "all 496 pairs, word similarity removed", 48, color=MODC)

# word-embedding alone
WX0, WY0, WW, WH = 1500, 1390, 400, 300
wr_ = rankdata(Dw[it]); wr_ = (wr_ - wr_.min()) / (wr_.max() - wr_.min())
X2, Y2 = panel(WX0, WY0, WW, WH, x, wr_, WORDC, alpha=.32, r=5)
T(WX0 - 70, WY0 - 120, "Words alone", 50, color=WORDC, weight="bold")
T(WX0 - 70, WY0 - 56, f"rank correlation −{abs(rho_word):.2f}", 48, color=WORDC)
T(WX0 - 30, WY0 + WH / 2, "distance →", 48, ha="center", rotation=90)
T(WX0 + WW / 2, WY0 + WH + 44, "along the body \u2192", 48, ha="center")

# by layer
LX0, LY0, LW, LH = 2070, 1390, 290, 300
lo, hi = -0.2, 0.6
LXf = lambda L: LX0 + LW * L / 36; LYf = lambda v: LY0 + LH * (hi - v) / (hi - lo)
ax.plot([LX0, LX0 + LW], [LYf(0), LYf(0)], color=K, lw=2.5)
ax.plot([LX0, LX0], [LY0, LY0 + LH], color=K, lw=2.5)
Ls = np.arange(1, 37); ax.plot(LXf(Ls), LYf(np.array(bylayer[1:])), color=MODC, lw=7, solid_joinstyle="round")
ax.plot([LXf(0), LXf(36)], [LYf(rho_word)] * 2, color=WORDC, lw=6, ls=(0, (2, 1.4)))
pk = int(np.nanargmax(bylayer)); ax.add_patch(Circle((LXf(pk), LYf(bylayer[pk])), 15, fc=MODC, ec="white", lw=4, zorder=6))
T(LX0 - 90, LY0 - 120, "Peak at layer 25", 50, weight="bold", color=MODC)
T(LX0 - 16, LYf(0), "0", 48, ha="right")
T(LX0 - 16, LYf(0.5), "0.5", 48, ha="right")
T(LX0 + LW / 2, LY0 + LH + 44, "layer 1 \u2192 36", 48, ha="center")

fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_map_chain.png"), dpi=100, facecolor="white")
