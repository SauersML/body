"""compute/fig/body_instill_v3c.png: the body design (exact sim geometry) on top, per-touch data below.
Hand-built SVG, rendered with Chrome for Testing. Data: v3/fig3c_{named,nameless}_s0.npz (fig3c_data.py), interp/closed-loop jsons."""
import json, glob, subprocess, sys, numpy as np
from matplotlib import cm
sys.path.insert(0, "."); import sim3 as sim
from body import fk, SEG_LEN, SHOULDER

W, H = 3000, 3900
FONT = "'Helvetica Neue',Helvetica,Arial,sans-serif"
NAVY, PURP, RED, GOLD, GRAY, INK = "#1f3f73", "#8e44ad", "#c0392b", "#d4a017", "#9aa3ad", "#111"
out = []
def T(x, y, s, fs=50, fw=400, fill=INK, anchor="start", extra=""):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{fs}" font-weight="{fw}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def rect(x, y, w, h, fill, rx=0, extra=""): out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>')
def line(x1, y1, x2, y2, c, w=4, extra=""): out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {extra}/>')
def circ(x, y, r, fill, extra=""): out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" {extra}/>')
def path(d, c, w=6, fill="none", extra=""): out.append(f'<path d="{d}" stroke="{c}" stroke-width="{w}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round" {extra}/>')
def poly(pts, fill, extra=""): out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" {extra}/>')
def vir(t): r, g, b, _ = cm.viridis(0.02 + 0.9 * float(np.clip(t, 0, 1))); return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
def heat(t):
    t = float(np.clip(t, 0, 1)); a = np.array([255, 255, 255]); b = np.array([192, 57, 43]); c = a + (b - a) * t ** .9
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
def leader(x1, y1, x2, y2): line(x1, y1, x2, y2, INK, 3); circ(x2, y2, 7, INK)

# ======================= TOP: the body, exact sim geometry =======================
q = np.array([-1.25, -0.35, -0.2, 0.06, 0.32, 0.28]); P = fk(q)
sc, ox, oy = 1240, 400, 1350
X = lambda p: np.array([ox + sc * p[0], oy - sc * p[1]])
HW = [56, 48, 40]  # drawn half-width of upper arm, forearm, hand
def unit(v): return v / np.linalg.norm(v)
def arm_point(a, arc, w):
    """screen point at arc along arm a, across-position w (0 = the sim's left-of-travel edge, 1 = the other edge)."""
    s = min(sim.seg_of(arc), 2); u = (arc - sum(SEG_LEN[:s])) / SEG_LEN[s]
    A_, B_ = X(P[a, s]), X(P[a, s + 1]); d = unit(B_ - A_); n = np.array([d[1], -d[0]])
    return A_ + u * (B_ - A_) + (0.5 - w) * 2 * HW[s] * n, n, s
touch = dict(seg=4, u=0.40, w=0.25, p=0.8, width=0.17); s0 = sim.arc_of(4, .40); w0 = .25
read = sim.channels([touch], np.ones(2))[8:]
# torso and head (torso has no skin)
def Xp(x, y): return X((x, y))
tor = [Xp(-0.36, 0.56), Xp(0.36, 0.56), Xp(0.3, 0.2), Xp(-0.3, 0.2)]
out.append(f'<path d="M {tor[0][0]:.0f} {tor[0][1]:.0f} L {tor[1][0]:.0f} {tor[1][1]:.0f} L {tor[2][0]:.0f} {tor[2][1]:.0f} L {tor[3][0]:.0f} {tor[3][1]:.0f} Z" fill="url(#torso)" stroke="#8d99a8" stroke-width="4" stroke-linejoin="round"/>')
hx, hy = X((0, .74)); out.append(f'<ellipse cx="{hx:.0f}" cy="{hy:.0f}" rx="96" ry="118" fill="url(#head)" stroke="#8d99a8" stroke-width="4"/>')
rect(hx - 38, hy + 110, 76, 50, "#c7d0db")
rect(tor[0][0] - 10, tor[0][1] - 4, tor[1][0] - tor[0][0] + 20, tor[2][1] - tor[0][1] + 10, "url(#fade)")
# joint range fans (right arm): shoulder +-2.3, elbow +-2.0, wrist +-1.2 rad around the straight-through direction
FAN = []
for s, R in ((0, 170), (1, 150), (2, 120)):
    J = X(P[1, s]); prev = X(P[1, s - 1]) if s else J - np.array([1, 0]); d = J - prev
    base = np.arctan2(-d[1], d[0]); lim = sim.QHI[s]; a0, a1 = base - lim, base + lim
    pts = [J] + [J + R * np.array([np.cos(t), -np.sin(t)]) for t in np.linspace(a0, a1, 90)]
    FAN.append((J, R, a0, a1))
    poly(pts, NAVY, 'fill-opacity=".09"')
    path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts[1:]), NAVY, 5, extra='stroke-opacity=".8"')
    for t in (a0, a1): e = J + R * np.array([np.cos(t), -np.sin(t)]); line(J[0], J[1], e[0], e[1], NAVY, 4, 'stroke-opacity=".8"')
# joint caps: the sheet continues around the elbow and wrist
for s in (1, 2):
    J = X(P[1, s]); f = 1 / (1 + ((sim.BOUND[s - 1] - s0) ** 2 + .15 * (.5 - w0) ** 2) / 0.12)
    circ(J[0], J[1], HW[s - 1], heat((f - .2) / .8), 'stroke="#4d5865" stroke-width="5"')
# right arm: one continuous skin sheet, colored by how strongly this touch spreads to each point (the sim's falloff)
a = 1
for s in range(3):
    A_, B_ = X(P[a, s]), X(P[a, s + 1]); d = unit(B_ - A_); n = np.array([d[1], -d[0]]); hw = HW[s]
    L0 = sum(SEG_LEN[:s]); NS = int(120 * SEG_LEN[s] / .5) + 10; NW = 24
    for i in range(NS):
        for j in range(NW):
            ua, ub = i / NS, (i + 1) / NS; wa, wb = j / NW, (j + 1) / NW
            arc = L0 + (ua + ub) / 2 * SEG_LEN[s]; w = (wa + wb) / 2
            f = 1 / (1 + ((arc - s0) ** 2 + .15 * (w - w0) ** 2) / 0.12); c = heat((f - .2) / .8)
            q4 = [A_ + u * (B_ - A_) + (0.5 - w_) * 2 * hw * n for u, w_ in ((ua, wa), (ub + .002, wa), (ub + .002, wb), (ua, wb))]
            poly(q4, c, f'stroke="{c}" stroke-width="2"')
    poly([A_ + hw * n, B_ + hw * n, B_ - hw * n, A_ - hw * n], "none", 'stroke="#4d5865" stroke-width="5" stroke-linejoin="round"')
for s in range(3): J = X(P[1, s]); circ(J[0], J[1], 15, "#2c3540")
# tendons: flexor on the w=0 edge, extensor on the w=1 edge, from mid-forearm across the wrist into the hand
TEN = []
for w, dash in ((0.17, ""), (0.83, 'stroke-dasharray="16 10"')):
    pts = [arm_point(1, t, w)[0] for t in np.linspace(.6, 1.1, 50)]
    path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), "#5e1309", 8, extra=dash); TEN.append(pts)
    p0 = pts[0]; d_ = unit(pts[0] - pts[3]); e = p0 + d_ * 46
    path(f"M {p0[0]:.1f} {p0[1]:.1f} L {e[0]:.1f} {e[1]:.1f}", "#5e1309", 8, extra='marker-end="url(#ahT)"')
# the touch: an outside disc pressing the forearm's upper edge (sim contact: radius 0.1, pressure 0.8)
tp, tn, _ = arm_point(1, s0, -0.75)
circ(tp[0], tp[1], 0.075 * sc * .55, "#3b3f45")
# electrodes and this touch's 8 readings as bars pointing out from each electrode
EL = []
for k, (e, w) in enumerate(sim.ELEC):
    p, n, s = arm_point(1, min(max(e, .012), sim.ARM_LEN - .012), w)
    o = n if w == 0 else -n; v = read[k]; L = 210 * v / read.max(); b0 = p + o * 24; b1 = b0 + o * L; dd = np.array([o[1], -o[0]]) * 16
    poly([b0 + dd, b1 + dd, b1 - dd, b0 - dd], RED); EL.append((p, b1, o))
    rect(p[0] - 17, p[1] - 17, 34, 34, GOLD, 6, 'stroke="#5a4300" stroke-width="4"')
# ----- labels with leader lines -----
LB = 52
def lab(tx_, ty_, s, lx, ly, px, py, anchor="middle", fill=INK, fw=800):
    T(tx_, ty_, s, LB, fw, fill, anchor); line(lx, ly, px, py, INK, 3); circ(px, py, 7, INK)
J0, J1, J2, J3 = (X(P[1, s]) for s in range(4))
def fanpt(i, deg): J, R = FAN[i][0], FAN[i][1]; t = np.radians(deg); return J + R * np.array([np.cos(t), -np.sin(t)])
def edge(arc, w): return arm_point(1, arc, w)[0]
# above the arm
T(560, 110, "skin: one continuous resistive sheet per arm", LB, 800); T(560, 170, "red: how this one touch spreads over the sheet", 50, 400)
p = edge(.22, -.05); line(900, 190, p[0], p[1], INK, 3); circ(p[0], p[1], 7, INK)
f = fanpt(0, 95); lab(f[0] + 20, f[1] - 70, "shoulder ±132°", f[0] + 20, f[1] - 52, f[0], f[1], fill=NAVY)
lab(tp[0] - 150, tp[1] - 110, "touch", tp[0] - 140, tp[1] - 100, tp[0] - 30, tp[1] - 20, "end")
b = EL[4][1]; lab(b[0] - 10, b[1] - 80, "electrode readings", b[0] - 10, b[1] - 66, b[0] + 4, b[1] - 8, fill=RED)
# right column
R0 = 2240
f = TEN[0][44]; T(R0, J3[1] - 30, "flexor tendon", LB, 800, "#5e1309"); T(R0, J3[1] + 30, "pull: wrist bends up", 50, 400, "#5e1309")
line(R0 - 12, J3[1] - 46, f[0], f[1], INK, 3); circ(f[0], f[1], 7, INK)
h = edge(1.05, .5); lab(R0, J3[1] + 140, "hand, 0.20", R0 - 12, J3[1] + 124, h[0] + 6, h[1] + 4, "start")
f = fanpt(2, -35); lab(R0, f[1] + 70, "wrist ±69°", R0 - 12, f[1] + 54, f[0], f[1], "start", NAVY)
f = TEN[1][24]; lab(R0, f[1] + 230, "extensor tendon", R0 - 12, f[1] + 214, f[0], f[1], "start", "#5e1309")
# below the arm
RB = 1000
p = edge(.25, 1.0); lab(p[0] - 60, RB, "upper arm, 0.50", p[0] - 30, RB - 50, p[0], p[1])
f = fanpt(1, -95); lab(f[0] + 90, RB, "elbow ±115°", f[0] + 60, RB - 50, f[0], f[1], fill=NAVY)
p = edge(.58, 1.0); lab(p[0] + 380, RB, "forearm, 0.45", p[0] + 330, RB - 50, p[0], p[1])
e = EL[1][0]; lab(380, RB + 90, "8 electrodes on the sheet edges", 520, RB + 40, e[0] - 10, e[1] + 14, "start", "#6b5100")
T(hx, X((0, 0.47))[1], "torso", 50, 700, "#56606b", "middle"); T(hx, X((0, 0.47))[1] + 58, "(no skin)", 50, 400, "#56606b", "middle")
T(hx, X((0, 0.47))[1] + 150, "left arm:", 50, 700, "#56606b", "middle"); T(hx, X((0, 0.47))[1] + 208, "same design", 50, 400, "#56606b", "middle")

# ----- bottom strip of the top half: what Qwen reads and writes -----
y0 = 1200
def tok(x, y, w, h, fill, txt):
    rect(x, y, w, h, fill, 18); T(x + w / 2, y + h / 2 + 18, txt, 50, 800, "#fff", "middle")
T(60, y0, "Each of 10 steps, Qwen3-8B reads 2 vectors and writes 1 token:", 54, 800)
tok(60, y0 + 40, 300, 90, NAVY, "joints"); tok(380, y0 + 40, 260, 90, RED, "skin"); tok(660, y0 + 40, 300, 90, "#3a3f46", "motor")
for i, s in enumerate(("6 angles", "6 speeds", "4 tendon", "tensions")): T(210, y0 + 190 + i * 56, s, 50, 400, INK, "middle")
for i, s in enumerate(("16 electrode", "readings")): T(510, y0 + 190 + i * 56, s, 50, 400, INK, "middle")
for i, s in enumerate(("1 of 16", "tokens")): T(810, y0 + 190 + i * 56, s, 50, 400, INK, "middle")
path(f"M 965 {y0+85} L 1040 {y0+85}", INK, 6, extra='marker-end="url(#ahK)"')
# motor token table
tx0, ty0 = 1090, y0 + 40; rows = ["shoulder +17°", "shoulder −17°", "elbow +17°", "elbow −17°", "pull flexor", "pull extensor"]
T(tx0 + 435, ty0 + 30, "left", 48, 800, INK, "middle"); T(tx0 + 555, ty0 + 30, "right", 48, 800, INK, "middle")
for i, r in enumerate(rows):
    yy = ty0 + 50 + i * 68
    T(tx0, yy + 50, r, 50, 400)
    for j, L in enumerate("ABCDEF"[i] + "GHIJKL"[i]):
        rect(tx0 + 395 + j * 120, yy + 6, 80, 58, "#3a3f46", 12); T(tx0 + 435 + j * 120, yy + 52, L, 46, 800, "#fff", "middle")
for i, (L, r) in enumerate((("M", "rest"), ("N", "dim left skin"), ("O", "dim right skin"), ("P", "undim skin"))):
    yy = ty0 + 50 + i * 68; xx = tx0 + 680
    rect(xx, yy + 6, 80, 58, "#3a3f46", 12); T(xx + 40, yy + 52, L, 46, 800, "#fff", "middle"); T(xx + 100, yy + 52, r, 50, 400)
# question and answer after the 10 steps
qx = 2230; qy = y0
T(qx, qy, "Then one question:", 54, 800)
rect(qx, qy + 30, 640, 260, "#2f7bf6", 40)
for i, s in enumerate(("How far along your", "arm is the touch, from", "shoulder (0) to", "fingertips (10)?")):
    T(qx + 36, qy + 88 + i * 58, s, 48, 500, "#fff")
import os as _os
_f = "v3/fig3c_named_s0.npz"
if _os.path.exists(_f):
    _d = np.load(_f); _m = _d["meta"]; _arcs = np.unique(_m[:, 1]); _a = _arcs[np.argmin(abs(_arcs - s0))]
    ANS = int(round(_d["EVp"][(_m[:, 0] == 1) & (_m[:, 1] == _a)].mean())); print("answer at", _a, _d["EVp"][(_m[:, 0] == 1) & (_m[:, 1] == _a)].round(1))
else: ANS = "?"
rect(qx + 270, qy + 310, 370, 100, "#e9ecf1", 40); T(qx + 455, qy + 378, f"Qwen: “{ANS}”", 50, 800, INK, "middle")
TOP_END = 1760
line(40, TOP_END, W - 40, TOP_END, "#d0d5db", 3)

# ======================= DATA =======================
import os
HAVE = all(os.path.exists(f"v3/fig3c_{k}_s0.npz") for k in ("named", "nameless"))
D = {k: np.load(f"v3/fig3c_{k}_s0.npz") for k in ("named", "nameless")} if HAVE else {}
ARCS = np.linspace(.03, sim.ARM_LEN - .03, 23)

if HAVE:
    def cvpred(Xm, y, k=5):
        Xm = np.asarray(Xm, float); y = np.asarray(y, float); n = len(y); idx = np.random.default_rng(0).permutation(n); p = np.zeros(n)
        def fit(Xa, ya, al):
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = ya.mean()
            U, S_, Vt = np.linalg.svd(Z, full_matrices=False); w = Vt.T @ (S_ / (S_ ** 2 + al) * (U.T @ (ya - ym)))
            return lambda Xb: ((Xb - mu) / sd) @ w + ym
        for f in range(k):
            te = idx[f::k]; tr = np.setdiff1d(idx, te); itr, iva = tr[: int(.8 * len(tr))], tr[int(.8 * len(tr)):]
            best = min(np.logspace(-1, 5, 13), key=lambda al: ((fit(Xm[itr], y[itr], al)(Xm[iva]) - y[iva]) ** 2).mean())
            p[te] = fit(Xm[tr], y[tr], best)(Xm[te])
        return p
    def wilson(k, n, z=1.96):
        p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, c - h, c + h
    def axes(px, py, pw, ph, xlim, ylim, xt, yt, xl, yl, xfmt=None, yfmt=None, grid=True, xl2=None, yo=135):
        xfmt = xfmt or (lambda t: f"{t:g}"); yfmt = yfmt or (lambda t: f"{t:g}")
        S = lambda x, y: (px + pw * (x - xlim[0]) / (xlim[1] - xlim[0]), py + ph * (ylim[1] - y) / (ylim[1] - ylim[0]))
        for t in xt:
            xx, _ = S(t, ylim[0])
            if grid: line(xx, py, xx, py + ph, "#eceef1", 2)
            line(xx, py + ph, xx, py + ph + 14, "#222", 3); T(xx, py + ph + 62, xfmt(t).replace("-", "−"), 48, 400, INK, "middle")
        for t in yt:
            _, yy = S(xlim[0], t)
            if grid: line(px, yy, px + pw, yy, "#eceef1", 2)
            line(px - 14, yy, px, yy, "#222", 3); T(px - 24, yy + 16, yfmt(t).replace("-", "−"), 48, 400, INK, "end")
        line(px, py + ph, px + pw, py + ph, "#222", 3); line(px, py, px, py + ph, "#222", 3)
        T(px + pw / 2, py + ph + 126, xl, 50, 400, INK, "middle")
        if xl2: T(px + pw / 2, py + ph + 184, xl2, 50, 400, INK, "middle")
        cx_, cy_ = px - yo, py + ph / 2
        T(cx_, cy_, yl, 50, 400, INK, "middle", f'transform="rotate(-90 {cx_} {cy_})"')
        return S
    def header(x, y, lines, anchor="start"):
        for i, s in enumerate(lines if isinstance(lines, (list, tuple)) else [lines]): T(x, y + i * 62, s, 54, 800, NAVY, anchor)
    TEN10 = lambda arc: 10 * arc / sim.ARM_LEN
    dn, dl = D["named"], D["nameless"]

    # ---------- row A: (a) every touch inside Qwen, both twins; (b) linear readout ----------
    ya = TOP_END + 90
    header(60, ya, "Inside Qwen, every touch lies on one smooth path from shoulder to fingertip, with or without body words")
    # position color key (shared): a real axis along the arm
    kx0, kx1, ky = 290, 290 + 1540, ya + 50
    for i in range(300): t = i / 299; rect(kx0 + t * (kx1 - kx0), ky, (kx1 - kx0) / 299 + 1, 34, vir(t))
    for nm, arc in (("shoulder", 0), ("elbow", sim.BOUND[0]), ("wrist", sim.BOUND[1]), ("fingertip", sim.ARM_LEN)):
        xx = kx0 + (kx1 - kx0) * arc / sim.ARM_LEN; line(xx, ky + 34, xx, ky + 50, "#222", 3)
        T(xx, ky + 100, nm, 48, 700, INK, "start" if arc == 0 else "end" if arc == sim.ARM_LEN else "middle")
    T(kx0 - 24, ky + 30, "color:", 50, 700, INK, "end")
    PA = 640; pyA = ya + 290
    for k, (arm, ttl, colr) in enumerate((("named", "questions say “elbow”, “wrist”", NAVY), ("nameless", "no body words: “line 2”, “bend”", PURP))):
        d = D[arm]; m = d["meta"]; sel = m[:, 0] == 1; Hs = d["H"][sel]; arc = m[sel, 1]
        C = np.stack([Hs[arc == a].mean(0) for a in ARCS]); mu = C.mean(0)
        U, S_, Vt = np.linalg.svd(C - mu, full_matrices=False); Z = (Hs - mu) @ Vt[:2].T; Zc = (C - mu) @ Vt[:2].T
        if Zc[0, 0] > Zc[-1, 0]: Z[:, 0] *= -1; Zc[:, 0] *= -1
        if Zc[11, 1] > Zc[0, 1]: Z[:, 1] *= -1; Zc[:, 1] *= -1
        lim = np.abs(Z).max() * 1.06; step = [s for s in (5, 10, 20, 25, 50, 100, 200) if lim / s <= 3][0]
        tk = [t for t in np.arange(-10 * step, 10 * step + 1, step) if abs(t) <= lim]
        px = 290 + k * 900; py = pyA
        T(px + PA / 2, py - 30, ttl, 50, 800, colr, "middle")
        S = axes(px, py, PA, PA, (-lim, lim), (-lim, lim), tk, tk, "hidden state, direction 1", "hidden state, direction 2", lambda t: f"{t:.0f}", lambda t: f"{t:.0f}")
        for i in np.argsort(np.random.default_rng(0).random(len(Z))): circ(*S(*Z[i]), 10, vir(arc[i] / sim.ARM_LEN), 'fill-opacity=".9" stroke="#fff" stroke-width="1.5"')
        for nm, b in (("elbow", sim.BOUND[0]), ("wrist", sim.BOUND[1])):
            p = np.array([np.interp(b, ARCS, Zc[:, 0]), np.interp(b, ARCS, Zc[:, 1])]); sx, sy = S(*p)
            circ(sx, sy, 30, "none", 'stroke="#111" stroke-width="6"')
            T(sx + (0 if nm == "elbow" else -44), sy + (86 if nm == "elbow" else 18), nm, 50, 800, INK, "middle" if nm == "elbow" else "end")
    T(290 + 900 + PA + 0, pyA + PA + 184, "", 10)
    T(290 + PA + 130, pyA + PA + 184, "each dot: one touch on the right arm (23 places × 12 arm postures), layer 18 of 36", 48, 400, INK, "middle")
    # (b) linear readout of position from the same hidden states (both arms, 5-fold cross-validated)
    y_true = TEN10(dn["meta"][:, 1]); y_hat = cvpred(dn["H"], y_true); R2 = 1 - ((y_hat - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    px = 2240; py = pyA
    T(px + PA / 2 - 60, pyA - 92, "A linear readout recovers position", 50, 800, NAVY, "middle")
    T(px + PA / 2 - 60, pyA - 30, "same hidden states (body words), both arms", 50, 400, INK, "middle")
    S = axes(px, py, PA, PA, (0, 10), (0, 10), range(0, 11, 2), range(0, 11, 2), "true position", "readout of position", xl2="(0 shoulder, 10 fingertip)")
    T(px - 52, py + PA / 2, "", 10)
    cx_, cy_ = px - 52, py + PA / 2; T(cx_, cy_, "", 10)
    line(*S(0, 0), *S(10, 10), "#888", 3, 'stroke-dasharray="10 10"')
    for i in np.argsort(np.random.default_rng(1).random(len(y_true))): circ(*S(y_true[i], np.clip(y_hat[i], 0, 10)), 9, vir(y_true[i] / 10), 'fill-opacity=".85"')
    T(px + 30, py + 80, f"R² = {R2:.2f}", 58, 800, INK)
    # second y-label line
    out[-1:] = out[-1:]
    yb = pyA + PA + 290

    # ---------- row B ----------
    PB = 540; pyB = yb + 170
    cols = [300, 1080, 1880, 2480]
    # (c) steering: blend the injected skin vectors of an upper-arm touch into those of a hand touch
    px = cols[0]; header(60, yb, ["Blend upper-arm touch into", "hand touch: answer rises"])
    S = axes(px, pyB, PB, PB, (0, 1), (0, 10), (0, .5, 1), range(0, 11, 2), "share of hand-touch signal", "answer: position 0–10", lambda t: f"{t*100:.0f}%", xl2="in the injected skin vector")
    xs = np.linspace(0, 1, 11); MU = {}
    for arm, colr in (("named", NAVY), ("nameless", PURP)):
        Ys = np.array([json.load(open(f))["steer"]["inject"]["pos"] for f in sorted(glob.glob(f"$BODY_RUNS{'named' if arm=='named' else 'namel'}*/interp_Qwen3-8B_{arm}_s*.json"))])
        lo, hi, mu_ = Ys.min(0), Ys.max(0), Ys.mean(0); MU[arm] = mu_
        poly([S(x, v) for x, v in zip(xs, hi)] + [S(x, v) for x, v in zip(xs[::-1], lo[::-1])], colr, 'fill-opacity=".18"')
        path("M " + " L ".join("%.1f %.1f" % S(x, v) for x, v in zip(xs, mu_)), colr, 8)
    T(px + 24, pyB + 60, "band: 3 training runs", 48, 400)
    T(*S(.02, MU["named"][0] + .75), "body words", 48, 700, NAVY)
    T(*S(.02, MU["nameless"][0] - 1.0), "no body words", 48, 700, PURP)
    # (d1) posture: the left-right answer for a fixed skin spot follows where the touch really is
    px = cols[1]; header(cols[1] - 190, yb, ["Same skin spot, new posture:", "left-right answer follows"])
    pm = dn["pm"]; ex = dn["EVx"]; tx_ = 10 * (dn["truex"] + 1.5) / 3; ok = ~np.isnan(tx_)
    S = axes(px, pyB, PB, PB, (0, 10), (0, 10), range(0, 11, 2), range(0, 11, 2), "true left-right place", "answer: left-right 0–10")
    line(*S(0, 0), *S(10, 10), "#888", 3, 'stroke-dasharray="10 10"')
    rs = []
    for sp in np.unique(pm[:, 1]):
        for a in (0, 1):
            m_ = ok & (pm[:, 1] == sp) & (pm[:, 0] == a); rs.append(np.corrcoef(tx_[m_], ex[m_])[0, 1])
            for i in np.where(m_)[0]: circ(*S(tx_[i], ex[i]), 10, vir(sp / sim.ARM_LEN), 'fill-opacity=".9" stroke="#fff" stroke-width="1.5"')
    T(px + PB - 20, pyB + PB - 90, f"r = {np.mean(rs):.2f}", 54, 800, INK, "end")
    T(px + PB - 20, pyB + PB - 34, "within one skin spot", 48, 400, INK, "end")
    # (d2) self vs other
    px = cols[2]; header(cols[2] - 150, yb, ["Own touch told apart", "from outside touch"])
    S = axes(px, pyB, 420, PB, (0, 2), (0, 100), [], range(0, 101, 25), "", "correct answers", yfmt=lambda t: f"{t:.0f}%", yo=160)
    for g, (cond, lab_) in enumerate(((True, "own fingertip"), (False, "outside object"))):
        for j, (arm, colr) in enumerate((("named", NAVY), ("nameless", PURP))):
            d = D[arm]; m_ = d["is_self"] == cond; k_ = int((d["said_yes"][m_] == cond).sum()); n_ = int(m_.sum())
            p_, l_, u_ = wilson(k_, n_); x0 = g + .14 + j * .36; x1 = x0 + .32
            (X0, Y0), (X1, Y1) = S(x0, p_ * 100), S(x1, 0); rect(X0, Y0, X1 - X0, Y1 - Y0, colr)
            if g == 0: xm_ = (X0 + X1) / 2 + 17; T(xm_, Y1 - 24, "body words" if j == 0 else "no body words", 48, 700, "#fff", "start", f'transform="rotate(-90 {xm_:.0f} {Y1 - 24:.0f})"')
            xm = (X0 + X1) / 2; (_, YL), (_, YU) = S(0, l_ * 100), S(0, u_ * 100); line(xm, YL, xm, YU, INK, 5); line(xm - 14, YL, xm + 14, YL, INK, 5); line(xm - 14, YU, xm + 14, YU, INK, 5)
        for i_, w_ in enumerate(lab_.split()): T(S(g + .5, 0)[0], pyB + PB + 62 + i_ * 56, w_, 48, 700, INK, "middle")
    T(px + 210, pyB + PB + 190, "what touched the arm", 50, 400, INK, "middle")
    # (e) closed loop
    cl = json.load(open("closed_loop_ci.json"))
    px = cols[3]; header(cols[3] - 120, yb, ["Qwen drives the", "body to a goal"])
    S = axes(px, pyB, 470, PB, (0, 3), (0, 100), [], range(0, 101, 25), "", "goal reached", yfmt=lambda t: f"{t:.0f}%", yo=160)
    for g, (k, lab_) in enumerate((("reach", "touch elbow"), ("find", "find object"), ("wall", "press wall"))):
        for j, (kk, nn, colr) in enumerate((tuple(cl["model"][f"named|{k}"]) + (NAVY,), tuple(cl["model"][f"nameless|{k}"]) + (PURP,), tuple(cl["random"][k]) + (GRAY,))):
            p_, l_, u_ = wilson(kk, nn); x0 = g + .08 + j * .28; x1 = x0 + .26
            (X0, Y0), (X1, Y1) = S(x0, p_ * 100), S(x1, 0); rect(X0, Y0, X1 - X0, Y1 - Y0, colr)
            if g == 2 and j == 2: T(X0 + 20 + (X1 - X0) / 2, Y0 - 50, "random moves", 48, 700, "#5b636c", "start", f'transform="rotate(-90 {X0 + 20 + (X1 - X0) / 2:.0f} {Y0 - 50:.0f})"')
            xm = (X0 + X1) / 2; (_, YL), (_, YU) = S(0, l_ * 100), S(0, u_ * 100); line(xm, YL, xm, YU, INK, 4); line(xm - 10, YL, xm + 10, YL, INK, 4); line(xm - 10, YU, xm + 10, YU, INK, 4)
        for i_, w_ in enumerate(lab_.split()): T(S(g + .5, 0)[0], pyB + PB + 62 + i_ * 56, w_, 48, 700, INK, "middle")

# ======================= render =======================
def render(Hh=H):
    defs = f'''<defs>
<linearGradient id="torso" x1="0" x2="1"><stop offset="0" stop-color="#c3ccd8"/><stop offset=".5" stop-color="#e8ecf1"/><stop offset="1" stop-color="#b7c2cf"/></linearGradient><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset=".55" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff" stop-opacity="1"/></linearGradient>
<radialGradient id="head" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#f1f3f7"/><stop offset="1" stop-color="#b3bfcd"/></radialGradient>
''' + ''.join(f'<marker id="ah{n}" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4,2 L0,4 z" fill="{c}"/></marker>' for n, c in (('K', '#111'), ('T', '#5e1309'), ('N', NAVY), ('P', PURP))) + '</defs>'
    svg = f'<html><head><meta charset="utf-8"></head><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="{FONT}">{defs}<rect width="{W}" height="{Hh}" fill="#fff"/>{"".join(out)}</svg></body></html>'
    F = "$BODY_WORK/fig/body_instill_v3c"
    open(F + ".html", "w").write(svg)
    CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    subprocess.run([CH, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{Hh}", f"--screenshot={F}.png", f"file://{F}.html"], capture_output=True, timeout=90)
    print("ok")
T(cols[3] + 235, pyB + PB + 190, "whiskers: 95% interval", 48, 400, INK, "middle")
render(int(pyB + PB + 220) if HAVE else TOP_END + 20)
