"""compute/fig/body_instill_v3b.png: setup / smooth arm inside Qwen / closed-loop acting. Hand-built SVG, Chrome for Testing."""
import json, glob, subprocess, numpy as np, sys
from matplotlib import cm
sys.path.insert(0, "."); import sim3 as sim
from body import fk
RUNS = "$BODY_RUNS*"
W, H = 2600, 2720
FONT = "'Helvetica Neue',Helvetica,Arial,sans-serif"
NAVY, RED, GOLD, GRAY, SKIN, INK = "#1f3f73", "#c0392b", "#c9971c", "#a7aeb6", "#e9c9a6", "#111"
out = []
def T(x, y, s, fs=48, fw=400, fill=INK, anchor="start", extra=""):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{fs}" font-weight="{fw}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def rect(x, y, w, h, fill, rx=0, extra=""): out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>')
def line(x1, y1, x2, y2, c, w=4, extra=""): out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {extra}/>')
def circ(x, y, r, fill, extra=""): out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" {extra}/>')
def path(d, c, w=6, fill="none", extra=""): out.append(f'<path d="{d}" stroke="{c}" stroke-width="{w}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round" {extra}/>')
def col(t): r, g, b, _ = cm.viridis(0.05 + 0.8 * t); return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"

# ================= panel 1: the setup =================
q = np.array([-1.35, 0.1, 0.1, -0.3, -1.05, -0.45]); P = fk(q)
sc, ox, oy = 360, 330, 400
X = lambda p: np.array([ox + sc * p[0], oy - sc * p[1]])
tx, ty = X((0, 0))
out.append(f'<rect x="{tx-105:.0f}" y="{ty-185:.0f}" width="210" height="400" rx="70" fill="url(#torso)"/>')
hx, hy = X((0, .8)); out.append(f'<ellipse cx="{hx:.0f}" cy="{hy+5:.0f}" rx="62" ry="74" fill="url(#head)"/>')
rect(hx - 28, hy + 55, 56, 60, "#c3ccd8")
WID = [62, 52, 40]
for a in range(2):
    for s in range(3):
        (x1, y1), (x2, y2) = X(P[a, s]), X(P[a, s + 1])
        line(x1, y1, x2, y2, "#9aa7b8", WID[s] + 8); line(x1, y1, x2, y2, "url(#limb)" if False else "#d6dde7", WID[s])
        line(x1, y1, x2, y2, "#f3f6fa", WID[s] * .35, 'stroke-opacity=".8"')
    for k in range(3): circ(*X(P[a, k + 1]) if k < 2 else X(P[a, k]), 0, "none")
# tendon on the right hand: pulled from the forearm, across the wrist, into the finger
f0, f1, f2, f3 = X(P[1, 1]), X(P[1, 2]), X(P[1, 3]), None
def off(pA, pB, t, d):
    v = pB - pA; n = np.array([-v[1], v[0]]) / np.linalg.norm(v); return pA + t * v + d * n
ta, tb, tc = off(f0, f1, .45, -14), off(f1, f2, 0, -14), off(f1, f2, 1.0, -10)
path(f"M {ta[0]:.0f} {ta[1]:.0f} L {tb[0]:.0f} {tb[1]:.0f} L {tc[0]:.0f} {tc[1]:.0f}", RED, 7)
# electrodes: 4 places along each arm, on both edges
for a in range(2):
    for e in sim.EPOS:
        arc = min(e, sim.ARM_LEN - 1e-4); s = sim.seg_of(arc); u = (arc - sum(sim.SEG_LEN[:s])) / sim.SEG_LEN[s]
        A_, B_ = X(P[a, s]), X(P[a, s + 1])
        for d in (-1, 1):
            p = off(A_, B_, min(u, .97), d * (WID[s] / 2 + 4))
            rect(p[0] - 11, p[1] - 11, 22, 22, GOLD, 4, 'stroke="#6b4e00" stroke-width="3"')
# touch on right forearm
tp = off(X(P[1, 1]), X(P[1, 2]), .62, 0)
circ(tp[0], tp[1], 38, RED, 'fill-opacity=".35"'); circ(tp[0], tp[1], 22, RED, 'fill-opacity=".85"')
# Qwen box + flows
qx, qy, qw, qh = 1020, 200, 520, 300
# inputs
path(f"M 760 230 C 860 230, 900 300, {qx-12} 300", GOLD, 10, extra='marker-end="url(#ahG)"')
T(800, 200, "skin", 52, 800, "#8a6400")
path(f"M 760 520 C 860 520, 900 400, {qx-12} 400", NAVY, 10, extra='marker-end="url(#ahN)"')
T(780, 600, "joints", 52, 800, NAVY)
rect(qx, qy, qw, qh, "url(#qwen)", 36)
T(qx + qw / 2, qy + qh / 2 + 22, "Qwen3-8B", 64, 800, "#fff", "middle")
# output 1: motor token -> joint moves (loop back to the body)
mx, my = 1760, 210
path(f"M {qx+qw+12} 300 C 1620 300, 1640 {my+60}, {mx-30} {my+60}", INK, 9, extra='marker-end="url(#ahK)"')
# small arm pictogram with a turning arrow at the elbow
ax0, ay0 = mx + 40, my + 110
line(ax0, ay0, ax0 + 150, ay0, "#9aa7b8", 44); line(ax0, ay0, ax0 + 150, ay0, "#d6dde7", 34)
line(ax0 + 150, ay0, ax0 + 230, ay0 - 120, "#9aa7b8", 36); line(ax0 + 150, ay0, ax0 + 230, ay0 - 120, "#d6dde7", 26)
path(f"M {ax0+250} {ay0-30} A 100 100 0 0 0 {ax0+190} {ay0-150}", INK, 8, extra='marker-end="url(#ahK)"')
T(mx + 330, my + 95, "move", 56, 800)
# loop arrow back to the body along the top
path(f"M {mx+140} {my-20} C {mx+60} 40, 900 60, 560 150", "#555", 6, extra='stroke-dasharray="4 16" marker-end="url(#ahS)"')
# output 2: answer bubble
bx_, by_ = 1760, 470
path(f"M {qx+qw+12} 400 C 1620 400, 1640 {by_+70}, {bx_-30} {by_+70}", INK, 9, extra='marker-end="url(#ahK)"')
rect(bx_, by_, 380, 140, "#e9ecf1", 70); path(f"M {bx_+40} {by_+120} L {bx_+10} {by_+165} L {bx_+90} {by_+130} Z", "#e9ecf1", 1, "#e9ecf1")
T(bx_ + 190, by_ + 90, "forearm", 60, 700, INK, "middle")

# ================= panel 2: the arm inside Qwen =================
A = np.linspace(.03, sim.ARM_LEN - .03, 23)
emb = {}
for arm in ("named", "nameless"):
    D = np.mean([np.load(f)[1][23:, 23:] for f in sorted(glob.glob(f"{RUNS}/rdm_Qwen3-8B_{arm}_s*.npy"))], 0)
    n = len(D); J = np.eye(n) - 1 / n; B = -.5 * J @ (D ** 2) @ J; w, v = np.linalg.eigh(B)
    Y = v[:, ::-1][:, :2] * np.sqrt(w[::-1][:2])
    Y *= np.sign(Y[-1] - Y[0])  # fingertip toward +x,+y... then fix by hand below
    emb[arm] = Y
for arm in emb:  # shoulder on the left, elbow at the bottom
    Y = emb[arm]
    if Y[0, 0] > Y[-1, 0]: Y[:, 0] *= -1
    if Y[11, 1] > Y[0, 1]: Y[:, 1] *= -1
    emb[arm] = Y
y2 = 820
T(60, y2, "Inside Qwen, touches along the arm form one continuous curve", 60, 800, NAVY)
lo, hi = -60, 60; pw = 860
for k, (arm, ttl) in enumerate((("named", "with part names"), ("nameless", "no body words"))):
    px = 330 + k * 1220; py = y2 + 110
    Y = emb[arm]; m = (Y.max(0) + Y.min(0)) / 2; Y = Y - m
    S = lambda p: (px + pw * (p[0] - lo) / (hi - lo), py + pw * (hi - p[1]) / (hi - lo))
    out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{pw}" fill="none" stroke="#222" stroke-width="3"/>')
    for t in (-50, -25, 0, 25, 50):
        xx, _ = S((t, 0)); _, yy = S((0, t))
        line(xx, py + pw, xx, py + pw + 16, "#222", 3); T(xx, py + pw + 66, f"{t}".replace("-", "−"), 44, 400, INK, "middle")
        line(px - 16, yy, px, yy, "#222", 3); T(px - 26, yy + 15, f"{t}".replace("-", "−"), 44, 400, INK, "end")
        line(xx, py, xx, py + pw, "#eef0f3", 2); line(px, yy, px + pw, yy, "#eef0f3", 2)
    T(px + pw / 2, py - 24, ttl, 54, 800, INK, "middle")
    T(px + pw / 2, py + pw + 135, "hidden-state axis 1", 48, 400, INK, "middle")
    cx_, cy_ = px - 130, py + pw / 2
    T(cx_, cy_, "hidden-state axis 2", 48, 400, INK, "middle", f'transform="rotate(-90 {cx_} {cy_})"')
    # fine interpolated curve colored by position
    fine = np.linspace(A[0], A[-1], 300); Yf = np.stack([np.interp(fine, A, Y[:, 0]), np.interp(fine, A, Y[:, 1])], 1)
    for i in range(len(fine) - 1):
        (x1, y1_), (x2, y2_) = S(Yf[i]), S(Yf[i + 1]); line(x1, y1_, x2, y2_, col((fine[i] - A[0]) / (A[-1] - A[0])), 16)
    for i in range(len(A)): circ(*S(Y[i]), 9, "#fff", f'stroke="{col(i/22)}" stroke-width="5"')
    for nm, arc, dx, dy, anc in (("shoulder", A[0], 0, -40, "middle"), ("elbow", sim.BOUND[0], 0, 80, "middle"),
                                 ("wrist", sim.BOUND[1], -40, 0, "end"), ("fingertip", A[-1], -40, 0, "end")):
        p = np.array([np.interp(arc, A, Y[:, 0]), np.interp(arc, A, Y[:, 1])]); sx, sy = S(p)
        if nm in ("elbow", "wrist"): circ(sx, sy, 20, "none", 'stroke="#111" stroke-width="6"')
        lab = {"shoulder": (sx + dx, sy + dy), "elbow": (sx, sy + 80), "wrist": (sx - 46, sy - 30), "fingertip": (sx - 34, sy + 16)}[nm]
        T(*lab, nm, 48, 800, INK, anc)
    emb[arm + "_S"] = None

# ================= panel 3: acting =================
cl = json.load(open("closed_loop_ci.json"))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, c - h, c + h
y3 = 2010
T(60, y3, "Qwen drives the body to a goal", 60, 800, NAVY)
bx0, bw = 860, 1560; tasks = [("touch own elbow or wrist", "reach"), ("find a hidden object", "find"), ("press a wall gently", "wall")]
top = y3 + 50
for i, (lb, k) in enumerate(tasks):
    yy = top + i * 165
    T(bx0 - 30, yy + 82, lb, 48, 700, INK, "end")
    km = cl["model"][f"named|{k}"][0] + cl["model"][f"nameless|{k}"][0]; nm_ = cl["model"][f"named|{k}"][1] + cl["model"][f"nameless|{k}"][1]
    for j, (kk, nn, c_, h_) in enumerate(((km, nm_, NAVY, 70), (*cl["random"][k], GRAY, 50))):
        p, l, u = wilson(kk, nn); by = yy + (0 if j == 0 else 76)
        rect(bx0, by, bw * p, h_, c_, 4)
        line(bx0 + bw * l, by + h_ / 2, bx0 + bw * u, by + h_ / 2, INK, 5)
        for e in (l, u): line(bx0 + bw * e, by + h_ / 2 - 14, bx0 + bw * e, by + h_ / 2 + 14, INK, 5)
        if i == 0: T(bx0 + bw * u + 24 if j else bx0 + 24, by + h_ / 2 + 17, "random moves" if j else "Qwen3-8B", 44, 800, "#5b636c" if j else "#fff")
ax_y = top + 3 * 165 - 20
line(bx0, top - 20, bx0, ax_y, "#222", 3); line(bx0, ax_y, bx0 + bw, ax_y, "#222", 3)
for t in range(0, 101, 25):
    xx = bx0 + bw * t / 100; line(xx, ax_y, xx, ax_y + 16, "#222", 3); T(xx, ax_y + 64, f"{t}%", 44, 400, INK, "middle")
T(bx0 + bw / 2, ax_y + 130, "episodes that reach the goal", 48, 400, INK, "middle")

defs = f'''<defs>
<linearGradient id="torso" x1="0" x2="1"><stop offset="0" stop-color="#b9c4d2"/><stop offset=".45" stop-color="#e3e8ef"/><stop offset="1" stop-color="#aab6c6"/></linearGradient>
<radialGradient id="head" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#eef1f6"/><stop offset="1" stop-color="#a9b5c5"/></radialGradient>
<linearGradient id="qwen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3a5b94"/><stop offset="1" stop-color="#1a3463"/></linearGradient>
''' + "".join(f'<marker id="ah{n}" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L5,2.5 L0,5 z" fill="{c}"/></marker>'
              for n, c in (("G", GOLD), ("N", NAVY), ("K", INK), ("S", "#555"))) + "</defs>"
svg = f'''<html><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}">{defs}
<rect width="{W}" height="{H}" fill="#fff"/>{"".join(out)}</svg></body></html>'''
F = "$BODY_WORK/fig/body_instill_v3b"
open(F + ".html", "w").write(svg)
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
subprocess.run([CH, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}", f"--screenshot={F}.png", f"file://{F}.html"], capture_output=True, timeout=60)
print("ok")
