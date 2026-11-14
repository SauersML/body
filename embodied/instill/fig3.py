"""compute/fig/body_instill_v3.png: hand-built SVG, rendered with Chrome for Testing."""
import json, glob, subprocess, numpy as np, sys
sys.path.insert(0, ".")
import sim3 as sim
from body import fk

RUNS = "$BODY_RUNS*"
R = {}
for f in glob.glob(f"{RUNS}/interp_Qwen3-*.json"):
    tag = f.split("interp_")[1][:-5]; m, a, s = tag.rsplit("_", 2)
    R.setdefault((m, a), []).append((json.load(open(f.replace("interp_", "train_"))), json.load(open(f))))
M8 = "Qwen3-8B"
W, H = 2400, 2860
FONT = "'Inter','SF Pro Display','Helvetica Neue',Arial,sans-serif"
NAVY, RED, GOLD, GRAY, TEAL = "#1f3f73", "#b8322a", "#d99a1e", "#9aa3ad", "#2a8c82"
out = []
T = lambda x, y, s, fs=46, fw=400, fill="#111", anchor="start", extra="": out.append(
    f'<text x="{x:.0f}" y="{y:.0f}" font-size="{fs}" font-weight="{fw}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def rect(x, y, w, h, fill, rx=0, extra=""): out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>')
def line(x1, y1, x2, y2, c, w=4, extra=""): out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {extra}/>')
def header(y, s): T(60, y, s, 64, 800, NAVY)

def runs(m, a): return R.get((m, a), [])
def agg(vals): v = np.array(vals, float); return v.mean(), (v.std() if len(v) > 1 else 0)

# ---------------- method: the body and the loop ----------------
rect(30, 30, W - 60, 800, "#eef2f8", 30)
T(70, 110, "A simulated body wired into Qwen3-8B", 64, 800, NAVY)
q = np.array([-0.5, -0.9, -0.3, 0.75, 1.4, 0.6]); P = fk(q); sc, ox, oy = 210, 400, 620
X = lambda p: (ox + sc * p[0], oy - sc * p[1])
tx, ty = X((0, 0)); rect(tx - 70, ty - 105, 140, 230, "#c9d3e0", 40)
hx, hy = X((0, .8)); out.append(f'<ellipse cx="{hx}" cy="{hy}" rx="50" ry="60" fill="#c9d3e0"/>')
for a in range(2):
    for s in range(3):
        (x1, y1), (x2, y2) = X(P[a, s]), X(P[a, s + 1]); line(x1, y1, x2, y2, "#7d8fa8", 40 - 7 * s); line(x1, y1, x2, y2, "#f4d9a8", 28 - 7 * s)
    for e in sim.EPOS:
        ex, ey = X(sim.landmark_point(q, a, min(e, sim.ARM_LEN - 1e-6))); out.append(f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="12" fill="{NAVY}"/>')
cx, cy = X(sim.landmark_point(q, 0, .7)); out.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="38" fill="{RED}" fill-opacity=".55"/>')
T(cx - 50, cy - 55, "touch", 44, 800, RED, "end")
T(70, 790, "8 electrodes read each arm's skin", 40, 700, NAVY)
# tokens row
x0, y0 = 860, 190
T(x0, y0 - 25, "each moment, three tokens", 44, 700)
labs = [("posture", "#dfe7f2", NAVY), ("skin", "#f6e2df", RED), ("motor", "#f6ecd6", "#8a5a00")]
for k in range(3):
    for j, (lb, bg, fg) in enumerate(labs):
        xx = x0 + k * 460 + j * 148; rect(xx, y0, 138, 90, bg, 16); T(xx + 69, y0 + 58, lb, 33, 700, fg, "middle")
T(x0 + 3 * 460 - 20, y0 + 60, "…", 60, 800)
rect(x0, y0 + 125, 1420, 90, NAVY, 18); T(x0 + 710, y0 + 185, "Qwen3-8B", 50, 800, "#fff", "middle")
T(x0, y0 + 290, "posture and skin enter through a linear map,", 40); T(x0, y0 + 340, "capped at half the size of the model's own activations", 40)
T(x0, y0 + 420, "motor tokens move joints or pull the hand's tendons;", 40); T(x0, y0 + 470, "the next tokens report what happened", 40)
T(x0, y0 + 550, "trained to:", 40, 800)
for j, lb in enumerate(["copy goal-directed moves", "predict the next sensation", "answer questions"]):
    T(x0 + 230 + j * 420, y0 + 550, lb if j == 0 else lb, 36, 600, "#333") if False else None
T(x0 + 230, y0 + 550, "copy goal-directed moves,  predict the next sensation,  answer questions", 36, 600, "#333")
out.append(f'<path d="M {x0 - 20} {y0 + 45} C {x0 - 160} {y0 + 45}, 760 300, 700 400" stroke="#8a5a00" stroke-width="6" fill="none" marker-end="url(#ah)"/>')

# ---------------- result 1: acting ----------------
y = 920; header(y, "The model moves its own body to reach a goal")
tasks = [("touch own elbow or wrist with the other hand", "reach", .085), ("find a hidden object by touch", "find", .045), ("press the wall, but not too hard", "wall", .04)]
for i, (lbl, k, rnd) in enumerate(tasks):
    yy = y + 70 + i * 110; T(60, yy + 58, lbl, 42)
    m_, s_ = agg([t["closed_loop"][k][0] for a in ("named", "nameless") for t, _ in runs(M8, a)])
    bx = 1080; bw = 1100
    rect(bx, yy + 10, bw * m_, 60, NAVY, 8); rect(bx, yy + 75, bw * rnd, 22, GRAY, 4)
    T(bx + bw * m_ + 20, yy + 58, f"{100 * m_:.0f}%", 46, 800, NAVY)
T(1080, y + 425, "dark: model in control", 38, 700, NAVY); T(1520, y + 425, "gray: random motor tokens", 38, 700, "#6b7480")

# ---------------- result 2: continuous arm ----------------
y = 1420; header(y, "Inside, the arm becomes one smooth line, with or without part names")
def rdm(m, a_, li):
    fs = sorted(glob.glob(f"{RUNS}/rdm_{m}_{a_}_s*.npy")); D = np.mean([np.load(f)[li][23:, 23:] for f in fs], 0); return D / D.max()
def heat(x, y, D, size, title, col):
    n = len(D); c = size / n
    for i in range(n):
        for j in range(n):
            v = D[i, j]; r_ = int(255 - (255 - int(col[1:3], 16)) * (1 - v)); g_ = int(255 - (255 - int(col[3:5], 16)) * (1 - v)); b_ = int(255 - (255 - int(col[5:7], 16)) * (1 - v))
            rect(x + j * c, y + i * c, c + .6, c + .6, f"rgb({r_},{g_},{b_})")
    for bd in sim.BOUND:
        k = np.searchsorted(np.linspace(.03, sim.ARM_LEN - .03, 23), bd) * c
        line(x + k, y, x + k, y + size, "#fff", 4); line(x, y + k, x + size, y + k, "#fff", 4)
    T(x + size / 2, y - 22, title, 40, 800, col, "middle")
hs, hy0 = 520, y + 140
heat(80, hy0, rdm(M8, "named", 0), hs, "skin signal as given", "#55606e")
heat(80 + hs + 80, hy0, rdm(M8, "named", 1), hs, "with part names", NAVY)
heat(80 + 2 * (hs + 80), hy0, rdm(M8, "nameless", 1), hs, "no body words", TEAL)
T(80, hy0 + hs + 60, "each square: two touches on the right arm, shoulder (top left) to fingertips; dark = similar inside Qwen3-8B", 36, 600, "#333")
T(80, hy0 + hs + 105, "white lines: elbow and wrist. A smooth dark diagonal is a continuous map; dark boxes would be categories.", 36, 600, "#333")
# joint contrast bars
bx, by = 1900, y + 110
T(bx, by + 30, "jump in distance", 40, 800); T(bx, by + 78, "across a joint (1× = none)", 40, 800)
items = [("as given", [i["boundary_contrast"]["inject"] for a in ("named", "nameless") for _, i in runs(M8, a)], "#55606e"),
         ("part names", [i["boundary_contrast"]["18"] for _, i in runs(M8, "named")], NAVY),
         ("no words", [i["boundary_contrast"]["18"] for _, i in runs(M8, "nameless")], TEAL)]
for j, (lb, vals, col) in enumerate(items):
    m_, s_ = agg(vals); yy = by + 130 + j * 150; full = 300 / .7
    T(bx, yy + 10, lb, 38, 700, col); rect(bx, yy + 30, full * (m_ - 1), 60, col, 8)
    line(bx + full * (m_ - 1 - s_), yy + 60, bx + full * (m_ - 1 + s_), yy + 60, "#111", 4)
    T(bx + full * (m_ - 1 + s_) + 20, yy + 75, f"{m_:.2f}×", 44, 800, col)

# ---------------- result 3: steering ----------------
y = 2340; header(y, "Blending an upper-arm touch into a hand touch slides the answer, no jump")
px, py, pw, ph = 230, y + 70, 1150, 360
rect(px, py, pw, ph, "#fafbfd")
for a, col in (("named", NAVY), ("nameless", TEAL)):
    C = np.array([i["steer"]["inject"]["pos"] for _, i in runs(M8, a)]); mu = C.mean(0); al = np.linspace(0, 1, len(mu))
    out.append(f'<polyline points="{" ".join(f"{px + pw * x:.0f},{py + ph - ph * v / 8:.0f}" for x, v in zip(al, mu))}" fill="none" stroke="{col}" stroke-width="9"/>')
T(px, py + ph + 55, "upper-arm touch", 38, 700, "#333"); T(px + pw, py + ph + 55, "hand touch", 38, 700, "#333", "end")
T(px + pw / 2, py + ph + 55, "skin input blended →", 38, 600, "#333", "middle")
T(px - 30, py + ph / 2, "answer, 0 to 10", 36, 600, "#333", "middle", f'transform="rotate(-90 {px - 30} {py + ph / 2})"')
T(px + 30, py + 60, "part names", 40, 800, NAVY); T(px + 30, py + 110, "no words", 40, 800, TEAL)

# ---------------- result 4: integration ----------------
bx = 1500
T(bx, y + 100, "Posture moves where the touch", 46, 800, NAVY); T(bx, y + 155, "is felt, computed inside the LLM", 46, 800, NAVY)
T(bx, y + 215, "readout of the touch's left-right", 36, 600, "#333"); T(bx, y + 255, "place in the world (1 = perfect)", 36, 600, "#333")
vals = [("as given", [i["x_probe_r2"]["inject"] for a in ("named", "nameless") for _, i in runs(M8, a)], "#55606e"),
        ("inside", [i["x_probe_r2"]["18"] for a in ("named", "nameless") for _, i in runs(M8, a)], NAVY)]
for j, (lb, v, col) in enumerate(vals):
    m_, s_ = agg(v); yy = y + 290 + j * 95
    T(bx, yy + 52, lb, 38, 700, col); rect(bx + 170, yy + 10, 560 * m_, 60, col, 8); T(bx + 170 + 560 * m_ + 15, yy + 55, f"{m_:.2f}", 42, 800, col)

svg = f'''<html><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}">
<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="6" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker></defs>
<rect width="{W}" height="{H}" fill="#fff"/>{"".join(out)}</svg></body></html>'''
open("$BODY_WORK/fig/body_instill_v3.html", "w").write(svg)
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
subprocess.run([CH, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                "--screenshot=$BODY_WORK/fig/body_instill_v3.png",
                "file://$BODY_WORK/fig/body_instill_v3.html"], capture_output=True, timeout=60)
print("ok", {k: len(v) for k, v in R.items()})
