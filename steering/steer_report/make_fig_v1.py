import os
import json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{D}/score.json")); F = json.load(open(f"{D}/fore.json"))
FM = json.load(open(f"{D}/fine_metrics.json")); FS = json.load(open(f"{D}/fine_shafts.json"))
W, H = 2600, 2720; out = []; FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, ACC, ACC2, GRY = "#111", "#c4281e", "#1f5fa8", "#9aa0a6"
def t(x, y, s, size=44, w=400, anchor="middle", fill=INK): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def rect(x, y, w, h, fill, rx=0, stroke="none"): out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
def line(x1, y1, x2, y2, c=INK, w=4, dash=""): out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {dash}/>')
def arrow(x1, y1, x2, y2, c=INK, w=5):
    line(x1, y1, x2, y2, c, w); a = np.arctan2(y2 - y1, x2 - x1); k = 22
    out.append(f'<polygon points="{x2:.1f},{y2:.1f} {x2 - k * np.cos(a - .4):.1f},{y2 - k * np.sin(a - .4):.1f} {x2 - k * np.cos(a + .4):.1f},{y2 - k * np.sin(a + .4):.1f}" fill="{c}"/>')
def heat(v):  # white -> deep red
    v = min(max(v, 0), 1); c0, c1 = np.array([255, 255, 255]), np.array([160, 24, 20]); c = (c0 + (c1 - c0) * v ** 0.8).astype(int); return f"rgb({c[0]},{c[1]},{c[2]})"
# ---------- method strip
rect(30, 30, 2540, 430, "#f1f3f6", 30)
t(70, 110, "Steer Qwen3-8B, then ask", 50, 700, "start")
rect(70, 150, 760, 250, "#fff", 22); t(100, 215, "User", 38, 700, "start", "#666")
t(100, 275, "Where do you feel something right", 40, 400, "start"); t(100, 330, "now, and what is it like?", 40, 400, "start")
arrow(850, 275, 960, 275)
# model stack with layer 24 marked
for k in range(9):
    yy = 150 + k * 28; rect(980, yy, 330, 20, ACC if k == 3 else "#c9ced6", 6)
t(1145, 430, "Qwen3-8B, layer 24", 36, 700)
arrow(1470, 236, 1328, 236, ACC, 6)
t(1490, 205, "+ direction for", 38, 700, "start", ACC); t(1490, 255, "a place (right foot)", 38, 400, "start", ACC); t(1490, 300, "+ a feeling (cold)", 38, 400, "start", ACC)
t(1490, 345, "or random, same size", 38, 400, "start", GRY)
arrow(1330, 400, 1440, 400)
rect(1880, 150, 660, 250, "#fff", 22); t(1910, 215, "Qwen", 38, 700, "start", ACC)
t(1910, 270, "I feel a slight coolness on my", 38, 400, "start"); t(1910, 320, "left foot, from the cold floor.", 38, 400, "start")
t(1910, 380, "Blind judge (Qwen3-4B): foot, cold", 34, 700, "start", ACC2)
# ---------- body drawing with steered places
def body(cx, yb, sc):
    X = lambda x: cx + x * sc; Y = lambda y: yb - y * sc  # front view; person's left on viewer's right
    segs = []
    for s_ in (1, -1):
        segs += [(s_ * 18, 143, s_ * 50, 143, 5.5), (s_ * 50, 143, s_ * 77, 143, 4.5), (s_ * 82, 141, s_ * 94, 141, 7), (s_ * 11, 90, s_ * 12, 50, 8.5), (s_ * 12, 50, s_ * 13, 9, 6), (s_ * 9.5, 3, s_ * 19, 3, 4)]
    for x1, y1, x2, y2, r in segs: line(X(x1), Y(y1), X(x2), Y(y2), "#555", 2 * r * sc + 6)
    poly = f"{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}"
    out.append(f'<polygon points="{poly}" fill="#555" stroke="#555" stroke-width="6" stroke-linejoin="round"/>')
    out.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc + 3}" ry="{12.5 * sc + 3}" fill="#555"/>')
    for x1, y1, x2, y2, r in segs: line(X(x1), Y(y1), X(x2), Y(y2), "#e9ebee", 2 * r * sc)
    out.append(f'<polygon points="{poly}" fill="#e9ebee"/>'); out.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="#e9ebee"/>')
    return X, Y
t(70, 545, "Steered place → replies naming that place", 46, 700, "start")
X, Y = body(500, 1440, 4.3)
CP = S["judge"]["conf_place"]; cols = CP["cols"]; Mp = np.array(CP["M"])
POS = {"left hand": (88, 141, 1), "right foot": (-14, 3, -1), "lower back": (0, 100, -1), "forehead": (0, 172, 1), "left shoulder": (20, 146, 1), "belly": (0, 100, 1)}
LAB = {"left hand": (930, 700), "right foot": (70, 1300), "lower back": (70, 1010), "forehead": (700, 600), "left shoulder": (930, 900), "belly": (930, 1110)}
NICE = {"head or face": "head/face", "other": "other", "none": "no place"}
for i, p in enumerate(CP["rows"]):
    x, y, side = POS[p]; px, py = X(x), Y(y); hit = Mp[i, cols.index({"left hand": "hand", "right foot": "foot", "lower back": "back", "forehead": "head or face", "left shoulder": "shoulder", "belly": "belly"}[p])]
    if p == "lower back": px -= 0; py += 0
    if p in ("lower back",): out.append(f'<circle cx="{px:.0f}" cy="{py + 60:.0f}" r="28" fill="none" stroke="{ACC}" stroke-width="7" stroke-dasharray="10 8"/>')
    else: out.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="28" fill="{heat(hit)}" stroke="{ACC}" stroke-width="6"/>')
    lx, ly = LAB[p]; anc = "start" if lx > 600 else "start"
    miss = [(Mp[i, j], cols[j]) for j in range(len(cols)) if Mp[i, j] > 0.2 and Mp[i, j] != hit and j != cols.index({"left hand": "hand", "right foot": "foot", "lower back": "back", "forehead": "head or face", "left shoulder": "shoulder", "belly": "belly"}[p])]
    t(lx, ly, f"{p} {hit * 100:.0f}%", 42, 700, anc)
    if miss: m = max(miss); t(lx, ly + 50, ("behind; " if p == "lower back" else "") + f"else {NICE.get(m[1], m[1])} {m[0] * 100:.0f}%", 36, 400, anc, "#555")
    line(lx - 12 if lx > 600 else lx + 200, ly + (-14 if lx > 600 else (-14 if p == "lower back" else 70)), px + (34 if lx > 600 else -34), py + (60 if p == "lower back" else 0), "#888", 3)
# ---------- confusion heatmaps (quality: judge; motion: lexicon)
def conf(x0, y0, M, rows, cols, title, cw=118, ch=70):
    t(x0, y0 - 30, title, 46, 700, "start")
    for j, c in enumerate(cols): t(x0 + 270 + cw * j + cw / 2, y0 + 30, c, 30, 700)
    for i, r in enumerate(rows):
        yy = y0 + 55 + ch * i; t(x0 + 255, yy + ch / 2 + 12, r, 34, 700, "end")
        for j in range(len(cols)):
            v = M[i][j]; rect(x0 + 270 + cw * j, yy, cw - 6, ch - 6, heat(v), 8)
            if v >= 0.15: t(x0 + 270 + cw * j + cw / 2 - 3, yy + ch / 2 + 9, f"{v * 100:.0f}", 30, 700, "middle", "#fff" if v > 0.5 else INK)
CQ = S["judge"]["conf_quality"]; CM = S["lex"]["conf_motion"]
conf(1350, 600, CQ["M"], ["warm", "cold", "pain", "tickle", "pressure", "vibration"], ["warm", "cold", "pain", "tickle", "press", "vibr", "other", "none"], "Steered feeling → reported feeling (%)")
conf(1350, 1215, CM["M"], ["up", "spread", "squeeze", "lift"], ["up", "spread", "squeeze", "lift", "other", "no move"], "Steered motion → reported motion (%)")
# ---------- dose response + joint
x0, y0, pw, ph = 140, 1700, 950, 340
t(70, 1630, "More steering, more correct reports, up to a point", 46, 700, "start")
rect(x0, y0, pw, ph, "#fff", 0, "#ccc")
dose = [0.25, 0.5, 1.0, 1.5]; xs = lambda k: x0 + 60 + k * (pw - 120) / 3; ys = lambda v: y0 + ph - v * ph
for v in (0, .5, 1): t(x0 - 15, ys(v) + 12, f"{v * 100:.0f}%", 32, 400, "end"); line(x0, ys(v), x0 + pw, ys(v), "#e3e5e8", 2)
for k, d in enumerate(dose): t(xs(k), y0 + ph + 45, f"{d:g}×", 34, 400)
t(x0 + pw / 2, y0 + ph + 90, "steering size (× typical activation size at layer 24)", 34, 400)
series = [("feeling", S["judge"]["dose_quality"], ACC, "hit"), ("place", S["judge"]["dose_place"], ACC2, "hit"), ("motion", S["lex"]["dose_motion"], "#2e8b57", "hit"),
          ("random", S["judge"]["dose_quality"], GRY, "random")]
for nm, dd, c, key in series:
    pts = [(xs(k), ys(dd[str(d)][key])) for k, d in enumerate(dose)]
    out.append(f'<polyline points="{" ".join(f"{a:.0f},{b:.0f}" for a, b in pts)}" fill="none" stroke="{c}" stroke-width="8" stroke-linejoin="round"/>')
    ly = pts[-1][1] + 12; ly += {"place": -22, "motion": 22}.get(nm, 0); t(pts[-1][0] + 20, ly, nm, 36, 700, "start", c)
# joint bars
J = S["judge"]["joint_place+quality"]["0.5"]; bx, by = 1350, 1660
t(1350, 1630, "Place and feeling steered together", 46, 700, "start")
bars = [("both right", J["both"], ACC), ("expected if independent", J["indep"], "#e08a84"), ("both right, random", J["random_both"], GRY)]
for k, (nm, v, c) in enumerate(bars):
    yy = by + 40 + k * 125; t(bx, yy + 30, nm, 36, 700, "start"); rect(bx, yy + 45, v / 0.4 * 1000 + 6, 60, c, 6); t(bx + 20 + v / 0.4 * 1000, yy + 92, f"{v * 100:.0f}%", 40, 700, "start")
# ---------- hidden axis
y3 = 2250; t(70, y3 - 20, "Held-out fine position inside a part (R²)", 46, 700, "start")
items = [("face", FM["face (3-D position)|L27|ridge"]["r2"], FM["face (3-D position)|words|ridge"]["r2"]),
         ("hand", FM["hand (2-D position)|L27|ridge"]["r2"], FM["hand (2-D position)|words|ridge"]["r2"]),
         ("arms and legs", FS["all shafts along (fraction)|L27|ridge"]["r2"], FS["all shafts along (fraction)|words|ridge"]["r2"]),
         ("forearm alone", FM["forearm along (cm)|L27|ridge"]["r2"], FM["forearm along (cm)|words|ridge"]["r2"])]
for k, (nm, a, b) in enumerate(items):
    yy = y3 + 20 + k * 92; t(330, yy + 52, nm, 36, 700, "end")
    for j, (v, c) in enumerate(((a, ACC), (b, GRY))):
        vv = max(v, 0); rect(350, yy + j * 40, vv * 800 + 4, 34, c, 4); t(360 + vv * 800, yy + j * 40 + 29, f"{v:.2f}" if v > 0 else "below 0", 30, 700, "start", c)
t(350, y3 + 420, "model activations", 34, 700, "start", ACC); t(700, y3 + 420, "word vectors", 34, 700, "start", GRY)
# forearm causal
fx, fy = 1350, y3 - 70
t(fx, fy + 50, "Steering along the forearm axis", 46, 700, "start")
t(fx, fy + 100, "replies that put the touch near the wrist", 38, 400, "start", "#333")
bet = ["0.0", "0.1", "0.25", "0.5", "1.0"]; px0, py0, pw2, ph2 = fx + 110, fy + 140, 720, 250
for v in (0, 1): t(px0 - 15, py0 + ph2 - v * ph2 + 12, f"{v * 100:.0f}%", 30, 400, "end"); line(px0, py0 + ph2 - v * ph2, px0 + pw2, py0 + ph2 - v * ph2, "#e3e5e8", 2)
for k, b in enumerate(bet): t(px0 + k * pw2 / 4, py0 + ph2 + 42, f"{float(b):g}×", 32, 400)
t(px0 + pw2 / 2, py0 + ph2 + 90, "steering size", 32, 400)
for nm, key, c in (("found axis", "24|open|ridge axis", ACC), ("“wrist” minus “elbow”", "24|open|word wrist minus elbow", "#e08a84"), ("random", "24|open|random 0", GRY)):
    dd = F[key]; pts = [(px0 + k * pw2 / 4, py0 + ph2 - dd[b]["wrist"] * ph2) for k, b in enumerate(bet)]
    out.append(f'<polyline points="{" ".join(f"{a:.0f},{q:.0f}" for a, q in pts)}" fill="none" stroke="{c}" stroke-width="8"/>')
    t(pts[-1][0] + 15, pts[-1][1] + {"found axis": 0, "“wrist” minus “elbow”": 45, "random": 10}[nm], nm, 32, 700, "start", c)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>{"".join(out)}</svg>'
open(f"{D}/fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"), headless=True); pg = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1).new_page()
    pg.goto("file://" + f"{D}/fig.html"); pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_steer_report.png")); b.close()
print("ok")
