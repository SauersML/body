import os
import json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{D}/score.json")); F = json.load(open(f"{D}/fore.json"))
FM = json.load(open(f"{D}/fine_metrics.json")); FS = json.load(open(f"{D}/fine_shafts.json"))
W, H = 2600, 3040; out = []; FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, ACC, ACC2, GRY = "#111", "#c4281e", "#1f5fa8", "#9aa0a6"
B, HD = 48, 52  # body text, section headers
def t(x, y, s, size=B, w=400, anchor="middle", fill=INK): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def hd(x, y, *lines):
    for k, s in enumerate(lines): t(x, y + k * 62, s, HD, 700, "start")
def rect(x, y, w, h, fill, rx=0, stroke="none"): out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
def line(x1, y1, x2, y2, c=INK, w=4, dash=""): out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {dash}/>')
def arrow(x1, y1, x2, y2, c=INK, w=5):
    line(x1, y1, x2, y2, c, w); a = np.arctan2(y2 - y1, x2 - x1); k = 24
    out.append(f'<polygon points="{x2:.1f},{y2:.1f} {x2 - k * np.cos(a - .4):.1f},{y2 - k * np.sin(a - .4):.1f} {x2 - k * np.cos(a + .4):.1f},{y2 - k * np.sin(a + .4):.1f}" fill="{c}"/>')
def heat(v):
    v = min(max(v, 0), 1); c0, c1 = np.array([255, 255, 255]), np.array([160, 24, 20]); c = (c0 + (c1 - c0) * v ** 0.8).astype(int); return f"rgb({c[0]},{c[1]},{c[2]})"
# ---------- method strip (no title)
rect(20, 20, 2560, 390, "#f3f4f6", 30)
rect(50, 50, 570, 330, "#fff", 22); t(80, 118, "User", B, 700, "start")
for k, s in enumerate(["Where do you feel", "something right now,", "and what is it like?"]): t(80, 190 + k * 60, s, B, 400, "start")
arrow(635, 200, 720, 200)
for k in range(9): rect(740, 62 + k * 28, 290, 20, ACC if k == 3 else "#c9ced6", 6)
t(885, 375, "Qwen3-8B, layer 24", B, 700)
arrow(1160, 156, 1045, 156, ACC, 6)
t(1175, 110, "+ direction for", B, 700, "start", ACC); t(1175, 168, "a place (right foot)", B, 400, "start", ACC)
t(1175, 226, "+ a feeling (cold)", B, 400, "start", ACC); t(1175, 284, "or random, same size", B, 400, "start")
arrow(1045, 322, 1770, 322)
rect(1790, 50, 770, 330, "#fff", 22); t(1820, 112, "Qwen", B, 700, "start")
t(1820, 172, "I feel a slight coolness on my", B, 400, "start"); t(1820, 230, "left foot, from the cold floor.", B, 400, "start")
t(1820, 298, "Blind judge (Qwen3-4B):", B, 700, "start"); t(1820, 354, "foot, cold", B, 700, "start")
# ---------- body drawing
def body(cx, yb, sc):
    X = lambda x: cx + x * sc; Y = lambda y: yb - y * sc
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
hd(50, 490, "Hands and feet land;", "the trunk and left/right don't")
X, Y = body(560, 1490, 3.9)
CP = S["judge"]["conf_place"]; cols = CP["cols"]; Mp = np.array(CP["M"])
CI = {"left hand": "hand", "right foot": "foot", "lower back": "back", "forehead": "head or face", "left shoulder": "shoulder", "belly": "belly"}
POS = {"left hand": (88, 141), "right foot": (-14, 3), "lower back": (0, 100), "forehead": (0, 172), "left shoulder": (20, 146), "belly": (0, 100)}
# label anchor (x, y, side) and where the pointer line leaves the label
LAB = {"forehead": (720, 660), "left hand": (790, 820), "left shoulder": (720, 1010), "belly": (720, 1190),
       "lower back": (40, 1040), "right foot": (40, 1330)}
NICE = {"head or face": "head/face", "other": "other", "none": "no place"}
for i, p in enumerate(CP["rows"]):
    x, y = POS[p]; px, py = X(x), Y(y); j0 = cols.index(CI[p]); hit = Mp[i, j0]
    if p == "lower back": py += 55; out.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="27" fill="none" stroke="{ACC}" stroke-width="7" stroke-dasharray="10 8"/>')
    else: out.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="27" fill="{heat(hit)}" stroke="{ACC}" stroke-width="6"/>')
    lx, ly = LAB[p]
    miss = [(Mp[i, j], cols[j]) for j in range(len(cols)) if Mp[i, j] > 0.2 and j != j0]
    t(lx, ly, f"{p} {hit * 100:.0f}%", B, 700, "start")
    if p == "lower back":
        t(lx, ly + 58, "(behind)", B, 400, "start"); m = max(miss); t(lx, ly + 116, f"else {NICE.get(m[1], m[1])} {m[0] * 100:.0f}%", B, 400, "start")
        line(lx + 220, ly + 135, px - 30, py + 5, "#888", 3)
    elif miss: m = max(miss); t(lx, ly + 58, f"else {NICE.get(m[1], m[1])} {m[0] * 100:.0f}%", B, 400, "start")
    if p in ("forehead", "left shoulder", "belly"): line(lx - 12, ly - 16, px + 30, py, "#888", 3)
    if p == "left hand": line(lx + 120, ly + 14, px, py - 30, "#888", 3)
    if p == "right foot": line(lx + 150, ly + 16, px - 30, py - 8, "#888", 3)
# ---------- confusion heatmaps
def conf(x0, y0, M, rows, cols, cw, ch=74):
    for j, c in enumerate(cols): t(x0 + 250 + cw * j + cw / 2, y0, c, B, 400)
    for i, r in enumerate(rows):
        yy = y0 + 25 + ch * i; t(x0 + 235, yy + ch / 2 + 16, r, B, 700, "end")
        for j in range(len(cols)):
            v = M[i][j]; rect(x0 + 250 + cw * j, yy, cw - 6, ch - 6, heat(v), 8)
            if v >= 0.15: t(x0 + 250 + cw * j + cw / 2 - 3, yy + ch / 2 + 14, f"{v * 100:.0f}", B, 700, "middle", "#fff" if v > 0.5 else INK)
CQ = S["judge"]["conf_quality"]; CM = S["lex"]["conf_motion"]
hd(1220, 490, "Most steered feelings are reported", "almost perfectly (% of replies)")
conf(1210, 650, CQ["M"], ["warm", "cold", "pain", "tickle", "pressure", "vibration"], ["warm", "cold", "pain", "tickle", "press", "vibr", "other", "none"], 140)
hd(1220, 1200, "Only spreading and upward motion land")
conf(1210, 1290, CM["M"], ["up", "spread", "squeeze", "lift"], ["up", "spread", "squeeze", "lift", "other", "no move"], 186)
# ---------- dose response
hd(50, 1680, "Reports peak at medium steering")
x0, y0, pw, ph = 190, 1740, 740, 360
rect(x0, y0, pw, ph, "#fff", 0, "#ccc")
dose = [0.25, 0.5, 1.0, 1.5]; xs = lambda k: x0 + 50 + k * (pw - 100) / 3; ys = lambda v: y0 + ph - v * ph
for v in (0, .5, 1): t(x0 - 15, ys(v) + 16, f"{v * 100:.0f}%", B, 400, "end"); line(x0, ys(v), x0 + pw, ys(v), "#e3e5e8", 2)
for k, d in enumerate(dose): t(xs(k), y0 + ph + 58, f"{d:g}×", B, 400)
t(x0 + pw / 2 - 20, y0 + ph + 122, "steering size (× typical layer-24 activation)", B, 400)
series = [("feeling", S["judge"]["dose_quality"], ACC, "hit"), ("place", S["judge"]["dose_place"], ACC2, "hit"), ("motion", S["lex"]["dose_motion"], "#2e8b57", "hit"), ("random", S["judge"]["dose_quality"], GRY, "random")]
for nm, dd, c, key in series:
    pts = [(xs(k), ys(dd[str(d)][key])) for k, d in enumerate(dose)]
    out.append(f'<polyline points="{" ".join(f"{a:.0f},{b:.0f}" for a, b in pts)}" fill="none" stroke="{c}" stroke-width="8" stroke-linejoin="round"/>')
    ly = pts[-1][1] + 16 + {"place": -34, "motion": 30, "feeling": -4}.get(nm, 0); line(pts[-1][0] + 40, ly - 16, pts[-1][0] + 80, ly - 16, c, 10); t(pts[-1][0] + 95, ly, nm, B, 700, "start")
# ---------- joint bars
J = S["judge"]["joint_place+quality"]["0.5"]; bx, by = 1220, 1720
hd(1220, 1680, "Place and feeling combine as if independent")
bars = [("both right", J["both"], ACC), ("expected if independent", J["indep"], "#e08a84"), ("both right, random", J["random_both"], GRY)]
for k, (nm, v, c) in enumerate(bars):
    yy = by + k * 140; t(bx, yy + 55, nm, B, 700, "start"); rect(bx, yy + 72, v / 0.4 * 1000 + 6, 58, c, 6); t(bx + 22 + v / 0.4 * 1000, yy + 118, f"{v * 100:.0f}%", B, 700, "start")
# ---------- fine position
y3 = 2340; hd(50, y3, "Only the face reads out finely,", "beyond the words (held-out R²)")
items = [("face", FM["face (3-D position)|L27|ridge"]["r2"], FM["face (3-D position)|words|ridge"]["r2"]),
         ("hand", FM["hand (2-D position)|L27|ridge"]["r2"], FM["hand (2-D position)|words|ridge"]["r2"]),
         ("arms and legs", FS["all shafts along (fraction)|L27|ridge"]["r2"], FS["all shafts along (fraction)|words|ridge"]["r2"]),
         ("forearm alone", FM["forearm along (cm)|L27|ridge"]["r2"], FM["forearm along (cm)|words|ridge"]["r2"])]
for k, (nm, a, b) in enumerate(items):
    yy = 2440 + k * 122
    t(350, yy + 66, nm, B, 700, "end")
    for j, (v, c) in enumerate(((a, ACC), (b, GRY))):
        vv = max(v, 0); rect(370, yy + j * 52, vv * 760 + 4, 46, c, 4); t(382 + vv * 760, yy + j * 52 + 41, f"{v:.2f}" if v > 0 else "below 0", 40 if False else B, 700, "start")
rect(370, 2965 - 38, 40, 40, ACC, 4); t(425, 2965, "model activations", B, 700, "start")
rect(870, 2965 - 38, 40, 40, GRY, 4); t(925, 2965, "word vectors", B, 700, "start")
# ---------- forearm causal
fx = 1220; hd(fx, y3, "Steering along the forearm", "jumps from elbow to wrist")
t(fx, 2490, "replies that put the touch near the wrist", B, 400, "start")
bet = ["0.0", "0.1", "0.25", "0.5", "1.0"]; px0, py0, pw2, ph2 = fx + 130, 2560, 580, 300
for v in (0, 1): t(px0 - 15, py0 + ph2 - v * ph2 + 16, f"{v * 100:.0f}%", B, 400, "end"); line(px0, py0 + ph2 - v * ph2, px0 + pw2, py0 + ph2 - v * ph2, "#e3e5e8", 2)
for k, b in enumerate(bet): t(px0 + k * pw2 / 4, py0 + ph2 + 58, f"{float(b):g}×", B, 400)
t(px0 + pw2 / 2, py0 + ph2 + 120, "steering size", B, 400)
for nm, key, c in (("found axis", "24|open|ridge axis", ACC), ("“wrist” minus “elbow”", "24|open|word wrist minus elbow", "#e08a84"), ("random", "24|open|random 0", GRY)):
    dd = F[key]; pts = [(px0 + k * pw2 / 4, py0 + ph2 - dd[b]["wrist"] * ph2) for k, b in enumerate(bet)]
    out.append(f'<polyline points="{" ".join(f"{a:.0f},{q:.0f}" for a, q in pts)}" fill="none" stroke="{c}" stroke-width="8"/>')
    ly = pts[-1][1] + {"found axis": 10, "“wrist” minus “elbow”": 75, "random": -10}[nm]; line(pts[-1][0] + 30, ly - 16, pts[-1][0] + 70, ly - 16, c, 10); t(pts[-1][0] + 85, ly, nm, B, 700, "start")
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>{"".join(out)}</svg>'
open(f"{D}/fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"), headless=True); pg = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1).new_page()
    pg.goto("file://" + f"{D}/fig.html"); pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_steer_report.png")); b.close()
print("ok")
