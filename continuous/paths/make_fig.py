import numpy as np, json, os, sys
from sklearn.decomposition import PCA
OUT, L, MODEL, NUMS = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.load(open(sys.argv[4]))
D = json.load(open(f"{OUT}/items.json")); I = D["items"]; X = np.load(f"{OUT}/reply.npy", mmap_mode="r")[D["keep"].index(L)].astype(np.float32)[:, 1:]
X = X - X.mean(1, keepdims=True)
def f(**kw): return [n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())][0]
fitset = [n for n, it in enumerate(I) if it["kind"] == "body" and it["agent"] == "water"]
pca = PCA(2).fit(X[fitset].reshape(-1, X.shape[2]))
SHOW = [("arm, fingertip to armpit", f(kind="body", path="arm_palm_left", agent="water", dir="fwd"), "#1d4e89"),
        ("arm, armpit to fingertip", f(kind="body", path="arm_palm_left", agent="water", dir="rev"), "#e07a00"),
        ("leg, toe to hip", f(kind="body", path="leg_left", agent="water", dir="fwd"), "#2a9d55"),
        ("face, crown to chin", f(kind="body", path="head", agent="water", dir="fwd"), "#9b2fae"),
        ("table, one end to the other", f(kind="nonbody", path="table", agent="water", dir="fwd"), "#8a8a8a")]
P = {n: pca.transform(X[n]) for _, n, _ in SHOW}
W, H = 2800, 2350; F = "Helvetica Neue, Helvetica, Arial, sans-serif"; out = []
def t(x, y, s, size=46, w=400, anchor="start", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def arrowhead(x1, y1, x2, y2, col, s=34):
    a = np.arctan2(y2 - y1, x2 - x1); p = [(x2, y2), (x2 - s * np.cos(a - .45), y2 - s * np.sin(a - .45)), (x2 - s * np.cos(a + .45), y2 - s * np.sin(a + .45))]
    out.append(f'<polygon points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in p)}" fill="{col}"/>')
t(W / 2, 92, f"A drop rolling along the body: {MODEL} tracks the step count, not the place on the body", 58, 700, "middle")
# method strip
y0 = 150; out.append(f'<rect x="40" y="{y0}" width="1900" height="250" rx="34" fill="#fff" stroke="#9aa4b1" stroke-width="4"/>')
t(80, y0 + 62, "User:", 40, 700, fill="#555")
for k, s in enumerate(["A single drop of cool water starts at the tip of your left middle finger and rolls", "slowly along the palm side of your left hand and arm toward your left armpit.",
                       "&#8230; still rolling &#8230; still rolling &#8230; still rolling &#8230;  (16 identical steps)"]): t(80, y0 + 115 + 50 * k, s, 40, 700 if k == 2 else 400)
out.append(f'<line x1="1950" y1="{y0+125}" x2="2010" y2="{y0+125}" stroke="#333" stroke-width="7"/>'); arrowhead(1990, y0 + 125, 2030, y0 + 125, "#333")
out.append(f'<rect x="2035" y="{y0}" width="725" height="250" rx="20" fill="#fff" stroke="#333" stroke-width="5"/>')
for k, s in enumerate([MODEL, f"layer {L}, state where the reply", "would start, after each step"]): t(2397, y0 + 80 + 62 * k, s, 44, 700 if k == 0 else 400, "middle")
# body drawing
sc = 5.9; cx, yb = 640, 1640
X_ = lambda x: cx + x * sc; Y_ = lambda y: yb - y * sc
def cap(x1, y1, x2, y2, r): out.append(f'<line x1="{X_(x1):.1f}" y1="{Y_(y1):.1f}" x2="{X_(x2):.1f}" y2="{Y_(y2):.1f}" stroke="#e2e2e2" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>')
for s in (1, -1):
    cap(s * 18, 143, s * 50, 143, 5.5); cap(s * 50, 143, s * 77, 143, 4.5); cap(s * 82, 141, s * 86, 141, 6.5)
    for yy, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s * 89, yy, s * (90 + Ln), yy, 1.5)
    cap(s * 81, 146, s * 86.5, 156.5, 1.7); cap(s * 11, 90, s * 12, 50, 8); cap(s * 12, 50, s * 13, 9, 5.5); cap(s * 10.5, 3, s * 16.5, 9, 3.2)
out.append(f'<polygon points="{X_(-20)},{Y_(148)} {X_(20)},{Y_(148)} {X_(15)},{Y_(108)} {X_(19)},{Y_(86)} {X_(-19)},{Y_(86)} {X_(-15)},{Y_(108)}" fill="#e2e2e2"/>')
cap(0, 140, 0, 152, 5); out.append(f'<ellipse cx="{X_(0)}" cy="{Y_(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="#e2e2e2"/>')
def path(pts, col, off=0, dash=""):
    q = [(X_(a), Y_(b) + off) for a, b in pts]
    out.append(f'<polyline points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in q)}" fill="none" stroke="{col}" stroke-width="13" stroke-linecap="round"{dash}/>'); arrowhead(*q[-2], *q[-1], col, 44)
# person's left = viewer's right (front view): x+ is person's left -> draw on right side
path([(99, 141), (80, 141), (50, 141), (24, 139)], "#1d4e89", -14)
path([(24, 147), (50, 147), (80, 147), (99, 147)], "#e07a00", -14)
path([(12, 4), (13, 30), (12, 50), (11, 75), (12, 92)], "#2a9d55")
path([(-3, 177), (-3, 165), (-3, 152)], "#9b2fae")
t(X_(60), Y_(160), "left arm", 40, 400, "middle", "#555")
# table control drawing
tx, ty = 230, 1900; out.append(f'<rect x="{tx}" y="{ty-40}" width="820" height="26" rx="6" fill="#d9c4a5"/>')
for lx in (tx + 30, tx + 770): out.append(f'<rect x="{lx}" y="{ty-14}" width="22" height="80" fill="#d9c4a5"/>')
out.append(f'<line x1="{tx+60}" y1="{ty-70}" x2="{tx+740}" y2="{ty-70}" stroke="#8a8a8a" stroke-width="13" stroke-linecap="round"/>'); arrowhead(tx + 700, ty - 70, tx + 770, ty - 70, "#8a8a8a", 44)
t(tx + 410, ty + 110, "control: table, no body", 42, 700, "middle", "#555")
# trajectory panel
px, py, pw, ph = 1280, 470, 1480, 1260
allp = np.concatenate(list(P.values())); u0, u1 = allp[:, 0].min(), allp[:, 0].max(); v0, v1 = allp[:, 1].min(), allp[:, 1].max()
ux = lambda a: px + 90 + (a - u0) / (u1 - u0) * (pw - 180); vy = lambda b: py + ph - 90 - (b - v0) / (v1 - v0) * (ph - 260)
out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="24" fill="#fff" stroke="#9aa4b1" stroke-width="4"/>')
t(px + 40, py + 72, "Activations after steps 1 to 16 (top 2 principal components)", 46, 700)
for nm, n, col in SHOW:
    q = P[n]; pts = " ".join(f"{ux(a):.1f},{vy(b):.1f}" for a, b in q)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="7" stroke-linejoin="round" opacity="0.85"/>')
    for k, (a, b) in enumerate(q): out.append(f'<circle cx="{ux(a):.1f}" cy="{vy(b):.1f}" r="{7 + 1.3 * k:.1f}" fill="{col}" fill-opacity="{0.25 + 0.75 * k / 15:.2f}" stroke="{col}" stroke-width="2"/>')
a, b = P[SHOW[0][1]][0]; t(ux(a) + 10, vy(b) - 30, "step 1", 42, 700)
a, b = P[SHOW[0][1]][-1]; t(ux(a) + 10, vy(b) - 34, "step 16", 42, 700)
ly = py + 130
for k, (nm, n, col) in enumerate(SHOW):
    yy = ly + 56 * k; out.append(f'<line x1="{px+40}" y1="{yy-14}" x2="{px+120}" y2="{yy-14}" stroke="{col}" stroke-width="12" stroke-linecap="round"/>'); t(px + 140, yy, nm, 40)
# bars
bx, by = 1280, 1790
t(bx, by + 10, "Arm, reverse direction matches forward direction at the", 44, 700)
bars = [("same step number", NUMS["fr_same_step_cos_full"], "#1d4e89"), ("same place on the arm", NUMS["fr_same_place_cos_full"], "#e07a00")]
zx = bx + 620; scl = 700
for k, (nm, v, col) in enumerate(bars):
    yy = by + 50 + 80 * k; t(zx - 70, yy + 46, nm, 42, 400, "end")
    x0 = zx if v >= 0 else zx + v * scl; out.append(f'<rect x="{x0:.0f}" y="{yy}" width="{abs(v) * scl:.0f}" height="60" fill="{col}"/>')
    t((zx + v * scl + 14) if v >= 0 else zx + 14, yy + 46, f"{v:.2f}", 42, 700)
out.append(f'<line x1="{zx}" y1="{by+40}" x2="{zx}" y2="{by+210}" stroke="#333" stroke-width="3"/>')
t(zx + scl / 2, by + 260, "cosine similarity of activations", 38, 400, "middle", "#333")
t(bx, by + 350, f"A step-count readout trained on the water drop on the arm also reads", 42, 700)
t(bx, by + 405, f"ant, feather, leg and face paths (r = {NUMS['probe_paths']:.2f}), but not touches named", 42, 400)
t(bx, by + 460, f"at landmarks from fingertip to shoulder (r = {NUMS['probe_spots']:.2f})", 42, 400)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
open("fig.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.screenshot(path="/tmp/fc.png"); br.close()
from PIL import Image; Image.open("/tmp/fc.png").resize((W, H), Image.LANCZOS).save(os.path.expanduser(sys.argv[5]))
