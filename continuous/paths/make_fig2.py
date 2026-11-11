import numpy as np, json, os, sys
from sklearn.decomposition import PCA
OUT, OUTHF, L, MODEL, NUMS, DEST = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], json.load(open(sys.argv[5])), sys.argv[6]
def load(o):
    D = json.load(open(f"{o}/items.json")); X = np.load(f"{o}/reply.npy", mmap_mode="r")[D["keep"].index(L)].astype(np.float32)[:, 1:]; return D["items"], X - X.mean(1, keepdims=True)
I0, X0 = load(OUT); I1, X1 = load(OUTHF)
def g(I, X, **kw): return X[[n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())][0]]
fit = np.concatenate([X0[[n for n, it in enumerate(I0) if it["kind"] == "body" and it["agent"] == "water"]].reshape(-1, X0.shape[2]), X1[[n for n, it in enumerate(I1) if it["kind"] == "body" and it["agent"] == "water"]].reshape(-1, X1.shape[2])])
pca = PCA(2).fit(fit)
C = dict(armf="#1d4e89", armr="#e07a00", leg="#2a9d55", face="#9b2fae", hand="#0f9ea8", cheek="#d1495b", back="#6a4c93", palm="#c9a227", tab="#8a8a8a")
SHOW = [("arm, fingertip to armpit", g(I0, X0, kind="body", path="arm_palm_left", agent="water", dir="fwd"), C["armf"]),
        ("arm, armpit to fingertip", g(I0, X0, kind="body", path="arm_palm_left", agent="water", dir="rev"), C["armr"]),
        ("leg, toe to hip", g(I0, X0, kind="body", path="leg_left", agent="water", dir="fwd"), C["leg"]),
        ("palm to index fingertip", g(I1, X1, kind="body", path="palm_to_index_left", agent="water", dir="fwd"), C["hand"]),
        ("chin to forehead", g(I1, X1, kind="body", path="face_up", agent="water", dir="fwd"), C["face"]),
        ("table", g(I0, X0, kind="nonbody", path="table", agent="water", dir="fwd"), C["tab"])]
P = [pca.transform(x) for _, x, _ in SHOW]
W, H = 2800, 2420; F = "Helvetica Neue, Helvetica, Arial, sans-serif"; out = []
def t(x, y, s, size=46, w=400, anchor="start", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def ah(x1, y1, x2, y2, col, s=34):
    a = np.arctan2(y2 - y1, x2 - x1); p = [(x2, y2), (x2 - s * np.cos(a - .45), y2 - s * np.sin(a - .45)), (x2 - s * np.cos(a + .45), y2 - s * np.sin(a + .45))]
    out.append(f'<polygon points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in p)}" fill="{col}"/>')
def pl(q, col, wd=13, dash="", head=True):
    out.append(f'<polyline points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in q)}" fill="none" stroke="{col}" stroke-width="{wd}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    if head: ah(*q[-2], *q[-1], col, 3.2 * wd)
GR = "#e2e2e2"
y0 = 30; out.append(f'<rect x="40" y="{y0}" width="2040" height="300" rx="34" fill="#fff" stroke="#9aa4b1" stroke-width="4"/>')
for k, s in enumerate(["A single drop of cool water starts at the tip of your left middle", "finger and rolls slowly along the palm side of your left hand and", "arm toward your left armpit.",
                       "&#8230; still rolling &#8230; still rolling &#8230;  (16 steps)"]): t(80, y0 + 76 + 66 * k, s, 54, 700 if k == 3 else 400)
out.append(f'<line x1="2090" y1="{y0+150}" x2="2140" y2="{y0+150}" stroke="#333" stroke-width="7"/>'); ah(2120, y0 + 150, 2160, y0 + 150, "#333")
out.append(f'<rect x="2165" y="{y0}" width="595" height="300" rx="20" fill="#fff" stroke="#333" stroke-width="5"/>')
for k, s in enumerate([MODEL, f"layer {L}", "after each step"]): t(2462, y0 + 95 + 75 * k, s, 52 if k == 0 else 48, 700 if k == 0 else 400, "middle")
# body
sc = 5.2; cx, yb = 640, 1300; X_ = lambda x: cx + x * sc; Y_ = lambda y: yb - y * sc
def cap(x1, y1, x2, y2, r): out.append(f'<line x1="{X_(x1):.1f}" y1="{Y_(y1):.1f}" x2="{X_(x2):.1f}" y2="{Y_(y2):.1f}" stroke="{GR}" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>')
for s in (1, -1):
    cap(s * 18, 143, s * 50, 143, 5.5); cap(s * 50, 143, s * 77, 143, 4.5); cap(s * 82, 141, s * 86, 141, 6.5)
    for yy, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s * 89, yy, s * (90 + Ln), yy, 1.5)
    cap(s * 81, 146, s * 86.5, 156.5, 1.7); cap(s * 11, 90, s * 12, 50, 8); cap(s * 12, 50, s * 13, 9, 5.5); cap(s * 10.5, 3, s * 16.5, 9, 3.2)
out.append(f'<polygon points="{X_(-20)},{Y_(148)} {X_(20)},{Y_(148)} {X_(15)},{Y_(108)} {X_(19)},{Y_(86)} {X_(-19)},{Y_(86)} {X_(-15)},{Y_(108)}" fill="{GR}"/>')
cap(0, 140, 0, 152, 5); out.append(f'<ellipse cx="{X_(0)}" cy="{Y_(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="{GR}"/>')
B = lambda pts, off=0: [(X_(a), Y_(b) + off) for a, b in pts]
pl(B([(99, 141), (80, 141), (50, 141), (24, 139)], -12), C["armf"]); pl(B([(24, 147), (50, 147), (80, 147), (99, 147)], -12), C["armr"])
pl(B([(12, 4), (13, 30), (12, 50), (11, 75), (12, 92)]), C["leg"]); pl(B([(0, 152), (0, 165), (0, 176)]), C["face"])
# hand panel (left hand, palm view: thumb on viewer's right; and back view)
def hand(hx, hy, palm, s=1.0):
    m = 1 if palm else -1
    out.append(f'<rect x="{hx-90}" y="{hy-110}" width="180" height="200" rx="60" fill="{GR}"/>'); out.append(f'<rect x="{hx-55}" y="{hy+60}" width="110" height="110" fill="{GR}"/>')
    for k, (dx, ln) in enumerate([(-66, 110), (-22, 150), (22, 160), (66, 140)]):
        dx = dx * m; out.append(f'<line x1="{hx+dx}" y1="{hy-90}" x2="{hx+dx}" y2="{hy-90-ln}" stroke="{GR}" stroke-width="40" stroke-linecap="round"/>')
    out.append(f'<line x1="{hx+m*70}" y1="{hy+40}" x2="{hx+m*150}" y2="{hy-70}" stroke="{GR}" stroke-width="44" stroke-linecap="round"/>')
hx1, hx2, hy = 540, 200, 1680
hand(hx1, hy, True); hand(hx2, hy, False)
pl([(hx1, hy + 70), (hx1 + 10, hy - 30), (hx1 + 60, hy - 110), (hx1 + 66, hy - 225)], C["hand"], 11)   # heel -> index tip (index next to thumb, viewer's right)
pl([(hx1 + 70, hy + 5), (hx1, hy + 5), (hx1 - 80, hy + 5)], C["palm"], 11)
pl([(hx2, hy + 150), (hx2 - 5, hy + 40), (hx2 - 22, hy - 85)], C["back"], 11)
t(hx1, 1960, "palm", 48, 700, "middle"); t(hx2, 1960, "back", 48, 700, "middle")
# face panel
fx, fy = 1010, 1660
out.append(f'<ellipse cx="{fx}" cy="{fy}" rx="150" ry="200" fill="{GR}"/>')
for s in (-1, 1): out.append(f'<ellipse cx="{fx+s*155}" cy="{fy-10}" rx="26" ry="55" fill="{GR}"/>'); out.append(f'<ellipse cx="{fx+s*58}" cy="{fy-50}" rx="26" ry="12" fill="#bdbdbd"/>')
out.append(f'<path d="M{fx-10},{fy-40} L{fx-18},{fy+35} L{fx+12},{fy+35}" fill="none" stroke="#bdbdbd" stroke-width="7"/>'); out.append(f'<path d="M{fx-45},{fy+85} Q{fx},{fy+110} {fx+45},{fy+85}" fill="none" stroke="#bdbdbd" stroke-width="8"/>')
pl([(fx, fy + 185), (fx, fy + 60), (fx, fy - 40), (fx, fy - 150)], C["face"], 11)
pl([(fx + 150, fy + 10), (fx + 90, fy + 15), (fx + 35, fy + 15)], C["cheek"], 11)
t(fx, 1960, "face", 48, 700, "middle")
# table
tx, ty = 230, 2160; out.append(f'<rect x="{tx}" y="{ty-30}" width="820" height="24" rx="6" fill="#d9c4a5"/>')
for lx in (tx + 30, tx + 770): out.append(f'<rect x="{lx}" y="{ty-6}" width="22" height="70" fill="#d9c4a5"/>')
pl([(tx + 60, ty - 62), (tx + 400, ty - 62), (tx + 770, ty - 62)], C["tab"], 13)
t(tx + 410, ty + 140, "control: table, no body", 48, 700, "middle")
# trajectory panel
px, py, pw, ph = 1290, 380, 1470, 1150
allp = np.concatenate(P); u0, u1 = allp[:, 0].min(), allp[:, 0].max(); v0, v1 = allp[:, 1].min(), allp[:, 1].max()
ux = lambda a: px + 90 + (a - u0) / (u1 - u0) * (pw - 180); vy = lambda b: py + ph - 70 - (b - v0) / (v1 - v0) * (ph - 460)
out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="24" fill="#fff" stroke="#9aa4b1" stroke-width="4"/>')
t(px + 40, py + 72, "Activations, steps 1 to 16 (top 2 PCs)", 52, 700)
for (nm, _, col), q in zip(SHOW, P):
    out.append(f'<polyline points="{" ".join(f"{ux(a):.1f},{vy(b):.1f}" for a, b in q)}" fill="none" stroke="{col}" stroke-width="6" stroke-linejoin="round" opacity="0.85"/>')
    for k, (a, b) in enumerate(q): out.append(f'<circle cx="{ux(a):.1f}" cy="{vy(b):.1f}" r="{6 + 1.2 * k:.1f}" fill="{col}" fill-opacity="{0.2 + 0.8 * k / 15:.2f}" stroke="{col}" stroke-width="2"/>')
q = np.array([p[0] for p in P]); i = q[:, 1].argmin(); t(ux(q[i, 0]) + 40, vy(q[i, 1]) + 16, "step 1", 48, 700, "start")
q = np.concatenate([p[-4:] for p in P]); t(ux(q[:, 0].max()) + 10, vy(q[:, 1].max()) - 55, "step 16", 48, 700, "end")
for k, (nm, _, col) in enumerate(SHOW):
    xx = px + 40 + (k % 2) * 700; yy = py + 150 + 66 * (k // 2)
    out.append(f'<line x1="{xx}" y1="{yy-14}" x2="{xx+70}" y2="{yy-14}" stroke="{col}" stroke-width="12" stroke-linecap="round"/>'); t(xx + 90, yy + 2, nm, 48)
# bars: reverse vs forward, per region
bx, by = 1290, 1620
t(bx, by, "Reverse path matches forward path at same:", 52, 700)
grp = [("arm, leg, back", "body"), ("hand", "hand"), ("face", "face")]
zx = bx + 620; scl = 700
t(zx + 0.5 * scl, by + 70, "step number", 48, 700, "middle", C["armf"]); t(zx - 20, by + 70, "place", 48, 700, "end", C["armr"])
for k, (lab, key) in enumerate(grp):
    yy = by + 100 + 120 * k; t(bx, yy + 66, lab, 48, 400)
    for j, (kk, col) in enumerate([("same_step", C["armf"]), ("same_place", C["armr"])]):
        v = NUMS[key][kk]; x0 = zx if v >= 0 else zx + v * scl
        out.append(f'<rect x="{x0:.0f}" y="{yy + 52 * j}" width="{abs(v) * scl:.0f}" height="50" fill="{col}"/>')
        t((zx + v * scl + 12) if v >= 0 else x0 + 10, yy + 42 + 52 * j, f"{v:.2f}", 48, 700, "start", "#111" if v >= 0 else "#fff")
out.append(f'<line x1="{zx}" y1="{by+90}" x2="{zx}" y2="{by+460}" stroke="#333" stroke-width="3"/>')
t(zx + scl / 2, by + 515, "cosine similarity", 48, 400, "middle")
t(bx, by + 620, "Step-count readout trained on the arm:", 52, 700)
t(bx, by + 688, f"reads every other path (r = {NUMS['probe_paths']:.2f})", 48, 400)
t(bx, by + 756, f"fails on touches named at arm spots (r = {NUMS['probe_spots']:.2f})", 48, 400)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
open("fig.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.screenshot(path="/tmp/fc.png"); br.close()
from PIL import Image; Image.open("/tmp/fc.png").resize((W, H), Image.LANCZOS).save(os.path.expanduser(DEST))
