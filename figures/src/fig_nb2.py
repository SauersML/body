import os
import numpy as np, os
F = "Helvetica Neue, Helvetica, Arial, sans-serif"
CH = os.environ.get("CHROME_BIN")
d = np.load(os.path.expandvars("$BODY_REPO/figures/src/nn2.npz"))
P, Nm, ph = d["P"], d["N"], d["ph"]; n = len(P)
out = []
def T(x, y, s, size=52, w=400, anchor="start", fill="#111"):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
# colour scale: deep navy (close) -> yellow (40+ cm)
STOPS = [(0, (8, 29, 88)), (8, (34, 94, 168)), (16, (29, 145, 192)), (24, (65, 182, 150)), (32, (161, 211, 90)), (40, (255, 214, 0))]
def col(v):
    v = min(max(v, 0), 40)
    for (a, ca), (b, cb) in zip(STOPS, STOPS[1:]):
        if v <= b:
            t = (v - a) / (b - a); c = [round(ca[k] + t * (cb[k] - ca[k])) for k in range(3)]; return "#%02x%02x%02x" % tuple(c)
# A-pose: rotate arm points about the shoulder
TH = np.radians(52)
def pose(x, y, arm=True):
    s = 1 if x >= 0 else -1; ax = abs(x)
    if not arm or ax < 19: return x, y
    dx, dy = ax - 19, y - 143
    rx = dx * np.cos(TH) + dy * np.sin(TH); ry = -dx * np.sin(TH) + dy * np.cos(TH)
    return s * (19 + rx), 143 + ry
def body(X, Y, fill="#e9ecef"):
    def cap(x1, y1, x2, y2, r, arm=True):
        (a, b), (c, e) = pose(x1, y1, arm), pose(x2, y2, arm)
        out.append(f'<line x1="{X(a):.1f}" y1="{Y(b):.1f}" x2="{X(c):.1f}" y2="{Y(e):.1f}" stroke="{fill}" stroke-width="{2*r*sc:.1f}" stroke-linecap="round"/>')
    for s in (1, -1):
        cap(s*14, 143, s*19.5, 143, 6); cap(s*19.5, 143, s*50, 143, 5.5); cap(s*50, 143, s*77, 143, 4.5); cap(s*82, 141, s*86, 141, 6.5)
        for y, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s*89, y, s*(90+Ln), y, 1.5)
        cap(s*81, 146, s*86.5, 156.5, 1.7); cap(s*11, 90, s*12, 50, 8, False); cap(s*12, 50, s*13, 9, 5.5, False); cap(s*10.5, 3, s*16.5, 3, 3.2, False)
    out.append(f'<polygon points="{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}" fill="{fill}" stroke="{fill}" stroke-width="{3*sc}" stroke-linejoin="round"/>')
    cap(0, 140, 0, 152, 5, False); out.append(f'<ellipse cx="{X(0):.1f}" cy="{Y(165):.1f}" rx="{10.5*sc:.1f}" ry="{12.5*sc:.1f}" fill="{fill}"/>')
REG = d["reg"]; ISARM = np.array([r[:3] in ("arm", "han") for r in REG])
PA = np.array([pose(x, y, a) for (x, y, z), a in zip(P, ISARM)])
front = Nm[:, 2] >= -0.3
W, H = 2600, 1960
sc = 4.35; rowh = 800
T(40, 78, "Qwen3-32B's most similar spots sit closer on the body than Qwen3-8B's", 62, 700)
drawn = {}
models = [("32B", "Qwen3-32B"), ("8B", "Qwen3-8B")]
for r, (Z, name) in enumerate(models):
    top = 130 + r * (rowh + 95)
    T(40, top + 60, name, 64, 700)
    yb = top + 80 + rowh - 30
    for v, (view, cx) in enumerate([("front", 380), ("back", 1080)]):
        sg = 1 if view == "front" else -1
        X = lambda x, cx=cx, sg=sg: cx + sg * x * sc; Y = lambda y, yb=yb: yb - y * sc
        body(X, Y)
        if r == 1: T(cx, yb + 75, view, 52, 400, "middle")
        ix = np.where(front if view == "front" else ~front)[0]
        ix = ix[np.argsort(sg * P[ix, 2])]  # nearer the viewer drawn last
        for i in ix:
            c = col(d["d_" + Z][i]); x, y = X(PA[i, 0]), Y(PA[i, 1])
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.5" fill="{c}" stroke="#ffffff" stroke-width="1.6"/>')
            drawn[(Z, i)] = (x, y, c)
# right panel: distributions on a shared cm axis, bars coloured by the same scale
x0, x1, xmax = 1560, 2520, 130
XA = lambda v: x0 + (x1 - x0) * v / xmax
T(x0 - 20, 205, "Distance on the body from each spot", 52, 700)
T(x0 - 20, 270, "to its 5 most similar spots", 52, 700)
T(x0 - 20, 335, "(left and right count as one place)", 48, 400)
rows = [("32B", "Qwen3-32B"), ("8B", "Qwen3-8B"), ("words", "input word vectors"), ("random", "random spots")]
rh = 385; ry0 = 360; bw = 2.5
for k, (Z, name) in enumerate(rows):
    base = ry0 + k * rh + 250
    v = d["d_" + Z]; h, e = np.histogram(np.clip(v, 0, xmax - 0.01), bins=np.arange(0, xmax + bw, bw))
    hs = 245 / 95  # px per spot, shared by all rows
    for c, a in zip(h, e[:-1]):
        if c: out.append(f'<rect x="{XA(a)+0.6:.1f}" y="{base - c*hs:.1f}" width="{XA(a+bw)-XA(a)-1.2:.1f}" height="{c*hs:.1f}" fill="{col(a + bw/2)}"/>')
    out.append(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="#111" stroke-width="2.5"/>')
    med = np.median(v)
    lx = XA(50) if Z != "random" else x0
    T(lx, base - 120, name, 52, 700 if k < 2 else 400, "start")
    T(lx, base - 58, f"median {med:.0f} cm", 52, 400, "start")
# shared axis
ya = ry0 + 3 * rh + 250
for t in range(0, 121, 20):
    out.append(f'<line x1="{XA(t):.1f}" y1="{ya}" x2="{XA(t):.1f}" y2="{ya+14}" stroke="#111" stroke-width="2.5"/>')
    T(XA(t), ya + 64, f"{t}" + (" cm" if t == 120 else ""), 48, 400, "middle")
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
h = "/tmp/nb2.html"; open(h, "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=CH, headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + h); pg.screenshot(path="/tmp/nb2_2x.png"); br.close()
from PIL import Image; Image.open("/tmp/nb2_2x.png").resize((W, H), Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_neighbours_v2.png"))
import pickle; pickle.dump(drawn, open("/tmp/nb2_drawn.pkl", "wb"))
print({Z: round(float(np.median(d["d_" + Z])), 1) for Z, _ in rows})
