"""Humanoid privilege v7. Hand-built SVG, rendered with Chrome for Testing. Every number read from the run files."""
import json, math, os, base64
CF = "$BODY_WORK/"
OUT = CF + "fig/hum12/"; H7 = CF + "fig/hum7/"
VAL = CF + "body/val/out32/"; HUMR = CF + "body/humanoid/res/"
ana = json.load(open(VAL + "ana_L52.json"))["robot"]
null = json.load(open(VAL + "null52.json"))["robot"]["null_pct"]          # median, 95th, 99th
sc = json.load(open(VAL + "steer_counts.json"))
t4 = json.load(open(HUMR + "t4.json"))["res"]
LM = json.load(open(H7 + "lm.json"))

def frac(key, part):
    c = sc["robot|" + key]; return c["first"].get(part, 0) / c["n"]
def rnd(part):
    return sum(sc[f"robot|random {i}"]["first"].get(part, 0) for i in (1, 2, 3)) / sum(sc[f"robot|random {i}"]["n"] for i in (1, 2, 3))
NUM = {}

W, H = 2400, int(os.environ.get("FIGH", 2330))
INK = "#1D1D1F"; BLUE = "#2B5FA8"; ORG = "#E07B24"; ORGD = "#C9600F"
CTRL = "#C9CDD3"; BAND = "#EDEEF1"
S = []
def add(s): S.append(s)
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, spans, size=44, anchor="start", weight=500, halo=False):
    if isinstance(spans, str): spans = [(spans, INK, weight)]
    inner = "".join(f'<tspan fill="{c}" font-weight="{w}">{esc(t)}</tspan>' for t, c, w in spans)
    h = ' paint-order="stroke" stroke="#FFFFFF" stroke-width="12" stroke-linejoin="round"' if halo else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter" font-size="{size}" text-anchor="{anchor}" dominant-baseline="central"{h}>{inner}</text>')
def rect(x, y, w, h, fill, r=0, stroke="none", sw=0, extra=""):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
def line(pts, c=INK, w=6, extra=""):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<path d="{d}" stroke="{c}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round" {extra}/>')
def circ(x, y, r, fill=INK, stroke="none", sw=0):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def img(path, x, y, w, h):
    b = base64.b64encode(open(path, "rb").read()).decode()
    add(f'<image href="data:image/png;base64,{b}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="xMidYMid meet"/>')
def arrow_down(x, y0, y1, c=INK, w=6):
    line([(x, y0), (x, y1 - 18)], c, w); add(f'<polygon points="{x-16},{y1-22} {x+16},{y1-22} {x},{y1}" fill="{c}"/>')

ROBF = "#FBE3CF"
def robot(rx, base, s=1.0, sw=6):
    P = lambda x, y: (rx + x * s, base + y * s)
    cb, ct = -120, -430
    x, y = P(-125, ct); rect(x, y, 250 * s, (cb - ct) * s, ROBF, r=30 * s, stroke=INK, sw=sw * s ** .3)
    line([P(0, ct), P(0, ct - 70)], INK, 12 * s)
    x, y = P(-80, ct - 170); rect(x, y, 160 * s, 100 * s, ROBF, r=24 * s, stroke=INK, sw=sw * s ** .3)
    cam = P(0, ct - 120); circ(*cam, 30 * s, INK); circ(cam[0] + 9 * s, cam[1] - 9 * s, 8 * s, "#FFFFFF")
    ant = P(-50, ct - 300); line([P(-40, ct - 170), ant], INK, 7 * s); circ(*ant, 13 * s, INK)
    gy = ct + 150; line([P(125, gy), P(200, gy), P(238, gy - 40)], INK, 12 * s); line([P(200, gy), P(238, gy + 40)], INK, 12 * s)
    wl, wr = P(70, -55), P(-70, -55)          # robot faces the viewer: its left wheel is on the viewer's right
    for w_ in (wl, wr): circ(*w_, 55 * s, INK); circ(*w_, 20 * s, ROBF)
    return {"wheelL": wl, "wheelR": wr, "gripper": P(232, gy), "camera": cam, "antenna": ant, "chassis": P(0, (ct + cb) / 2 + 20)}

def arc(p, q, w, bend, gid):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2; dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = mx - dy * bend, my + dx * bend
    add(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{p[0]:.0f}" y1="{p[1]:.0f}" x2="{q[0]:.0f}" y2="{q[1]:.0f}"><stop offset="0" stop-color="{ORG}"/><stop offset="1" stop-color="{BLUE}"/></linearGradient>')
    add(f'<path d="M{p[0]:.1f},{p[1]:.1f} Q{cx:.1f},{cy:.1f} {q[0]:.1f},{q[1]:.1f}" stroke="url(#{gid})" stroke-width="{w:.1f}" fill="none" stroke-linecap="round" opacity="0.92"/>')
    circ(*p, 14, ORG, "#FFFFFF", 4); circ(*q, 16, BLUE, "#FFFFFF", 4)

from PIL import ImageFont
_F = {}
def tw(t, size, weight):
    k = (size, weight)
    if k not in _F: _F[k] = ImageFont.truetype(CF + "fonts/inter/extras/ttf/Inter-" + {500: "Medium", 700: "Bold", 800: "ExtraBold"}[weight] + ".ttf", size)
    return _F[k].getlength(t)
def wrap(spans, width, size=42):
    """spans: list of (text, color, weight) or str; returns list of lines (each a span list)."""
    if isinstance(spans, str): spans = [(spans, INK, 500)]
    toks = []
    for t, c, w in spans:
        for i, wd in enumerate(t.split(" ")):
            toks.append((("" if i == 0 else " "), wd, c, w))
    lines, cur, cw = [], [], 0
    for sp, wd, c, w in toks:
        add_w = tw(sp + wd, size, w)
        if cur and sp and cw + add_w > width:
            lines.append(cur); cur, cw = [], 0; sp = ""; add_w = tw(wd, size, w)
        cur.append((sp + wd, c, w)); cw += add_w
    if cur: lines.append(cur)
    return lines
def bubble(x, y, w, paras, fill, color=INK, size=42, lh=54, pad=26, r=34, spans_fn=None):
    lines = []
    for p in paras:
        if p is None: lines.append(None); continue
        if isinstance(p, tuple) and p[0] == "raw":
            q_ = p[1]; lines.append([(q_, color, 500)] if isinstance(q_, str) else q_); continue
        if isinstance(p, str): p = [(p, color, 500)]
        lines += wrap(p, w - 2 * pad - 8, size)
    if w == "fit": w = max(sum(tw(t, size, wt) for t, c, wt in ln) for ln in lines if ln) + 2 * pad + 8; x = x - w
    gap = 16
    h = sum(lh if ln else gap for ln in lines) + 2 * pad
    rect(x, y, w, h, fill, r=r)
    yy = y + pad
    for ln in lines:
        if not ln: yy += gap; continue
        text(x + pad + 4, yy + lh * .5, ln, size); yy += lh
    return y + h
import re
H = int(os.environ.get("FIGH", 2420))
def robot(rx, base, s=1.0, sw=6, sx=1):
    P = lambda x, y: (rx + sx * x * s, base + y * s)
    cb, ct = -120, -430
    x, y = P(-125 * sx, ct); rect(x, y, 250 * s, (cb - ct) * s, ROBF, r=30 * s, stroke=INK, sw=sw * s ** .3)
    line([P(0, ct), P(0, ct - 70)], INK, 12 * s)
    x, y = P(-80 * sx, ct - 170); rect(x, y, 160 * s, 100 * s, ROBF, r=24 * s, stroke=INK, sw=sw * s ** .3)
    cam = P(0, ct - 120); circ(*cam, 30 * s, INK); circ(cam[0] + 9 * s, cam[1] - 9 * s, 8 * s, "#FFFFFF")
    ant = P(-50, ct - 300); line([P(-40, ct - 170), ant], INK, 7 * s); circ(*ant, 13 * s, INK)
    gy = ct + 150; line([P(125, gy), P(200, gy), P(238, gy - 40)], INK, 12 * s); line([P(200, gy), P(238, gy + 40)], INK, 12 * s)
    wl, wr = P(70, -55), P(-70, -55)
    for w_ in (wl, wr): circ(*w_, 55 * s, INK); circ(*w_, 20 * s, ROBF)
    return {"wheelL": wl, "wheelR": wr, "gripper": P(232, gy), "camera": cam, "antenna": ant, "chassis": P(0, (ct + cb) / 2 + 20)}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [k, n, p, max(0, c - h), min(1, c + h)]
def rob(key, part):
    c = sc["robot|" + key]; return wilson(c["first"].get(part, 0), c["n"])
def rnd(part):
    return wilson(sum(sc[f"robot|random {i}"]["first"].get(part, 0) for i in (1, 2, 3)), sum(sc[f"robot|random {i}"]["n"] for i in (1, 2, 3)))
FW = {"paw": r"\bpaws?\b", "frond": r"\bfronds?\b"}
def t4c(host, d, word):
    rs = [r for r in t4 if r["host"] == host and r["dir"] == d and (r["alpha"] == 1.0 or d == "none")]
    return wilson(sum(bool(re.search(FW[word], r["reply"].lower())) for r in rs), len(rs))
def pct(v): return f"{round(v[2] * 100)}%"
def cross(x, y, s, c, w=None):
    w = w or s * .2
    line([(x - s * .45, y - s * .45), (x + s * .45, y + s * .45)], c, w); line([(x - s * .45, y + s * .45), (x + s * .45, y - s * .45)], c, w)
def curve(p, q, bend, c, w, dash=False, head=True, op=1.0):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2; dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = mx - dy * bend, my + dx * bend
    ex = q
    if head:   # shorten for arrowhead
        tx, ty = q[0] - cx, q[1] - cy; L_ = math.hypot(tx, ty); ux, uy = tx / L_, ty / L_
        hl = w * 2.6; ex = (q[0] - ux * hl * .8, q[1] - uy * hl * .8)
    da = f'stroke-dasharray="{w*1.6:.0f} {w*1.1:.0f}"' if dash else ""
    add(f'<path d="M{p[0]:.1f},{p[1]:.1f} Q{cx:.1f},{cy:.1f} {ex[0]:.1f},{ex[1]:.1f}" stroke="{c}" stroke-width="{w:.1f}" fill="none" stroke-linecap="round" opacity="{op}" {da}/>')
    if head:
        px_, py_ = -uy, ux
        a = (q[0], q[1]); b = (q[0] - ux * hl + px_ * hl * .55, q[1] - uy * hl + py_ * hl * .55); d = (q[0] - ux * hl - px_ * hl * .55, q[1] - uy * hl - py_ * hl * .55)
        add(f'<polygon points="{a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f} {d[0]:.1f},{d[1]:.1f}" fill="{c}" opacity="{op}"/>')
    return (0.25 * p[0] + 0.5 * cx + 0.25 * q[0], 0.25 * p[1] + 0.5 * cy + 0.25 * q[1])
def flipimg(path, x, y, w, h):
    b = base64.b64encode(open(path, "rb").read()).decode()
    add(f'<g transform="translate({x + w:.1f},{y:.1f}) scale(-1,1)"><image href="data:image/png;base64,{b}" x="0" y="0" width="{w:.1f}" height="{h:.1f}"/></g>')
RAW = lambda *a: ("raw", list(a) if len(a) > 1 or not isinstance(a[0], str) else a[0])
QWEN = "$BODY_WORK/figures/assets/watching_robot.png"

text(36, 62, "Inside Qwen, different bodies use one map of limb roles", 66, weight=800)
line([(36, 118), (W - 36, 118)], INK, 3)
# =================================================================== LEFT
LX, LW = 36, 1140
text(LX, 172, "Robot parts land on the human part", 54, weight=700)
text(LX, 232, "with the same job", 54, weight=700)
cy0 = 284; cw = 558
yb = bubble(LX, cy0, cw, [RAW("“You are a small robot."), RAW("You roll on two wheels…”"), None,
                          RAW("“A feather brushes slowly"), RAW(("across your ", INK, 500), ("left wheel", ORGD, 700), (".”", INK, 500)),
                          RAW(("× 6 robot parts", INK, 700))], "#FCEBDD", pad=22)
yb2 = bubble(LX + LW - cw, cy0, cw, [RAW("“You’re standing in a"), RAW("quiet, sunny room…”"), None,
                          RAW("“A feather brushes slowly"), RAW(("across your ", INK, 500), ("left heel", BLUE, 700), (".”", INK, 500)),
                          RAW(("× 54 human places", INK, 700))], "#E4ECF8", pad=22)
yb = max(yb, yb2)
text(LX + LW / 2, yb + 48, "Qwen3-32B internal state: closest human place", 46, "middle", 700)
BH = int(os.environ.get("BH", 1000)); bs = BH / 1700; hx = LX + LW - 1100 * bs + 20; hy = yb + 86
img(H7 + "body.png", hx, hy, 1100 * bs, 1700 * bs)
L = {k: (hx + v[0] * bs, hy + v[1] * bs) for k, v in LM.items()}
R = robot(LX + 280, hy + BH * 0.965, 1.02)
PAIRS = [("wheelL", "heelL", 0.06), ("wheelR", "heelR", 0.13), ("gripper", "palmR", 0.04), ("camera", "earL", 0.06), ("antenna", "earR", 0.10), ("chassis", "chest", 0.12)]
for i, (a, b, bend) in enumerate(PAIRS):
    sim = ana[a]["sims"][b]; NUM[f"map {a}->{b}"] = sim
    assert ana[a]["top"][0][0] == b, (a, ana[a]["top"][0])
    arc(R[a], L[b], sim * 62, bend, f"g{i}")
for k in ("wheelL", "wheelR"):
    circ(*R[k], 55 * 1.02, INK); circ(*R[k], 20 * 1.02, ROBF); circ(*R[k], 14, ORG, "#FFFFFF", 4)
text(R["antenna"][0] - 30, R["antenna"][1], [("antenna", ORGD, 700)], 46, "end")
text(R["camera"][0] - 100, R["camera"][1], [("camera", ORGD, 700)], 46, "end")
text(R["chassis"][0], R["chassis"][1] + 70, [("chassis", ORGD, 700)], 46, "middle")
text(R["gripper"][0] - 20, R["gripper"][1] + 80, [("gripper", ORGD, 700)], 46, "middle")
text(R["wheelR"][0] - 70, R["wheelL"][1], [("wheels", ORGD, 700)], 46, "end")
text(L["earL"][0] + 34, L["earL"][1] - 6, [("ears", BLUE, 700)], 46, "start", halo=True)
text(L["chest"][0] + 34, L["chest"][1] + 2, [("chest", BLUE, 700)], 46, "start", halo=True)
text(L["palmR"][0] - 22, L["palmR"][1] - 48, [("palm", BLUE, 700)], 46, "end", halo=True)
text((L["heelL"][0] + L["heelR"][0]) / 2 + 150, L["heelL"][1] + 50, [("heels", BLUE, 700)], 46, "start", halo=True)
# bars
NM = {"wheelL": "left wheel", "wheelR": "right wheel", "gripper": "gripper", "camera": "camera", "antenna": "antenna", "chassis": "chassis",
      "heelL": "left heel", "heelR": "right heel", "palmR": "palm", "earL": "ear", "earR": "ear", "chest": "chest"}
rows = sorted(PAIRS, key=lambda p: -NUM[f"map {p[0]}->{p[1]}"])
NP = json.load(open(VAL + "null52_perpart.json"))
import numpy as np
bx0, bx1, vmin, vmax = LX + 530, LX + LW - 30, 0.0, 0.48
X = lambda v: bx0 + (bx1 - bx0) * (v - vmin) / (vmax - vmin)
ry0 = hy + 1700 * bs + int(os.environ.get("G0", 125)); rh = int(os.environ.get("RH", 72))
VF, VS = "#F7D2B3", ORG
for i, (a, b, _) in enumerate(rows):
    y = ry0 + i * rh + rh / 2; v = NUM[f"map {a}->{b}"]
    nv = np.array(NP["null"][a]); med, p95 = NP["pct"][a]
    NUM[f"shuffle {a} median"], NUM[f"shuffle {a} 95th"] = med, p95
    g = np.linspace(nv.min(), nv.max(), 160); bw = 1.06 * nv.std() * len(nv) ** -0.2
    d = np.exp(-0.5 * ((g[:, None] - nv[None]) / bw) ** 2).sum(1); d = d / d.max() * rh * 0.44
    pts = [(X(x), y - h) for x, h in zip(g, d)] + [(X(x), y + h) for x, h in zip(g[::-1], d[::-1])]
    add('<path d="M' + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f' Z" fill="{VF}" stroke="{VS}" stroke-width="3"/>')
    dm = np.interp(med, g, d); line([(X(med), y - dm), (X(med), y + dm)], INK, 5)
    text(bx0 - 22, y, [(NM[a], ORGD, 700), (" → ", INK, 500), (NM[b], BLUE, 700)], 44, "end")
    circ(X(v), y, 21, ORGD, "#FFFFFF", 5)
y0r = ry0 + rh / 2
text(X(NUM[f"map {rows[0][0]}->{rows[0][1]}"]), ry0 - 34, [("real", ORGD, 700)], 44, "middle")
text(X(NP["pct"][rows[0][0]][0]), ry0 - 96, [("shuffled parts", ORGD, 700)], 44, "middle")
text(X(NP["pct"][rows[0][0]][0]), ry0 - 40, [("median", INK, 700)], 44, "middle"); line([(X(NP["pct"][rows[0][0]][0]), ry0 - 14), (X(NP["pct"][rows[0][0]][0]), ry0 + 4)], INK, 5)
ay = ry0 + rh * 6 + 8
line([(X(vmin), ay), (X(vmax), ay)], INK, 4)
for t in (0, .1, .2, .3, .4):
    line([(X(t), ay), (X(t), ay + 14)], INK, 4); text(X(t), ay + 44, "0" if t == 0 else f".{round(t*10)}", 44, "middle", 700)
lastmed = NP["pct"][rows[-1][0]][0]
text((X(vmin) + X(vmax)) / 2, ay + 104, "similarity of internal states", 44, "middle", 700)
ay += 70
LEFT_END = ay + 60
line([(1206, 150), (1206, H - 30)], INK, 3)
exec(open(OUT + 'right14.py').read())

print("left end", LEFT_END, "right end", RIGHT_END, "H", H)

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(S) + "</svg>"
FD = CF + "fonts/inter/extras/ttf/"
faces = "".join(f'@font-face{{font-family:"Inter";src:url("file://{FD}Inter-{n}.ttf");font-weight:{w};}}' for n, w in [("Regular", 400), ("Medium", 500), ("SemiBold", 600), ("Bold", 700), ("ExtraBold", 800)])
html = f'<html><head><style>{faces} body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>'
open(OUT + "fig.html", "w").write(html); open(OUT + "fig.svg", "w").write(svg)
json.dump(NUM, open(OUT + "numbers.json", "w"), indent=1)
from playwright.sync_api import sync_playwright
EXE = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=EXE, headless=True)
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.goto("file://" + OUT + "fig.html"); pg.wait_for_timeout(600)
    pg.screenshot(path=CF + "fig/humanoid_v15.png", full_page=False)
    pg.pdf(path=OUT + "humanoid_v15.pdf", width=f"{W}px", height=f"{H}px", print_background=True)
    b.close()
from PIL import Image
Image.open(CF + "fig/humanoid_v15.png").convert("RGB").save(CF + "fig/humanoid_v15.jpg", quality=92)
