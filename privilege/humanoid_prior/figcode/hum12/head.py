"""Humanoid privilege v7. Hand-built SVG, rendered with Chrome for Testing. Every number read from the run files."""
import json, math, os, base64
CF = "$BODY_WORK/"
OUT = CF + "fig/hum11/"; H7 = CF + "fig/hum7/"
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
