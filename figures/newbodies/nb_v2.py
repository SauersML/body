"""New bodies v2 figure: SVG built by hand, rendered with Chrome for Testing. Data: fig/nb_v2_data.json (from body/val/ana)."""
import os
import json, math, os, sys, subprocess
CF = os.path.expandvars("$BODY_WORK/")
D = json.load(open(CF + "fig/nb_v2_data.json"))
W, HGT = 2600, int(os.environ.get("FIGH", 2560))
INK = "#111418"; ACC = "#E0552B"; HUM = "#2E6DB4"; BODY = "#E4DED4"; CARD = "#F1F3F6"; FILL = "#EFE9DF"; LINE = "#9AA3AE"
S = []
def add(s): S.append(s)
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, spans, size=50, anchor="start", weight=400, fam="Inter"):
    """spans: str or list of (text, color, weight)"""
    if isinstance(spans, str): spans = [(spans, INK, weight)]
    inner = "".join(f'<tspan fill="{c}" font-weight="{w}">{esc(t)}</tspan>' for t, c, w in spans)
    add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" text-anchor="{anchor}" dominant-baseline="central">{inner}</text>')
def rect(x, y, w, h, fill=CARD, r=24, stroke="none", sw=0, extra=""):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
def line(pts, c=INK, w=6, cap="round", extra=""):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<path d="{d}" stroke="{c}" stroke-width="{w}" fill="none" stroke-linecap="{cap}" stroke-linejoin="round" {extra}/>')
def circ(x, y, r, fill=INK, stroke="none", sw=0, extra=""):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
def arrow(x0, y0, x1, y1, c=INK, w=7, bend=0):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
    cx, cy = mx - dy / L * bend, my + dx / L * bend
    ang = math.atan2(y1 - cy, x1 - cx); hx, hy = x1 - 26 * math.cos(ang), y1 - 26 * math.sin(ang)
    add(f'<path d="M{x0},{y0} Q{cx:.1f},{cy:.1f} {hx:.1f},{hy:.1f}" stroke="{c}" stroke-width="{w}" fill="none" stroke-linecap="round"/>')
    p = [(x1, y1), (x1 - 34 * math.cos(ang - .45), y1 - 34 * math.sin(ang - .45)), (x1 - 34 * math.cos(ang + .45), y1 - 34 * math.sin(ang + .45))]
    add(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in p)}" fill="{c}"/>')

# ---------- human silhouette ----------
def human(hx, top, s=1.0, c=BODY, op=1.0):
    P = lambda x, y: (hx + x * s, top + y * s)
    g = [f'<g opacity="{op}">']
    def seg(pts, w):
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in (P(*p) for p in pts))
        g.append(f'<path d="{d}" stroke="{c}" stroke-width="{w * s:.1f}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    seg([(0, 170), (0, 235)], 62); seg([(-150, 272), (150, 272)], 72)
    x, y = P(-122, 250); g.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{244 * s:.1f}" height="{385 * s:.1f}" rx="{62 * s:.1f}" fill="{c}"/>')
    for sg in (-1, 1):
        seg([(sg * 160, 282), (sg * 196, 480), (sg * 214, 660)], 54)
        x, y = P(sg * 222, 722); g.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{32 * s:.1f}" ry="{52 * s:.1f}" fill="{c}"/>')
        seg([(sg * 62, 600), (sg * 72, 880), (sg * 78, 1130)], 76)
        seg([(sg * 78, 1160), (sg * 122, 1172)], 46)
    x, y = P(0, 92); g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{90 * s:.1f}" fill="{c}"/>')
    g.append("</g>"); add("".join(g)); return P
PL = {}
for sg, s in [(1, "L"), (-1, "R")]:
    for k, (x, y) in {"thumb": (196, 742), "index": (226, 790), "middle": (240, 785), "ring": (250, 772), "little": (256, 752), "palm": (222, 728),
                      "backhand": (226, 708), "wrist": (216, 668), "forearm": (205, 572), "elbow": (196, 480), "upperarm": (178, 380), "shoulder": (150, 282),
                      "toes": (128, 1174), "sole": (104, 1176), "heel": (80, 1170), "ankle": (78, 1128), "shin": (75, 1010), "knee": (72, 880),
                      "thigh": (66, 745), "hip": (92, 612)}.items():
        PL[k + s] = (sg * x, y)
PL.update({"forehead": (0, 30), "crown": (0, 6), "cheekL": (48, 118), "cheekR": (-48, 118), "lips": (0, 148), "tongue": (0, 160), "earL": (88, 100),
           "earR": (-88, 100), "neckF": (0, 215), "neckB": (0, 196), "chest": (0, 360), "belly": (0, 520), "upback": (0, 330), "lowback": (0, 565)})
NAME = {"heel": "heel", "ankle": "ankle", "sole": "sole", "toes": "toes", "palm": "palm", "backhand": "knuckles", "thumb": "thumb", "lips": "lips", "ear": "ear",
        "forehead": "forehead", "crown": "scalp", "chest": "chest", "upback": "upper back", "lowback": "lower back", "neckB": "nape", "neckF": "throat",
        "forearm": "forearm", "knee": "knee", "little": "little finger", "ring": "ring finger", "index": "index finger", "middle": "middle finger",
        "cheek": "cheek", "belly": "belly", "shin": "shin", "thigh": "thigh", "wrist": "wrist", "elbow": "elbow", "upperarm": "upper arm", "shoulder": "shoulder",
        "hip": "hip", "tongue": "tongue"}
def nm(k): return NAME[k[:-1] if k[-1] in "LR" and k[:-1] in NAME else k]

# ---------- pictograms ----------
def robot(rx, base, s=1.0, names=None, label=True, size=44):
    """rx = centre x, base = wheel bottom y. returns anchor points"""
    P = lambda x, y: (rx + x * s, base + y * s)
    cb, ct = -120, -430
    x, y = P(-125, ct); rect(x, y, 250 * s, (cb - ct) * s, fill=FILL, r=30 * s, stroke=INK, sw=6 * s)
    line([P(0, ct), P(0, ct - 70)], INK, 12 * s)
    x, y = P(-80, ct - 170); rect(x, y, 160 * s, 100 * s, fill=FILL, r=24 * s, stroke=INK, sw=6 * s)
    cam = P(0, ct - 120); circ(*cam, 30 * s, INK); circ(cam[0] + 9 * s, cam[1] - 9 * s, 8 * s, "#FFFFFF")
    ant = P(-50, ct - 300); line([P(-40, ct - 170), ant], INK, 7 * s); circ(*ant, 13 * s, INK)
    gy = ct + 150; line([P(125, gy), P(200, gy), P(238, gy - 40)], INK, 12 * s); line([P(200, gy), P(238, gy + 40)], INK, 12 * s)
    wl, wr = P(-70, -55), P(70, -55)
    for w in (wl, wr): circ(*w, 55 * s, INK); circ(*w, 20 * s, FILL)
    A = {"wheelL": wl, "wheelR": wr, "gripper": P(232, gy), "camera": cam, "antenna": ant, "chassis": P(0, (ct + cb) / 2)}
    return A
def octopus(ox, base, s=1.0):
    P = lambda x, y: (ox + x * s, base + y * s)
    # arms
    arms = []
    for i in range(8):
        a0 = -90 + (i - 3.5) * 22
        x0 = (i - 3.5) * 26
        pts = []
        for t in range(0, 21):
            u = t / 20; y = -400 + u * 400; x = x0 * (1 + 2.2 * u) + 34 * math.sin(u * 5 + i) * u
            pts.append(P(x, y))
        line(pts, "#1E2430", (40 - 26 * 0) * s * 0.75 * (1 - 0.0), extra=f'opacity="1"'); arms.append(pts)
    x, y = P(0, -560); add(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{120 * s:.1f}" ry="{170 * s:.1f}" fill="{FILL}" stroke="{INK}" stroke-width="{6 * s:.1f}"/>')
    e1, e2 = P(-48, -420), P(48, -420)
    for e in (e1, e2): circ(*e, 22 * s, "#FFFFFF", INK, 5 * s); circ(*e, 9 * s, INK)
    return {"mantle": P(0, -610), "eye": e2, "arm1": arms[1][9], "arm5": arms[5][12], "suckers": arms[2][15], "armtip": arms[7][20]}

def arc(p, q, c, w, bend=0.18, dot=True):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2; dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = mx - dy * bend, my + dx * bend
    add(f'<path d="M{p[0]:.1f},{p[1]:.1f} Q{cx:.1f},{cy:.1f} {q[0]:.1f},{q[1]:.1f}" stroke="{c}" stroke-width="{w:.1f}" fill="none" stroke-linecap="round"/>')
    if dot: circ(*p, 13, c, "#FFFFFF", 4); circ(*q, 15, c, "#FFFFFF", 4)

exec(open(CF + "fig/nb_v2_layout.py").read())

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{HGT}" viewBox="0 0 {W} {HGT}"><rect width="{W}" height="{HGT}" fill="#FFFFFF"/>' + "".join(S) + "</svg>"
FD = CF + "fonts/inter/extras/ttf/"
faces = "".join(f'@font-face{{font-family:"Inter";src:url("file://{FD}Inter-{n}.ttf");font-weight:{w};}}' for n, w in [("Regular", 400), ("Medium", 500), ("SemiBold", 600), ("Bold", 700)])
faces += "".join(f'@font-face{{font-family:"Inter Display";src:url("file://{FD}InterDisplay-{n}.ttf");font-weight:{w};}}' for n, w in [("SemiBold", 600), ("Bold", 700)])
html = f'<html><head><style>{faces} body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>'
open(CF + "fig/nb_v2.html", "w").write(html)
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.expanduser("~/.pw"))
EXE = os.path.expandvars(os.environ.get("CHROME_BIN"))
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, headless=True)
    pg = b.new_context(viewport={"width": W, "height": HGT}, device_scale_factor=1).new_page()
    pg.goto("file://" + CF + "fig/nb_v2.html"); pg.wait_for_timeout(400)
    pg.screenshot(path=CF + "fig/new_bodies_v2.png", full_page=False); b.close()
print("ok")
