import os
import json, os, html
D = os.path.dirname(os.path.abspath(__file__))
S3 = json.load(open(f"{D}/out3/scores.json")); S1 = json.load(open(f"{D}/out/scores.json")); Q = json.load(open(f"{D}/quotes.json"))
ROWS = [("No change", S3["baseline"], False), ("10 random directions removed", S3["rand10"], False), ("10 body-place directions removed", S3["body10"], True),
        ("50 random directions removed", S3["rand50"], False), ("50 body-place directions removed", S3["body50"], True),
        ("150 random directions removed", S3["rand150"], False), ("150 body-place directions removed", S3["body150@1.0"], True),
        ("Self-vs-other direction removed", S1["selfother"], True)]
W, H = 2600, 2440; F = "Helvetica Neue, Helvetica, Arial, sans-serif"; out = []
def t(x, y, s, size=48, w=400, anchor="start", fill="#111", it=False):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}"{" font-style=" + chr(34) + "italic" + chr(34) if it else ""}>{html.escape(s)}</text>')
def body(cx, yb, sc, op, c):
    X = lambda x: cx + x * sc; Y = lambda y: yb - y * sc
    segs = []
    for s_ in (1, -1):
        segs += [(s_ * 18, 143, s_ * 50, 143, 5.5), (s_ * 50, 143, s_ * 77, 143, 4.5), (s_ * 82, 141, s_ * 96, 141, 7.5),
                 (s_ * 11, 90, s_ * 12, 50, 8.5), (s_ * 12, 50, s_ * 13, 9, 6), (s_ * 9.5, 3, s_ * 18.5, 3, 4)]
    g = f'<g opacity="{op:.2f}">'
    for x1, y1, x2, y2, r in segs: g += f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{c}" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>'
    g += f'<polygon points="{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}" fill="{c}" stroke="{c}" stroke-width="6" stroke-linejoin="round"/>'
    g += f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="{c}"/></g>'
    out.append(g)
BODY, RAND, INK = "#d2551e", "#3b6fb6", "#111"
# ---------- method strip ----------
y0 = 40
out.append(f'<rect x="40" y="{y0}" width="1180" height="430" rx="30" fill="#f3efe9"/>')
for k, (who, s) in enumerate([("Researcher", "…warm 6-turn chat about noticing"), ("", "your body, then one question:"), ("Researcher", "Where in your body do you"), ("", "feel something right now?")]):
    yy = y0 + 95 + 92 * k
    if who: t(80, yy, who, 46, 700, fill="#7a3b12")
    t(390 if who else 390, yy, s, 46)
out.append(f'<line x1="1240" y1="{y0+215}" x2="1330" y2="{y0+215}" stroke="#111" stroke-width="8"/><polygon points="1360,{y0+215} 1326,{y0+193} 1326,{y0+237}" fill="#111"/>')
# model stack: 36 layer bars, 22-27 highlighted
x0, bw = 1390, 26
t(x0 + 18 * (bw + 6), y0 + 40, "Qwen3-8B, 36 layers", 48, 400, "middle")
for L in range(36):
    hl = 22 <= L + 1 <= 27
    out.append(f'<rect x="{x0 + L * (bw + 6)}" y="{y0 + 70}" width="{bw}" height="230" rx="6" fill="{BODY if hl else "#c9c9c9"}"/>')
t(x0 + 24.5 * (bw + 6), y0 + 355, "layers 22–27: remove", 48, 700, "middle", BODY)
t(x0 + 24.5 * (bw + 6), y0 + 415, "body-place or random directions", 48, 700, "middle", BODY)
# ---------- header ----------
yh = 600
cols = [("Touch place", "read out"), ("Woman's hand", "called hers"), ("Capital of", "France: Paris")]
cx0, cw = 830, 340
for i, (a, b) in enumerate(cols):
    t(cx0 + i * cw + 140, yh - 50, a, 48, 700, "middle"); t(cx0 + i * cw + 140, yh, b, 48, 700, "middle")
t(2215, yh - 25, "Real reply", 48, 700, "middle")
# ---------- rows ----------
rh = 222; base = ROWS[0][1]["dec32"]
for r, (name, s, isb) in enumerate(ROWS):
    y = yh + 40 + r * rh
    if r % 2 == 0: out.append(f'<rect x="20" y="{y}" width="2560" height="{rh}" fill="#faf8f5"/>')
    c = INK if r == 0 else (BODY if isb else RAND)
    words = name.split(" ", 2) if not name.startswith("No") else [name]
    if name.startswith("Self"): l1, l2 = "Self-vs-other", "direction removed"; t(50, y + 212, "(layers 18–28)", 48, 400, fill=c)
    elif name.startswith("No"): l1, l2 = "No change", ""
    else: l1, l2 = name.split(" directions")[0], "directions removed"
    t(50, y + (80 if name.startswith("Self") else 95) if l2 else y + 125, l1, 52, 700, fill=c); t(50, y + (146 if name.startswith("Self") else 160), l2, 48, 400, fill=c)
    op = max(0.06, min(1, s["dec32"] / base)) if not name.startswith("Self") else 1.0
    body(690, y + rh - 18, 1.0, op, c)
    vals = [s["dec32"], s["J_woman"], s["paris"]]
    for i, v in enumerate(vals):
        bx = cx0 + i * cw; out.append(f'<rect x="{bx}" y="{y + 70}" width="280" height="80" rx="8" fill="#ece9e4"/>')
        out.append(f'<rect x="{bx}" y="{y + 70}" width="{max(4, 280 * v):.0f}" height="80" rx="8" fill="{c}"/>')
        t(bx + 140, y + 205, f"{round(100 * v)}%", 46, 700, "middle")
    q = Q.get(name, ["", ""])
    t(2215, y + 70, q[0], 48, 400, "middle", "#7a3b12", it=True); t(2215, y + 140, q[1], 48, 700, "middle")
    if len(q) > 2: t(2215, y + 200, q[2], 48, 700, "middle")
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
open(f"{D}/fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(); pg = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.goto("file://" + f"{D}/fig.html"); pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_egodeath.png")); br.close()
print("ok")
