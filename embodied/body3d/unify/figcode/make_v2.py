import json, os, base64
from playwright.sync_api import sync_playwright
from PIL import Image
D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open("$BODY_WORK/body3d/unify/r1/stats.json"))
OUT = "$BODY_WORK/fig/body_unify_v2"
W, H = 2060, 1680
INK = "#1D1D1F"; C_TR = "#4B1C7A"; C_HO = "#9A4FD0"
FS = 46; LH = 56
legs = base64.b64encode(open(D + "/legs.png", "rb").read()).decode()
lw0, lh0 = Image.open(D + "/legs.png").size

def txt(x, y, lines, **kw):
    a = " ".join(f'{k.replace("_","-")}="{v}"' for k, v in kw.items())
    return "".join(f'<text x="{float(x):.1f}" y="{float(y) + i*LH:.1f}" font-size="{FS}" {a}>{t}</text>' for i, t in enumerate(lines))

def arrow(x1, y1, x2, y2):
    import math
    ang = math.atan2(y2 - y1, x2 - x1); L = 26
    p1 = (x2 - L*math.cos(ang - 0.45), y2 - L*math.sin(ang - 0.45)); p2 = (x2 - L*math.cos(ang + 0.45), y2 - L*math.sin(ang + 0.45))
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 14*math.cos(ang):.1f}" y2="{y2 - 14*math.sin(ang):.1f}" stroke="{INK}" stroke-width="5"/>'
            f'<path d="M{x2} {y2}L{p1[0]:.1f} {p1[1]:.1f}L{p2[0]:.1f} {p2[1]:.1f}Z" fill="{INK}"/>')

g = []
# ---------- top band: method
TOP = 170
lh = 540; lw = lh * lw0 / lh0; lx = 470; ly = TOP
g.append(f'<image href="data:image/png;base64,{legs}" x="{lx}" y="{ly}" width="{lw:.1f}" height="{lh}"/>')
g.append(txt(40, TOP + 50, ["Training input"], font_weight=700))
g.append(txt(40, TOP + 50 + 1.2*LH, ["In words:"], font_weight=700))
g.append(txt(40, TOP + 50 + 2.2*LH, ["“Someone taps", "your right shin.”"]))
ry = ly + lh*0.776
g.append(txt(40, ry - 30, ["On the skin:"], font_weight=700))
g.append(txt(40, ry - 30 + LH, ["a probe presses", "the right shin"]))
TX = lx + lw + 50
g.append(txt(TX, TOP + 50, ["Test"], font_weight=700))
g.append(txt(TX, TOP + 50 + 1.2*LH, ["build a steering direction from", "sentences only, add the direction", "while Qwen3-8B reaches for", "a target place, see where", "the hand ends"]))
# ---------- chart
arms = [("ctrl", ["no words"]), ("pair", ["words + skin", "at the same time"]),
        ("align", ["words with a", "matching loss"]), ("both", ["both: same time", "+ matching loss"])]
X0, X1, Y0, Y1 = 290, 1690, 770, 1480
vmin, vmax = -2, 30
def yv(v): return Y1 - (v - vmin) / (vmax - vmin) * (Y1 - Y0)
gw = (X1 - X0) / len(arms); bw = gw * 0.34
for i, (a, lab) in enumerate(arms):
    cx = X0 + gw * (i + 0.5)
    for k, (key, col) in enumerate((("trained", C_TR), ("held", C_HO))):
        m, lo, hi = S[a][f"b1_word_{key}_gain_cm"]
        x = cx + (k - 1) * bw + (k * 2 - 1) * 4
        y0, y1 = yv(max(m, 0)), yv(min(m, 0))
        g.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{y1-y0:.1f}" fill="{col}"/>')
        xc = x + bw / 2
        g.append(f'<path d="M{xc:.1f} {yv(lo):.1f}V{yv(hi):.1f}M{xc-18:.1f} {yv(lo):.1f}h36M{xc-18:.1f} {yv(hi):.1f}h36" stroke="{INK}" stroke-width="4" fill="none"/>')
        if a == "both":
            if key == "trained": tr_x = x + bw
            else: ho_x = x + bw
    g.append(txt(f"{cx:.1f}", Y1 + 70, lab, text_anchor="middle"))
g.append(f'<line x1="{X0}" x2="{X1}" y1="{yv(0):.1f}" y2="{yv(0):.1f}" stroke="{INK}" stroke-width="3"/>')
g.append(f'<line x1="{X0}" x2="{X0}" y1="{Y0}" y2="{Y1}" stroke="{INK}" stroke-width="3"/>')
for t in range(0, 31, 10):
    g.append(f'<line x1="{X0-14}" x2="{X0}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" stroke="{INK}" stroke-width="3"/>')
    g.append(f'<text x="{X0-24}" y="{yv(t)+16:.1f}" text-anchor="end" font-size="{FS}">{t}</text>')
ymid = (Y0 + Y1) / 2
g.append(f'<text transform="translate({X0-185},{ymid}) rotate(-90)" text-anchor="middle" font-size="{FS}">'
         f'<tspan x="0" dy="-8">hand ends closer to the target</tspan><tspan x="0" dy="{LH}">than with a random direction (cm)</tspan></text>')
LX = X1 + 20
for key, col, xr, lines in (("trained", C_TR, tr_x, ["places", "described", "in words"]),
                            ("held", C_HO, ho_x, ["places never", "described", "in words"])):
    m = S["both"][f"b1_word_{key}_gain_cm"][0]; yy = yv(m)
    g.append(f'<line x1="{xr+6:.1f}" x2="{LX-8}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="5"/>')
    g.append(txt(LX, f"{yy+16:.1f}", lines[:1], fill=col, font_weight=700))
    g.append(txt(LX, f"{yy+16+LH:.1f}", lines[1:2], fill=col, font_weight=700))
    g.append(txt(LX, f"{yy+16+2*LH:.1f}", lines[2:], fill=col, font_weight=700))

T1 = "Steering with a direction built from words moves Qwen3-8B’s hand"
T2 = "to the target place only after training with a matching loss"
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica Neue, Helvetica, Arial" fill="{INK}">
<rect width="{W}" height="{H}" fill="#fff"/>
<text x="40" y="70" font-size="56" font-weight="700">{T1}</text><text x="40" y="138" font-size="56" font-weight="700">{T2}</text>
{''.join(g)}</svg>'''
open(OUT + ".svg", "w").write(svg)
open(D + "/v2.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')
    pg = b.new_page(viewport={'width': W, 'height': H}, device_scale_factor=2); pg.goto('file://' + D + '/v2.html')
    pg.screenshot(path=D + '/v2_2x.png'); b.close()
im = Image.open(D + '/v2_2x.png').convert('RGB').resize((W, H), Image.LANCZOS)
im.save(OUT + '.png'); im.save(OUT + '.jpg', quality=93)
