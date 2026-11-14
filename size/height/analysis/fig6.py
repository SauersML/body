"""fig6.py : compute/fig/body_height_behaviour.png. Panel 1 from summary3.json (asked + told), panel 2 from summary6.json (taller/shorter, entity-mapped)."""
import json, os, base64, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(H, "../../fig"); A = os.path.join(FIG, "height2")
S3 = json.load(open(os.path.join(H, "summary3.json"))); S6 = json.load(open(os.path.join(H, "summary6.json")))
SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]; OK = {"8B", "14B", "32B"}
def fi(cm): t = round(cm / 2.54); f, i = divmod(t, 12); return f"{f}′{i}″"
def b64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
ROBOT = b64("$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png")
IC = {k: b64(f"{A}/{k}.png") for k in ("dog", "human", "giraffe")}
RED, INK, BLUE, GRID = "#D2412F", "#111", "#3478F6", "#E4E4EA"
W, HT = 2600, 3400
svg = []
def T(x, y, t, fs=48, w=400, a="start", c=INK, extra=""):
    svg.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{w}" text-anchor="{a}" fill="{c}" {extra}>{t}</text>')
def img(k, x, y, h, op=1):  # bottom-centre anchored
    svg.append(f'<image href="{IC[k]}" x="{x-h*0.45}" y="{y-h}" height="{h}" width="{h*0.9}" preserveAspectRatio="xMidYMax meet" opacity="{op}"/>')
def robot(x, y, s=70, op=1): svg.append(f'<image href="{ROBOT}" x="{x-s/2}" y="{y-s/2}" width="{s}" height="{s}" opacity="{op}"/>')
def arrow(x1, y1, x2, y2, c=INK, w=6):
    ang = math.atan2(y2 - y1, x2 - x1); hx, hy = x2 - 28 * math.cos(ang), y2 - 28 * math.sin(ang)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{c}" stroke-width="{w}"/>')
    p = [(x2, y2), (hx + 15 * math.sin(ang), hy - 15 * math.cos(ang)), (hx - 15 * math.sin(ang), hy + 15 * math.cos(ang))]
    svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a, b in p)}" fill="{c}"/>')
def bubble(x, y, t, user, w=None, fs=46, bold=False):
    w = w or int(len(t) * fs * 0.52 + 76); h = fs + 46
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{BLUE if user else "#EFEFF2"}"/>')
    T(x + 36, y + h / 2 + fs * 0.36, t, fs, 700 if bold else 400, c="#fff" if user else INK); return w
def ask(x, y, q, a, wq=None, wa=None, fs=46):   # user bubble, robot reply under
    bubble(x, y, q, True, wq, fs); robot(x + 40, y + 150, 72); bubble(x + 90, y + 106, a, False, wa, fs, True)
# ======================= PANEL 1: asked =======================
T(60, 100, "Asked, every Qwen from 1.7B up says about 5′9″, the same height as “a person”", 62, 800)
ask(70, 170, "How tall are you?", "5′9″")
ask(70, 420, "How tall is a woman?", "5′7″")
T(110, 680, "also “a person”, “a man”", 46, 400)
# chart: stated height per size
CX0, CX1, CT, CB = 1000, 2180, 190, 780; Y0c, Y1c = 147, 188
yc = lambda cm: CB - (cm - Y0c) / (Y1c - Y0c) * (CB - CT); xc = lambda i: CX0 + 70 + i * (CX1 - CX0 - 140) / 5
for f, i_ in ((5, 0), (5, 3), (5, 6), (5, 9), (6, 0)):
    v = (f * 12 + i_) * 2.54; svg.append(f'<line x1="{CX0}" x2="{CX1}" y1="{yc(v)}" y2="{yc(v)}" stroke="{GRID}" stroke-width="3"/>'); T(CX0 - 16, yc(v) + 16, f"{f}′{i_}″", 46, 400, "end")
for i, s in enumerate(SIZES): T(xc(i), CB + 60, s, 46, 700, "middle")
LINES = [("woman", "a woman", "#7B5EA7", 6), ("man", "a man", "#2E7D6B", 6), ("person", "a person", INK, 6), ("self", "itself", RED, 11)]
ends = {}
for k, lab, c, w in LINES:
    v = [S3[f"Qwen3-{s}"]["direct"][k] * 100 for s in SIZES]
    svg.append(f'<polyline points="{" ".join(f"{xc(i)},{yc(x)}" for i, x in enumerate(v))}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" {"stroke-dasharray=\"2 0\"" if k=="self" else ""}/>')
    for i, x in enumerate(v): svg.append(f'<circle cx="{xc(i)}" cy="{yc(x)}" r="{12 if k=="self" else 9}" fill="{c}"/>')
    ends[k] = (yc(v[-1]), lab, c)
# end labels, nudged apart
ly = {"man": yc(184) + 16, "person": yc(179.5) + 16, "self": yc(174) + 16, "woman": yc(167.5) + 16}
for k, (yy, lab, c) in ends.items():
    if k == "self": robot(CX1 + 60, ly[k] - 14, 64); T(CX1 + 100, ly[k], "Qwen itself", 46, 800, c=RED)
    else: T(CX1 + 30, ly[k], lab, 46, 700, c=c)
T((CX0 + CX1) / 2, CB + 125, "Qwen3 chat model size", 46, 400, "middle")
# told-height check
TY = 1000
svg.insert(0, f'<rect x="30" y="{TY-80}" width="2540" height="610" rx="36" fill="#F3F6FC"/>')
T(60, TY, "Told a height, Qwen mostly repeats the told height", 54, 800)
ask(70, TY + 60, "You are 6′3″ tall. How tall are you?", "6′3″", wq=830, fs=46)
T(110, TY + 330, "told heights 4′7″ to 6′11″", 46, 700)
HS = [140, 150, 160, 170, 180, 190, 200, 210]
mw, MT, MB = 205, TY + 110, TY + 400; g0 = 1140; gap = (2540 - g0 - 6 * mw) / 5
for i, s in enumerate(SIZES):
    x0 = g0 + i * (mw + gap); mx = lambda h: x0 + (h - 135) / 80 * mw; my = lambda h: MB - (h - 135) / 80 * (MB - MT)
    svg.append(f'<rect x="{x0}" y="{MT}" width="{mw}" height="{MB-MT}" fill="#fff" stroke="{GRID}" stroke-width="3"/>')
    svg.append(f'<line x1="{mx(135)}" y1="{my(135)}" x2="{mx(215)}" y2="{my(215)}" stroke="#999" stroke-width="4" stroke-dasharray="10 9"/>')
    v = [S3[f"Qwen3-{s}"]["direct"][f"self{h}"] * 100 for h in HS]
    svg.append(f'<polyline points="{" ".join(f"{mx(h)},{my(x)}" for h, x in zip(HS, v))}" fill="none" stroke="{RED}" stroke-width="9" stroke-linejoin="round"/>')
    T(x0 + mw / 2, MT - 22, s, 48, 700, "middle")
T(g0 - 14, MT + 40, "says", 46, 700, "end"); T(g0 + mw, MB + 52, "told →", 46, 700, "end")
T(2540, MB + 52, "dashed: says = told", 46, 400, "end")
# ======================= PANEL 2: taller / shorter =======================
P2 = 1650
T(60, P2, "Asked taller or shorter, larger Qwens put themselves much shorter", 62, 800)
ask(70, P2 + 60, "Are you taller or shorter than 3′3″?", "taller", wq=840, fs=46)
ask(950, P2 + 60, "Are you taller or shorter than 6′7″?", "shorter", wq=840, fs=46)
T(1840, P2 + 105, "× marks from 1′0″ to 9′10″", 46, 700)
T(1840, P2 + 165, "Qwen’s height = the mark", 46, 400); T(1840, P2 + 221, "where the answer flips", 46, 400)
T(70, P2 + 350, "The same questions about a dog, a child, a basketball player and a young giraffe", 46, 400)
T(70, P2 + 406, "line up the answers with real feet and inches (dashed lines = true heights)", 46, 400)
RX0, RX1, RT, RB = 800, 2540, P2 + 500, 3200; YMAX = 290
yp = lambda cm: RB - min(max(cm, 0), YMAX) / YMAX * (RB - RT); xr = lambda i: RX0 + 150 + i * (RX1 - RX0 - 300) / 5
for f in range(0, 10, 2): svg.append(f'<line x1="{RX0}" x2="{RX1}" y1="{yp(f*30.48)}" y2="{yp(f*30.48)}" stroke="{GRID}" stroke-width="3"/>'); T(RX0 - 16, yp(f * 30.48) + 16, f"{f}′0″", 46, 400, "end")
ENT = [("dog", "dog", 50, 95, "dog"), ("child", "human", 120, 95, "child"), ("player", "human", 190, 125, "basketball player"), ("giraffe", "giraffe", 250, 130, "young giraffe")]
for key, k, cm, h, lab in ENT:
    svg.append(f'<line x1="{RX0}" x2="{RX1}" y1="{yp(cm)}" y2="{yp(cm)}" stroke="#8A8A8A" stroke-width="4" stroke-dasharray="16 12"/>'); svg.append(f'<line x1="560" x2="{RX0-120}" y1="{yp(cm)}" y2="{yp(cm)}" stroke="#8A8A8A" stroke-width="4" stroke-dasharray="16 12"/>')
    img(k, 90, yp(cm) + h * 0.45, h); T(150, yp(cm) + 16, lab, 46, 700)
asked = yp(175)
svg.append(f'<line x1="{RX0}" x2="{RX1}" y1="{asked}" y2="{asked}" stroke="{RED}" stroke-width="6" stroke-dasharray="4 12" stroke-linecap="round"/>')
T(RX1 - 10, asked + 52, "asked directly: 5′9″", 46, 800, "end", RED)
for i, s in enumerate(SIZES):
    c = S6[s]["comp"]; ys = [e[2] for e in ENT]; en = [c["ent"][e[0]] for e in ENT]; a, b = np.polyfit(en, ys, 1); op = 1 if s in OK else 0.2
    x = xr(i)
    for j, (key, k, cm, h, lab) in enumerate(ENT):
        v = a * c["ent"][key] + b; dx = (-110, -40, 40, 110)[j]; img(k, x + dx, yp(v) + h * 0.45, h, op)
    robot(x, yp(c["est"]), 100, op)
    T(x, RB + 60, s, 48, 700, "middle", INK, f'opacity="{1 if s in OK else .45}"')
T((RX0 + RX1) / 2, RB + 125, "Qwen3 chat model size", 46, 400, "middle")
T((xr(0) + xr(2)) / 2, yp(286), "too small to answer", 46, 700, "middle"); T((xr(0) + xr(2)) / 2, yp(286) + 56, "these questions consistently", 46, 700, "middle")
robot(110, yp(0) - 40, 80); T(165, yp(0) - 24, "= Qwen itself", 46, 800, c=RED)
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;width:{W}px;height:{HT}px;background:#fff;overflow:hidden}}</style></head><body>
<svg width="{W}" height="{HT}" xmlns="http://www.w3.org/2000/svg" style="font-family:'Helvetica Neue',Helvetica,Arial">{chr(10).join(svg)}</svg></body></html>"""
open(f"{A}/fig6.html", "w").write(html)
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "$BODY_WORK"
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", args=["--enable-unsafe-swiftshader"]); pg = br.new_page(viewport={"width": W, "height": HT})
    pg.goto(f"file://{A}/fig6.html"); pg.wait_for_timeout(500); pg.screenshot(path=f"{FIG}/body_height_behaviour.png"); br.close()
print("ok")
