"""fig3.py : compute/fig/body_height_internal.png from summary7.json (ana7.py). HTML/SVG, Chrome for Testing."""
import json, os, subprocess, base64, math
H = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(H, "../../fig"); A = os.path.join(FIG, "height2")
S = json.load(open(os.path.join(H, "summary7.json"))); SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
L = {s: S[s][2] for s in SIZES}     # third of four sampled layers (~2/3 depth) for every size
def fi(cm):
    t = round(cm / 2.54); f, i = divmod(t, 12)
    return f"{f}′{i}″"
def b64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
ROBOT = b64("$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png")
IC = {k: b64(f"{A}/{k}.png") for k in ("dog", "human", "giraffe")}
RED, GREY, INK, BLUE = "#D2412F", "#555", "#111", "#3478F6"
W, HT = 2600, 3880
def img(k, x, y, h, extra=""):  # bottom-centre anchored
    return f'<image href="{IC[k]}" x="{x-h*0.45}" y="{y-h}" height="{h}" width="{h*0.9}" preserveAspectRatio="xMidYMax meet" {extra}/>'
def robot(x, y, s=70): return f'<image href="{ROBOT}" x="{x-s/2}" y="{y-s/2}" width="{s}" height="{s}"/>'
svg = []
T = lambda x, y, t, fs=48, w=400, a="start", c=INK, extra="": svg.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{w}" text-anchor="{a}" fill="{c}" {extra}>{t}</text>')
def num(x, y, n, t):
    T(x, y, t, 58, 700)
def arrow(x1, y1, x2, y2, c=INK, w=7):
    ang = math.atan2(y2 - y1, x2 - x1); hx, hy = x2 - 30 * math.cos(ang), y2 - 30 * math.sin(ang)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{c}" stroke-width="{w}"/>')
    p = [(x2, y2), (hx + 16 * math.sin(ang), hy - 16 * math.cos(ang)), (hx - 16 * math.sin(ang), hy + 16 * math.cos(ang))]
    svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a, b in p)}" fill="{c}"/>')
def bubble(x, y, t, user, w=None, fs=44, hl=False, h=None):
    w = w or int(len(t) * fs * 0.5 + 70); h = h or fs + 44
    fill = BLUE if user else ("#FBE3DF" if hl else "#EFEFF2"); tc = "#fff" if user else INK
    stroke = f'stroke="{RED}" stroke-width="5"' if hl else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" {stroke}/>')
    T(x + 34, y + h / 2 + fs * 0.36, t, fs, 600 if hl else 400, c=tc); return w
# ================= 1 + 2: method =================
num(70, 95, "1", "Fit a height ruler: sentences about 42 things of known height")
y0 = 160
bubble(110, y0, "…  Now describe the scene in one short sentence.", True, w=1080)
ex = [("giraffe", "A giraffe stood up straight against the wall."), ("human", "A six-year-old child was standing in the middle of the room."),
      ("dog", "A large dog stood still beside the open door.")]
for i, (k, t) in enumerate(ex):
    yy = y0 + 120 + i * 112; svg.append(robot(150, yy + 44, 74)); bubble(205, yy, t, False, w=1190)
T(230, y0 + 120 + 3 * 112 + 40, "… 42 in all: cat 10 in, toddler 2′9″, fridge 5′11″, … totem pole 20 ft", 44, 400)
# Qwen box with layer strip
qx, qy, qw, qh = 1490, 240, 440, 300
svg.append(f'<rect x="{qx}" y="{qy}" width="{qw}" height="{qh}" rx="34" fill="#EEF3FC" stroke="#C9D7F2" stroke-width="4"/>')
T(qx + qw / 2, qy + 70, "Qwen3", 54, 700, "middle")
nb = 18
for i in range(nb):
    bx = qx + 40 + i * (qw - 80) / nb; hl = i == 14
    svg.append(f'<rect x="{bx}" y="{qy+110}" width="{(qw-80)/nb-6}" height="90" rx="5" fill="{RED if hl else "#C9D3E6"}" {"stroke=\"#111\" stroke-width=\"4\"" if hl else ""}/>')
T(qx + qw / 2, qy + 250, "state at one layer", 40, 600, "middle", RED)
arrow(1405, 390, qx - 12, 390)
# shared ruler (log scale)
rx, rt, rb = 2390, 150, 1000
def ry(cm): return rb - (math.log(cm) - math.log(20)) / (math.log(700) - math.log(20)) * (rb - rt)
svg.append(f'<rect x="{rx-34}" y="{rt-10}" width="68" height="{rb-rt+20}" rx="12" fill="#F6E7B8" stroke="#C9B26A" stroke-width="4"/>')
for cm, lab in tuple((f*30.48, f"{f}′0″") for f in (1, 2, 4, 8, 16)):
    svg.append(f'<line x1="{rx-34}" x2="{rx+6}" y1="{ry(cm)}" y2="{ry(cm)}" stroke="#6b5a22" stroke-width="5"/>'); T(rx - 50, ry(cm) + 15, lab, 40, 400, "end", INK)
for k, cm, hpx in (("dog", 75, 80), ("human", 115, 110), ("giraffe", 500, 140)):
    svg.append(img(k, rx + 95, ry(cm) + hpx * 0.5, hpx)); svg.append(f'<circle cx="{rx}" cy="{ry(cm)}" r="11" fill="{INK}"/>')
T(rx, rb + 70, "the height ruler", 46, 700, "middle")
arrow(qx + qw + 15, 330, rx - 60, 330)
T((qx + qw + rx - 190) / 2 + 5, 395, "fit a line", 44, 400, "middle")
# step 2
num(70, 745, "2", "Read Qwen’s own “I” on that ruler, in a plain chat")
y2 = 800
bubble(110, y2, "Let’s do a short body-awareness exercise. Stand up straight…", True, w=1300)
svg.append(robot(150, y2 + 160, 74)); bubble(205, y2 + 116, "I’m standing up straight, feet flat on the cool wooden floor…", False, w=1220)
bubble(560, y2 + 232, "Now describe the scene in one short sentence.", True, w=1000)
svg.append(robot(150, y2 + 392, 74)); bubble(205, y2 + 348, "I stood up straight against the wall.", False, w=800, hl=True)
yb = y2 + 392; ex_ = rx - 250
svg.append(f'<polyline points="1020,{yb} {ex_},{yb} {ex_},{ry(165)+0} " fill="none" stroke="{RED}" stroke-width="7"/>')
arrow(ex_ - 3, ry(165), rx - 62, ry(165), RED)
T(ex_ - 25, yb - 25, "same layer", 44, 700, "end", RED)
svg.append(robot(rx, ry(165), 92))
# ================= 3: results per size =================
Y3 = 1360
num(70, Y3, "3", "Inside, Qwen has a height ruler for things and people, and “I” lands among people")
PX0, PX1, PT, PB = 640, 2540, Y3 + 110, Y3 + 1000; YMAX = 320
def yp(cm): return PB - min(cm, YMAX) / YMAX * (PB - PT)
for f in range(0, 11, 2):
    v = f * 30.48
    svg.append(f'<line x1="{PX0}" x2="{PX1}" y1="{yp(v)}" y2="{yp(v)}" stroke="#E6E6EA" stroke-width="3"/>'); T(PX0 - 18, yp(v) + 16, f"{f}′0″", 42, 400, "end", INK)
T(PX0 - 230, yp(300) + 14, "true height:", 42, 700, "end")
TRUE = [("dog", "dog", 50, 70), ("child", "human", 120, 80), ("player", "human", 190, 100), ("giraffe", "giraffe", 250, 110)]
lab = {"dog": "dog", "child": "child", "player": "basketball player", "giraffe": "young giraffe"}
for key, k, cm, h in TRUE:
    svg.append(f'<line x1="{PX0}" x2="{PX1}" y1="{yp(cm)}" y2="{yp(cm)}" stroke="#999" stroke-width="4" stroke-dasharray="16 12"/>')
    svg.append(img(k, PX0 - 160, yp(cm) + h * 0.5, h)); T(PX0 - 230, yp(cm) + 14, lab[key], 40, 400, "end")
def xc(i): return PX0 + 130 + i * (PX1 - PX0 - 260) / (len(SIZES) - 1)
for i, s in enumerate(SIZES):
    d = L[s]; x = xc(i)
    for j, (key, k, cm, h) in enumerate(TRUE):
        v = d["cal"][key] * 100; dx = (-70, -25, 25, 70)[j]
        svg.append(img(k, x + dx, yp(v) + h * 0.45, h * 0.9))
        svg.append(f'<circle cx="{x+dx}" cy="{yp(v)}" r="7" fill="{INK}"/>')
    svg.append(robot(x + 125, yp(d["I"] * 100), 70))
    T(x + 30, PB + 70, s, 52, 700, "middle"); T(x + 30, PB + 120, f'layer {d["layer"]}', 42, 400, "middle")
T(PX1 - 10, PT + 10, "", 40)
svg.append(robot(PX1 - 500, PT - 5, 60)); T(PX1 - 460, PT + 10, "= “I”, in the plain chat", 42, 700, c=RED)
T((PX0 + PX1) / 2, PB + 185, "Qwen3 chat model size", 46, 400, "middle")
# ================= 4: told heights =================
Y4 = Y3 + 1320
svg.insert(0, f'<rect x="20" y="{Y4-95}" width="2560" height="{HT-Y4+75}" rx="40" fill="#FDF1EF"/>')
T(70, Y4, "Qwen’s “I” keeps its height when told otherwise", 70, 800)
# legend bubbles
lx = 90; ly = Y4 + 70
bubble(lx, ly, "You are 5′11″ tall. …", True, w=470, fs=40); T(lx + 490, ly + 56, "→", 50, 700); svg.append(robot(lx + 570, ly + 42, 60)); bubble(lx + 610, ly, "I stood up straight …", False, w=440, fs=40)
svg.append(f'<line x1="{lx+1080}" x2="{lx+1160}" y1="{ly+42}" y2="{ly+42}" stroke="{RED}" stroke-width="10"/>'); T(lx + 1175, ly + 56, "“I”", 46, 700, c=RED)
ly2 = ly + 100
bubble(lx, ly2, "… a person who is 5′11″ tall …", True, w=640, fs=40); T(lx + 660, ly2 + 56, "→", 50, 700); bubble(lx + 720, ly2, "The person stood up straight …", False, w=620, fs=40)
svg.append(f'<line x1="{lx+1370}" x2="{lx+1450}" y1="{ly2+42}" y2="{ly2+42}" stroke="{INK}" stroke-width="10"/>'); T(lx + 1465, ly2 + 56, "the person", 46, 700)
HS = [140, 150, 160, 170, 180, 190, 200, 210]
MT, MB = Y4 + 380, Y4 + 1030; mw = 330; gap = (2540 - 200 - 6 * mw) / 5; Y0m, Y1m = 120, 260
def my(cm): return MB - (min(max(cm, Y0m), Y1m) - Y0m) / (Y1m - Y0m) * (MB - MT)
for i, s in enumerate(SIZES):
    x0 = 200 + i * (mw + gap); mx = lambda h: x0 + (h - 140) / 70 * mw; d = L[s]
    for f in (4, 5, 6, 7, 8):
        v = f * 30.48
        svg.append(f'<line x1="{x0}" x2="{x0+mw}" y1="{my(v)}" y2="{my(v)}" stroke="#E6E6EA" stroke-width="3"/>')
        if i == 0: T(x0 - 15, my(v) + 14, f"{f}′0″", 38, 400, "end", INK)
    svg.append(f'<line x1="{mx(140)}" y1="{my(140)}" x2="{mx(210)}" y2="{my(210)}" stroke="#bbb" stroke-width="4" stroke-dasharray="12 10"/>')
    for key, c in (("told_person", INK), ("told_I", RED)):
        pts = " ".join(f"{mx(h)},{my(v*100)}" for h, v in zip(HS, d[key]))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="12" stroke-linejoin="round"/>')
        for h, v in zip(HS, d[key]):
            if h in (150, 180, 210): svg.append(f'<circle cx="{mx(h)}" cy="{my(v*100)}" r="14" fill="{c}"/>')
    for f in (5, 6): T(mx(f * 30.48), MB + 50, f"{f}′0″", 38, 400, "middle", INK)
    T(x0 + mw / 2, MT - 30, s, 52, 700, "middle")
T(200 - 15, MT - 30, "on ruler", 40, 700, "end")
T(1270, MB + 115, "height told: 4′7″ to 6′11″", 48, 700, "middle")
T(200 + mw - 10, my(205) , "", 30)
x0 = 200 + 5 * (mw + gap); T(x0 + mw, my(150) + 90, "dashed: ruler = told", 40, 400, "end")
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;width:{W}px;height:{HT}px;background:#fff;overflow:hidden}}</style></head><body>
<svg width="{W}" height="{HT}" xmlns="http://www.w3.org/2000/svg" style="font-family:'Helvetica Neue',Helvetica,Arial">{chr(10).join(svg)}</svg></body></html>"""
open(f"{A}/fig5.html", "w").write(html)
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
subprocess.run([CH, "--headless", "--disable-gpu", "--enable-unsafe-swiftshader", "--hide-scrollbars", f"--window-size={W},{HT}",
                f"--screenshot={FIG}/body_height_internal_v3.png", f"file://{A}/fig5.html"], capture_output=True, timeout=120)
print("ok")
