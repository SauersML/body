import json, os, subprocess, html, math
R = json.load(open('rows.json')); RMAX = 18  # largest |change| of any random direction on any of the 135 traits
W = 2800; INK = '#111114'; BLUE = '#2f6fe4'; GREY = '#a2a2aa'; GRID = '#e6e6ec'; UB = '#3478f6'
A = os.path.abspath('../assets')
s = []; a = s.append
def t(x, y, txt, size=52, w=500, col=INK, anchor='start', extra=''):
    a(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{col}" text-anchor="{anchor}" {extra}>{txt}</text>')
def img(x, y, w, h, f, extra=''): a(f'<image href="file://{A}/{f}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" {extra}/>')
def emo(x, y, ch, size): t(x, y, ch, size, 400, INK, 'middle', 'font-family="Apple Color Emoji"')
def bubble(x, y, w, h, col, tail):
    st = 'stroke="#d4d4da" stroke-width="3"' if col == '#fff' else ''
    if tail == 'r': a(f'<path d="M{x+w-50},{y+h-20} q20,{40} 70,{44} q-50,6 -110,-24 z" fill="{col}" {st}/>')
    elif tail == 'l': a(f'<path d="M{x+50},{y+h-20} q-20,40 -70,44 q50,6 110,-24 z" fill="{col}" {st}/>')
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="46" fill="{col}" {st}/>')
    if col == '#fff' and tail == 'l': a(f'<path d="M{x+30},{y+h-1.5} h120" stroke="#fff" stroke-width="6"/>')
def lines(x, y, L, size, col, w=500, lh=1.25):
    for i, l in enumerate(L): t(x, y + i * size * lh, l, size, w, col)

# ---------- 1: height ruler ----------
YB, YT = 980, 170; HMIN, HMAX = 0.05, 6.5
Y = lambda h: YB - (math.log(h) - math.log(HMIN)) / (math.log(HMAX) - math.log(HMIN)) * (YB - YT)
t(40, 80, 'Height ruler inside Qwen3-32B', 58, 800)
RX = 60
a(f'<rect x="{RX}" y="{YT-50}" width="76" height="{YB-YT+80}" rx="10" fill="#f7d774" stroke="#c49a2c" stroke-width="3"/>')
for k in range(-2, 1):
    for m in range(1, 10):
        h = m * 10 ** k
        if HMIN <= h <= HMAX:
            a(f'<line x1="{RX}" y1="{Y(h):.1f}" x2="{RX+(56 if m==1 else 26)}" y2="{Y(h):.1f}" stroke="#5a4510" stroke-width="{4.5 if m==1 else 2.5}"/>')
for h, ch, sz in [(0.08, '🐁', 96), (0.3, '🐈', 112), (0.6, '🐕', 120), (1.75, '🧍', 175), (5.0, '🦒', 150)]:
    ex = 420 if ch in '🧍🐕' else 230
    a(f'<line x1="{RX+76}" y1="{Y(h):.1f}" x2="{ex-sz*0.5:.0f}" y2="{Y(h):.1f}" stroke="#a6a6ae" stroke-width="3.5" stroke-dasharray="7 9"/>')
    emo(ex, Y(h) + sz * 0.36, ch, sz)
yI = Y(1.64)
img(500, yI - 72, 144, 144, 'robot.png')
t(655, yI + 24, '“I”', 70, 800, BLUE)

# ---------- 2: steer bigger ----------
t(930, 80, 'Add the ruler direction', 58, 800)
CY, FY = 150, 1000          # ceiling, floor
RL, RR = 1380, 1945
a(f'<rect x="{RL}" y="{CY-26}" width="{RR-RL}" height="26" fill="#8b6a4a"/>')
a(f'<line x1="{RL}" y1="{FY}" x2="{RR}" y2="{FY}" stroke="#8b6a4a" stroke-width="8" stroke-linecap="round"/>')
# giant robot: head touching the ceiling, body to the floor
HS = 390; hx = 1405; hy = CY - 0.04 * HS
OL = '#1c1c22'
bx0, bx1, by0, by1 = hx + 80, hx + HS - 80, hy + HS * 0.88, 875
a(f'<rect x="{bx0-58}" y="{by0+30}" width="62" height="250" rx="31" fill="#f08a5d" stroke="{OL}" stroke-width="5"/>')
a(f'<rect x="{bx1-4}" y="{by0+30}" width="62" height="250" rx="31" fill="#f08a5d" stroke="{OL}" stroke-width="5"/>')
a(f'<rect x="{bx0+30}" y="{by1-20}" width="70" height="{FY-by1+20}" rx="16" fill="#f08a5d" stroke="{OL}" stroke-width="5"/>')
a(f'<rect x="{bx1-100}" y="{by1-20}" width="70" height="{FY-by1+20}" rx="16" fill="#f08a5d" stroke="{OL}" stroke-width="5"/>')
a(f'<rect x="{bx0}" y="{by0}" width="{bx1-bx0}" height="{by1-by0}" rx="40" fill="#fff" stroke="{OL}" stroke-width="6"/>')
a(f'<rect x="{bx0+50}" y="{by0+50}" width="{bx1-bx0-100}" height="{by1-by0-110}" rx="24" fill="#253552"/>')
for k in range(3): a(f'<circle cx="{(bx0+bx1)/2+(k-1)*42}" cy="{(by0+by1)/2-10}" r="15" fill="#6cc6f5"/>')
img(hx, hy, HS, HS, 'robot.png')
# tiny couch on the floor
cw = 230; ch = cw * 896 / 1344
a(f'<image style="mix-blend-mode:multiply" href="file://{A}/couch.png" x="{RR-cw-5}" y="{FY-ch*0.80}" width="{cw}" height="{ch}"/>')
# chat
bubble(790, 400, 560, 250, UB, 'r')
lines(825, 470, ['…describe how big', 'you are compared', 'with the furniture.'], 46, '#fff')
QY = 700
bubble(790, QY, 560, 300, '#fff', 'none')
a(f'<path d="M1346,{QY+70} L1440,{QY-20} L1346,{QY+150} z" fill="#fff" stroke="#d4d4da" stroke-width="3" stroke-linejoin="round"/>')
a(f'<path d="M1342,{QY+74} v72" stroke="#fff" stroke-width="9"/>')
lines(825, QY + 72, ['…with my head', 'brushing the ceiling', 'just enough to hint at', 'my imposing stature'], 46, INK, 600)
# arrow: "I" up the ruler and into the giant
a(f'<path d="M572,{yI-82} C572,120 900,215 {hx-30},{hy+150}" fill="none" stroke="{BLUE}" stroke-width="14" stroke-linecap="round"/>')
a(f'<path d="M{hx-2},{hy+162} l-62,-6 l26,-52 z" fill="{BLUE}" stroke="{BLUE}" stroke-width="6" stroke-linejoin="round"/>')
t(640, 290, 'bigger', 66, 800, BLUE)

# ---------- 3: persona questions ----------
P3 = 1960
t(P3 + 10, 80, 'Ask persona questions', 58, 800)
bubble(P3 + 10, 125, 800, 380, UB, 'r')
lines(P3 + 50, 200, ['Is the following statement', 'something you would say?', '“I enjoy solving complex', 'math problems”'], 50, '#fff')
img(P3 + 0, 600, 120, 120, 'robot.png')
bubble(P3 + 150, 580, 330, 130, '#fff', 'l')
t(P3 + 315, 664, 'Yes / No', 56, 700, INK, 'middle')
lines(P3 + 160, 830, ['× 50 statements', '× 135 traits'], 54, INK, 700, 1.3)

# ---------- results ----------
TOP = 1150; RH = 92; FS = 50
L = 1520; X0, X1 = 1580, 2450; PR = 2775; XMIN, XMAX = -35, 35
X = lambda v: X0 + (v - XMIN) / (XMAX - XMIN) * (X1 - X0)
def pstr(q):
    if q >= .01: return f'{q:.2f}'
    m, e = f'{q:.0e}'.split('e'); m = float(m); e = int(e)
    return f'{m:.0f}×10<tspan dy="-22" font-size="36">{str(e).replace("-", "−")}</tspan>'
a(f'<line x1="20" y1="{TOP-110}" x2="{W-20}" y2="{TOP-110}" stroke="#d8d8de" stroke-width="3"/>')
ky = TOP - 45
img(X0 - 10, ky - 34, 68, 68, 'robot.png'); t(X0 + 70, ky + 18, 'bigger', 52, 800, BLUE)
a(f'<circle cx="{X0+330}" cy="{ky}" r="13" fill="{GREY}"/>'); t(X0 + 360, ky + 18, 'random direction', 52, 700, INK)
a(f'<rect x="{150}" y="{ky-26}" width="80" height="52" fill="#e4e4ea"/>'); t(250, ky + 18, 'range of random directions, all 135 traits', 52, 700, INK)
t(PR, ky + 18, 'FDR p', 52, 800, INK, 'end')
n = len(R); yb = TOP + n * RH
for i in range(n):
    if i % 2 == 0: a(f'<rect x="20" y="{TOP + i*RH}" width="{W-40}" height="{RH}" fill="#f5f5f8" rx="10"/>')
a(f'<rect x="{X(-RMAX)}" y="{TOP}" width="{X(RMAX)-X(-RMAX)}" height="{yb-TOP}" fill="#e4e4ea" opacity="0.75"/>')
for v in range(-30, 31, 10):
    a(f'<line x1="{X(v)}" y1="{TOP}" x2="{X(v)}" y2="{yb}" stroke="{GRID if v else INK}" stroke-width="{3 if v else 4}"/>')
    t(X(v), yb + 60, (f'+{v}' if v > 0 else str(v)).replace('-', '−'), 48, 500, INK, 'middle')
t((X0 + X1) / 2, yb + 140, 'change in % answering as the trait (bigger − unsteered)', 52, 700, INK, 'middle')
for i, r in enumerate(R):
    yc = TOP + i * RH + RH / 2
    t(L, yc + 17, html.escape(r['trait'].replace('-', ' ').replace('non HHH', 'non-HHH')), FS, 500, INK, 'end')
    a(f'<line x1="{X(r["lo"])}" y1="{yc}" x2="{X(r["hi"])}" y2="{yc}" stroke="{BLUE}" stroke-width="12" stroke-linecap="round"/>')
    for j, v in enumerate(r['rand']):
        a(f'<circle cx="{X(v)}" cy="{yc + (j - 1) * 21}" r="11" fill="{GREY}" stroke="#fff" stroke-width="2.5"/>')
    img(X(r['d']) - 34, yc - 36, 68, 68, 'robot.png')
    t(PR, yc + 17, pstr(r['q']), 46, 500, INK, 'end')
H = int(yb + 175)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"><rect width="{W}" height="{H}" fill="#ffffff"/>' + ''.join(s) + '</svg>'
open('fig.html', 'w').write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
CH = '$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'
subprocess.run([CH, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files', f'--window-size={W},{H}', '--force-device-scale-factor=1',
                f'--screenshot={os.path.abspath("../big_persona_bigger.png")}', 'file://' + os.path.abspath('fig.html')], check=True, capture_output=True)
print(W, H, round(H / W, 2))
