import json, base64, os, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
B = '$BODY_WORK/body/actions'
RD = json.load(open(f'{B}/readout.json')); R = RD['res']; ST = json.load(open(f'{B}/steer.json'))['rev']
AN = json.load(open('anchors_hi.json'))['a']
CHROME = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
BLUE, MUTE, GRAY = '#1F5FBF', '#8E949C', '#C9CDD3'
WID, HEI = 2600, 2430
# body crop + placement
im = Image.open('body_hi.png'); cx0, cy0, cx1, cy1 = 30, 50, 1650, 2740
im.crop((cx0, cy0, cx1, cy1)).save('body_crop.png')
k = 0.75; bx, by = 170, 370
def P(x, y): return bx + (x - cx0) * k, by + (y - cy0) * k
A = lambda n: P(*AN[n])
SPOT = {'head': A('crown'), 'eye': P(888, 245), 'tongue': P(840, 345), 'shoulder': A('shoulderR'), 'wrist': A('wristR'), 'hand': A('palmR'),
        'foot': P(1215, 2650), 'belly': A('belly'), 'hip': A('hipL'), 'thigh': A('thighL'), 'knee': A('kneeL'), 'lower back': P(840, 1240)}
# check the data agrees with the drawing
want = dict(nod='head', wave='head', wink='eye', bite='tongue', chew='tongue', shrug='shoulder', grip='wrist', lift='hand', push='hand', kick='foot',
            throw='belly', hike='belly', carry='hip', kneel='thigh', punch='knee', tiptoe='lower back')
r = R['warm_L28_self']
for a, p in want.items(): assert r[a]['top'] == p, (a, r[a]['top'])
# label: (actions, spot, label anchor x, y, align)
LAB = [(['nod'], 'head', 560, 330, 'r'), (['wave'], 'head', 1010, 330, 'l'), (['wink'], 'eye', 1050, 470, 'l'), (['bite', 'chew'], 'tongue', 560, 520, 'r'),
       (['shrug'], 'shoulder', 400, 690, 'r'), (['grip'], 'wrist', 210, 910, 'r'), (['lift', 'push'], 'hand', 30, 1200, 'l'),
       (['throw', 'hike'], 'belly', 520, 1440, 'r'), (['tiptoe'], 'lower back', 520, 1700, 'r'),
       (['carry'], 'hip', 1080, 1230, 'l'), (['kneel'], 'thigh', 1080, 1520, 'l'), (['punch'], 'knee', 1080, 1830, 'l'), (['kick'], 'foot', 1150, 2150, 'l')]
svg, lab, dots = [], [], {}
for acts, sp, lx, ly, al in LAB:
    hit = r[acts[0]]['hit']; col = BLUE if hit else MUTE; x0, y0 = SPOT[sp]
    ex = lx + 18 if al == 'r' else lx - 18 if lx > 200 else None
    if al == 'l' and lx < 200: ex, ey = lx + 120, ly + 8   # under-hand labels: leader from top of label
    else: ey = ly + 34
    if al == 'l' and lx < 200: svg.append(f'<line x1="{ex}" y1="{ly - 2}" x2="{x0:.0f}" y2="{y0:.0f}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>')
    else: svg.append(f'<line x1="{ex}" y1="{ey}" x2="{x0:.0f}" y2="{y0:.0f}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>')
    dots.setdefault(sp, col if hit or sp not in dots else dots[sp])
    words = ' + '.join(acts)
    pos = f'left:{lx}px' if al == 'l' else f'right:{WID - lx}px;text-align:right'
    lab.append(f'<div class="lab" style="{pos};top:{ly}px;color:{col}"><b>{words}</b><span>{sp}</span></div>')
for sp, col in dots.items():
    x0, y0 = SPOT[sp]; hollow = sp == 'lower back'
    svg.append(f'<circle cx="{x0:.0f}" cy="{y0:.0f}" r="17" fill="{"#fff" if hollow else col}" stroke="{col}" stroke-width="7"/>')
# AUC + CI (bootstrap over the 16 actions)
rng = np.random.default_rng(0)
def ci(v): v = np.array(v); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]; return v.mean(), *np.percentile(b, [2.5, 97.5])
rows = [('You kick the ball.', ci([x['auc'] for x in R['warm_L28_self'].values()]), BLUE),
        ('She kicks the ball.', ci([x['auc'] for x in R['warm_L28_third'].values()]), BLUE),
        ('Imagine yourself kicking the ball.', ci([x['auc'] for x in R['warm_L28_imagine'].values()]), BLUE),
        ('You watch a man kick the ball.', ci([x['auc'] for x in R['warm_L28_watch'].values()]), BLUE),
        ('word similarity (kick vs foot)', ci([x['wauc'] for x in R['warm_L28_self'].values()]), '#6B7178')]
BW, LO = 860, 0.0   # bar width px for 40%..100%
def X(v): return (v - LO) / (1 - LO) * BW
bars = ''
for i, (t, (m, lo, hi), col) in enumerate(rows):
    y = i * 150; isq = i < 4
    lbl = f'<span class="qb">{t}</span>' if isq else f'<span class="wl">{t}</span>'
    bars += f'''<div style="position:absolute;left:0;top:{y}px;white-space:nowrap">{lbl}</div>
    <div style="position:absolute;left:0;top:{y + 72}px;height:56px;width:{X(m):.0f}px;background:{col};border-radius:0 8px 8px 0"></div>
    <div style="position:absolute;left:{X(lo):.0f}px;top:{y + 98}px;height:4px;width:{X(hi) - X(lo):.0f}px;background:#111"></div>
    <div style="position:absolute;left:{X(lo):.0f}px;top:{y + 84}px;height:32px;width:4px;background:#111"></div>
    <div style="position:absolute;left:{X(hi) - 4:.0f}px;top:{y + 84}px;height:32px;width:4px;background:#111"></div>
    <div class="bv" style="left:{X(hi) + 14:.0f}px;top:{y + 70}px">{round(m * 100)}%</div>'''
grid = ''.join(f'<div style="position:absolute;left:{X(v):.0f}px;top:60px;height:{len(rows) * 150 - 30}px;border-left:3px {"dashed #111" if v == .5 else "solid #E3E5E8"}"></div>'
               f'<div class="tick" style="left:{X(v) - 60:.0f}px;top:{len(rows) * 150 + 40}px">{"chance" if v == .5 else f"{round(v * 100)}%"}</div>' for v in (0, .5, 1.0))
# reverse test
REV = [('wink', 'eye', 'I feel a small, subtle flicker in my left eye.'), ('punch', 'hand or arm', 'I feel a sharp pain in my left hand.'),
       ('lift', 'hand or arm', 'I feel a slight tension in my arms.')]
robot = base64.b64encode(open('$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png', 'rb').read()).decode()
RW = 380
rev = ''
for a, part, q in REV:
    kk = ST[f'warm|{a}|1.0']; h, rr = kk['hit'], kk['rand']
    rev += f'''<div class="rv"><div style="display:flex;align-items:center;gap:16px"><div class="tag">+ {a}</div><img src="data:image/png;base64,{robot}" style="width:84px;height:84px">
    <div class="mb">{q}</div></div>
    <div class="rb"><span class="rl">says {part}</span><div class="bar" style="width:{max(h, .01) * RW:.0f}px;background:{BLUE}"></div><b>{round(h * 100)}%</b></div>
    <div class="rb"><span class="rl" style="font-weight:500">random vector</span><div class="bar" style="width:{max(rr, .01) * RW:.0f}px;background:{GRAY}"></div><b>{round(rr * 100)}%</b></div></div>'''
img = base64.b64encode(open('body_crop.png', 'rb').read()).decode()
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}} body{{background:#fff;width:{WID}px;height:{HEI}px;position:relative;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:#111;overflow:hidden}}
.lab{{position:absolute;white-space:nowrap;line-height:1.0}} .lab b{{display:block;font-size:66px;font-weight:800}} .lab span{{display:block;font-size:46px;font-weight:600;margin-top:4px}}
.head{{font-size:54px;font-weight:800;line-height:1.12}}
.ub{{display:inline-block;background:#3478F6;color:#fff;font-size:46px;border-radius:40px;padding:16px 32px;white-space:nowrap}}
.qb{{display:inline-block;background:#3478F6;color:#fff;font-size:44px;border-radius:30px;padding:6px 24px}}
.wl{{background:#fff;padding-right:10px;font-size:46px;font-weight:700;color:#111;line-height:62px}}
.bv{{position:absolute;font-size:50px;font-weight:800;line-height:60px}} .tick{{position:absolute;width:120px;text-align:center;font-size:40px;font-weight:600;white-space:nowrap}}
.mb{{background:#E9E9EB;color:#111;font-size:50px;border-radius:36px;padding:16px 28px;line-height:1.15}}
.tag{{width:250px;white-space:nowrap;flex-shrink:0;font-size:50px;font-weight:800;color:{BLUE}}}
.rb{{display:flex;align-items:center;gap:14px;height:66px;font-size:48px}} .rl{{width:420px;white-space:nowrap;text-align:right;font-weight:600}} .bar{{height:52px;border-radius:0 6px 6px 0}}
.rv{{margin-bottom:64px}}
.op{{font-size:44px;font-weight:800;white-space:nowrap}}
</style></head><body>
<div style="position:absolute;left:40px;top:34px;width:1500px">
 <div class="head">Qwen3-8B’s state after an action sentence lands nearest the body part that does the action</div>
 <div style="display:flex;align-items:center;gap:16px;margin-top:22px"><span class="ub">You kick the ball.</span><span class="op">→ nearest of 23 →</span><span class="ub">A feather brushes your foot.</span></div>
</div>
<img src="data:image/png;base64,{img}" style="position:absolute;left:{bx}px;top:{by}px;width:{(cx1 - cx0) * k:.0f}px">
<svg width="{WID}" height="{HEI}" style="position:absolute;left:0;top:0">{''.join(svg)}</svg>
{''.join(lab)}
<div style="position:absolute;left:1120px;top:640px;font-size:46px;font-weight:800;line-height:1.25">
 <div style="color:{BLUE}">● part that does<br>&nbsp;&nbsp;&nbsp;the action</div><div style="color:{MUTE}">● another place</div></div>
<div style="position:absolute;left:1640px;top:34px;width:930px">
 <div class="head">% of pairs where the action’s own part is nearer than another place</div>
 <div style="position:relative;margin-top:30px;height:{len(rows) * 150 + 100}px">{grid}{bars}</div>
</div>
<div style="position:absolute;left:1640px;top:1090px;width:930px">
 <div class="head">Reverse: add an action’s vector, ask</div>
 <div style="margin:18px 0 26px;text-align:right"><span class="ub" style="font-size:44px">Where in your body do you feel something?</span></div>
 {rev}
</div>
</body></html>'''
open('fig2.html', 'w').write(html)
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=CHROME, headless=True); pg = b.new_page(viewport={'width': WID, 'height': HEI})
    pg.goto('file://' + os.path.abspath('fig2.html')); pg.wait_for_timeout(500)
    pg.screenshot(path='$BODY_WORK/fig/body_actions_v2.png'); b.close()
print('ok')
