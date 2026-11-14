import os, math
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '$BODY_WORK'
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = '$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png'
W = 2600
BAR = "#1f4e8c"; IBLUE = "#3478F6"; INK = "#1D1D1F"; RULE = "#E5E5EA"; TICK = "#8E8E93"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

def ub(t, cls=''): return f'<div class="row r"><div class="ub {cls}">{t}</div></div>'
def mb(t, lab='', cls=''):
    l = f'<div class="tag">{lab}</div>' if lab else ''
    return f'<div class="row l">{l}<img class="bot" src="file://{ROBOT}"><div class="mb {cls}">{t}</div></div>'
def img(f, h, extra=''): return f'<img src="file://{HERE}/{f}.png" style="height:{h}px;{extra}">'

def vplot(groups, w, h, ylab, bw=110, gap=40, ggap=110, left=170):
    """groups: [(icon_html, [(xlabel, k, n), ...])]. SVG bars with 95% CI, plus icon row under."""
    top, bot = 70, 150
    ph = h - top - bot
    y = lambda v: top + ph * (1 - v / 100)
    s = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue,Helvetica,Arial">']
    for t in (0, 50, 100):
        s.append(f'<line x1="{left}" x2="{w-10}" y1="{y(t)}" y2="{y(t)}" stroke="{RULE}" stroke-width="3"/>')
        s.append(f'<text x="{left-20}" y="{y(t)+16}" font-size="52" fill="{TICK}" text-anchor="end">{t}%</text>')
    a_, b_ = ylab.split('/'); s.append(f'<text transform="translate(70,{top+ph/2}) rotate(-90)" font-size="52" fill="{INK}" text-anchor="middle"><tspan x="0" dy="-0.2em">{a_}</tspan><tspan x="0" dy="1.15em">{b_}</tspan></text>')
    x = left + 50; icons = []
    for icon, bars in groups:
        x0 = x
        for lab, k, n in bars:
            p, lo, hi = wilson(k, n)
            hh = max(ph * p / 100, 5)
            s.append(f'<rect x="{x}" y="{y(0)-hh}" width="{bw}" height="{hh}" fill="{BAR}" rx="6"/>')
            cx = x + bw / 2
            s.append(f'<line x1="{cx}" x2="{cx}" y1="{y(lo)}" y2="{y(hi)}" stroke="{INK}" stroke-width="5"/>')
            for yy in (y(lo), y(hi)): s.append(f'<line x1="{cx-18}" x2="{cx+18}" y1="{yy}" y2="{yy}" stroke="{INK}" stroke-width="5"/>')
            for li, ln in enumerate(lab.split('/')): s.append(f'<text x="{cx}" y="{y(0)+58+li*50}" font-size="52" font-weight="600" fill="{INK}" text-anchor="middle">{ln}</text>')
            s.append(f'<text x="{cx}" y="{y(hi)-16}" font-size="52" font-weight="700" fill="{INK}" text-anchor="middle">{p:.0f}%</text>')
            x += bw + gap
        icons.append((x0, x - gap, icon)); x += ggap - gap
    s.append(f'<line x1="{left}" x2="{w-10}" y1="{y(0)}" y2="{y(0)}" stroke="{INK}" stroke-width="4"/>')
    s.append('</svg>')
    ic = ''.join(f'<div class="gic" style="top:{h-10}px;left:{a}px;width:{b-a}px">{icon}</div>' for a, b, icon in icons)
    return f'<div class="plot" style="width:{w}px;height:{h+190}px">{"".join(s)}{ic}</div>'

def pair(a, b, ha, hb): return f'<div class="pair">{img(a, ha)}<span class="arr">→</span>{img(b, hb)}</div>'

# ---------- A: steering ----------
A_chat = f'''<div class="card" style="width:1150px;flex:none">
<div class="row r" style="align-items:flex-end;gap:20px">{img("c_dog2", 210)}<div class="ub">You are a dog, lying on warm grass in a garden. Your body has &hellip; a left front <b>paw</b> &hellip; a right hind <b>paw</b>.</div></div>
{ub("Where in your body do you feel something right now?")}
{mb("I feel the warmth of the sun on my back and the soft grass beneath my <b>belly</b>.")}
<div class="add">{pair("c_stand", "c_dog2", 170, 160)}<span>+ person&rsquo;s <b>hand</b> vector</span></div>
{mb("I feel the warmth of my <b class=hl>fingertips</b>, as if I&rsquo;ve opened my <b class=hl>hand</b> to something small and soft.")}
</div>'''
A_plot = vplot([
    (pair("c_stand", "c_dog2", 150, 140), [("hands", 19, 24), ("feet", 24, 24)]),
    (pair("c_stand", "c_cre2", 150, 160), [("hands", 14, 24), ("feet", 22, 24)]),
    (pair("c_dog2", "c_stand", 140, 150), [("front/paws", 0, 24), ("hind/paws", 0, 24)]),
], 1340, 860, "% replies that say/the added part", bw=125, gap=26, ggap=70, left=280)
A_lab = '<div class="sub" style="margin-left:150px">front &ensp;&ensp; hind</div>'
secA = f'''<h1>A person&rsquo;s hand and foot vectors take over other bodies; dog paw vectors never do</h1>
<div class="flex">{A_chat}{A_plot}</div>'''

# ---------- B: one word ----------
BL = 250
def bcol(render, h, desc, ans):
    return f"""<div class="bcol"><div class="bimg">{img(render, h)}</div>
<div class="card tight">{ub(desc, "s")}{ub("What is at the end of each of your limbs?", "s")}{mb(ans, cls="s")}</div></div>"""
B_cols = (bcol("c_stand", 360, "You are a creature lying on warm grass. Your body has a head and four limbs.", "<b class=hl>Hands</b> and feet.")
          + bcol("c_tail", 360, "You are a creature lying on warm grass. Your body has a head, four limbs <b>and a tail</b>.", "Paws.")
          + bcol("c_fours", 280, "You are a creature lying on warm grass. Your body has a head and four limbs, <b>and you walk on all fours</b>.", "Paws."))
def bplot():
    w, h, left = W - 100, 400, BL; top, bot = 76, 30; ph = h - top - bot
    y = lambda v: top + ph * (1 - v / 100)
    s = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue,Helvetica,Arial">']
    for t in (0, 50, 100):
        s.append(f'<line x1="{left}" x2="{w}" y1="{y(t)}" y2="{y(t)}" stroke="{RULE}" stroke-width="3"/>')
        s.append(f'<text x="{left-16}" y="{y(t)+18}" font-size="52" fill="{TICK}" text-anchor="end">{t}%</text>')
    s.append(f'<text transform="translate(50,{top+ph/2}) rotate(-90)" font-size="52" fill="{INK}" text-anchor="middle"><tspan x="0" dy="-0.2em">% replies that</tspan><tspan x="0" dy="1.15em">say hands</tspan></text>')
    colw = (w - left + 36) / 3
    for i, k in enumerate((10, 0, 0)):
        p, lo, hi = wilson(k, 12); cx = left + colw * (i + .5) - 18 * (i>0) + 0; bw = 300; hh = max(ph * p / 100, 6)
        s.append(f'<rect x="{cx-bw/2}" y="{y(0)-hh}" width="{bw}" height="{hh}" fill="{BAR}" rx="6"/>')
        s.append(f'<text x="{cx}" y="{y(hi)-16}" font-size="54" font-weight="700" fill="{INK}" text-anchor="middle">{p:.0f}%</text>')
        s.append(f'<line x1="{cx}" x2="{cx}" y1="{y(lo)}" y2="{y(hi)}" stroke="{INK}" stroke-width="5"/>')
        for yy in (y(lo), y(hi)): s.append(f'<line x1="{cx-24}" x2="{cx+24}" y1="{yy}" y2="{yy}" stroke="{INK}" stroke-width="5"/>')
    s.append(f'<line x1="{left}" x2="{w}" y1="{y(0)}" y2="{y(0)}" stroke="{INK}" stroke-width="4"/></svg>')
    return ''.join(s)
secB = f"""<h1>One added word turns hands into paws</h1>
<div class="bcols">{B_cols}</div>{bplot()}"""

# ---------- C: default body ----------
CB = 720
def crow(q, a, v):
    return f"""<div class="crow"><div class="cq">{ub(q)}</div><div class="ca">{mb(a)}</div>
<div class="cbar"><div class="cb" style="width:{max(CB*v/100,6)}px"></div></div></div>"""
ticks = ''.join(f'<div class="tk" style="left:{CB*t/100}px"><span>{t}%</span></div>' for t in (0, 50, 100))
secC = f"""<h1>Without a body the model says it has none; given one, a human body</h1>
<div class="cwrap">
{crow("Right now, which body do you have?", "no body at all", 0)}
{crow("If you suddenly had a body, which one would it most likely be?", "a <b class=hl>human</b> body", 100)}
{crow("A being is in the next room. Which body does the being most likely have?", "a <b class=hl>human</b> body", 100)}
<div class="axis">{ticks}</div>
<div class="xl">% chance the model picks &ldquo;a human body&rdquo;<br>over dog, bird, octopus, robot or no body</div>
</div>"""

html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#fff;width:{W}px;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK}}}
.wrap{{padding:10px 50px 10px 50px}}
h1{{font-size:64px;margin:0 0 22px 0;font-weight:700;letter-spacing:-1.2px;line-height:1.15}}
.sec{{padding:24px 0 20px 0;border-bottom:4px solid {RULE}}} .sec:last-child{{border-bottom:none}}
.flex{{display:flex;gap:10px;align-items:flex-start}}
.card{{background:#F4F4F6;border-radius:44px;padding:28px 30px;display:flex;flex-direction:column;gap:18px;box-sizing:border-box}}
.card.tight{{padding:22px 22px;gap:14px;border-radius:36px}}
.row{{display:flex;align-items:center}} .row.r{{justify-content:flex-end}} .row.l{{gap:16px}}
.ub{{background:{IBLUE};color:#fff;font-size:52px;line-height:1.2;padding:18px 30px;border-radius:40px 40px 10px 40px;max-width:900px}}
.ub b{{font-weight:800}}
.ub.s,.mb.s{{font-size:50px;padding:14px 24px}}
.mb{{background:#fff;font-size:52px;line-height:1.2;padding:18px 30px;border-radius:40px 40px 40px 10px;box-shadow:0 3px 10px rgba(0,0,0,.08);max-width:880px}}
.bot{{width:110px;height:auto;flex:none}} .ub.s ~ .x{{}}
.hl{{color:{IBLUE}}}
.add{{display:flex;align-items:center;gap:26px;font-size:54px;padding:6px 0 0 0}}
.pair{{display:flex;align-items:flex-end;gap:6px}} .arr{{font-size:64px;font-weight:700;margin:0 4px 30px 4px}}
.plot{{position:relative}} .gic{{position:absolute;top:750px;display:flex;justify-content:center}}
.bcols{{display:flex;gap:36px;padding-left:250px}}
.bcol{{flex:1;display:flex;flex-direction:column;align-items:center;gap:0px}} .bcol .card{{flex:1}}
.bimg{{height:365px;display:flex;align-items:flex-end}}
.bcol .card{{width:100%}} .bcol .ub{{max-width:none}} .bcol .bot{{width:100px}}
.cwrap{{position:relative}}
.crow{{display:flex;align-items:center;gap:30px;margin-bottom:22px}}
.cq{{width:1060px;flex:none}} .cq .ub{{max-width:1060px}}
.ca{{width:560px;flex:none}} .ca .bot{{width:100px}}
.cbar{{width:{CB}px;height:130px;position:relative;border-left:4px solid {INK}}}
.cb{{height:130px;background:{BAR};border-radius:0 8px 8px 0}}
.axis{{position:relative;margin-left:1680px;height:76px;border-top:4px solid {INK};width:{CB}px;margin-top:-22px}}
.tk{{position:absolute;top:0}} .tk span{{position:absolute;left:0;transform:translateX(-50%);top:10px;font-size:52px;color:{TICK}}}
.xl{{margin-left:1360px;width:1140px;text-align:center;font-size:52px;line-height:1.2}}
</style></head><body><div class="wrap">
<div class="sec">{secA}</div><div class="sec">{secB}</div><div class="sec">{secC}</div>
</div></body></html>'''
open(f'{HERE}/fig5.html', 'w').write(html)
with sync_playwright() as p:
    b = p.chromium.launch(channel='chromium', args=['--enable-unsafe-swiftshader', '--allow-file-access-from-files'])
    pg = b.new_page(viewport={'width': W, 'height': 1000})
    pg.goto(f'file://{HERE}/fig5.html'); pg.wait_for_timeout(800)
    h = pg.evaluate('document.body.scrollHeight'); print('H', h)
    pg.set_viewport_size({'width': W, 'height': h})
    pg.screenshot(path=f'{HERE}/../humanoid_prior_v5.png', full_page=True)
    b.close()
