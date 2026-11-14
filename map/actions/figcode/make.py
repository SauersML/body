import json, base64
from playwright.sync_api import sync_playwright
B = '$BODY_WORK/body/actions'
RD = json.load(open(f'{B}/readout.json')); res = RD['res']['warm_L28_self']; perm = RD['perm']
FW = json.load(open(f'{B}/fwd_did.json')); ST = json.load(open(f'{B}/steer.json'))['rev']
A2 = json.load(open('anchors2d.json')); AN = A2['a']
CHROME = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
BLUE, INK, GRAY = '#1F5FBF', '#111', '#C9CDD3'
# ---- body panel geometry: body.png 840x1400 shown at scale s, offset (bx, by)
s = 1.27; bx, by = 330, 300
def mid(a, b, t=0.5): return [AN[a][0] * (1 - t) + AN[b][0] * t, AN[a][1] * (1 - t) + AN[b][1] * t]
PT = {'fingers': AN['indexR'], 'hand': AN['palmR'], 'wrist': AN['wristR'], 'forearm': AN['forearmR'], 'elbow': AN['elbowR'], 'upper arm': AN['upperarmR'],
      'shoulder': AN['shoulderR'], 'toes': AN['toesL'], 'foot': mid('toesL', 'ankleL', 0.5), 'ankle': AN['ankleL'], 'shin': AN['shinL'], 'knee': AN['kneeL'],
      'thigh': AN['thighL'], 'hip': AN['hipL'], 'lips': [410, 150], 'jaw': [470, 172], 'tongue': [410, 150], 'eye': [452, 100], 'neck': AN['neckF'],
      'head': AN['crown'], 'chest': AN['chest'], 'belly': AN['belly'], 'upper back': [520, 372], 'lower back': [540, 600]}
def P(p): x, y = PT[p]; return bx + x * s, by + y * s
# label slots: (actions, side, y)
LEFT = [(['shrug'], 640), (['grip'], 870), (['push', 'lift'], 1030), (['throw', 'hike'], 1330), (['tiptoe'], 1510)]
RIGHT = [(['nod', 'wave'], 330), (['wink'], 460), (['bite', 'chew'], 590), (['carry'], 1250), (['kneel'], 1450), (['punch'], 1690), (['kick'], 1960)]
svg, lab = [], []
used = set()
for side, slots in (('L', LEFT), ('R', RIGHT)):
    for acts, y in slots:
        p = res[acts[0]]['top']; x0, y0 = P(p); hit = res[acts[0]]['hit']
        col = BLUE if hit else INK
        lx = 330 if side == 'L' else 1130
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{x0:.0f}" y2="{y0:.0f}" stroke="{col}" stroke-width="4"/>')
        if p not in used:
            hollow = p in ('lower back', 'upper back')
            svg.append(f'<circle cx="{x0:.0f}" cy="{y0:.0f}" r="15" fill="{"#fff" if hollow else col}" stroke="{col}" stroke-width="6"/>'); used.add(p)
        words = ' + '.join(acts)
        place = p
        style = f'top:{y - 62}px;' + (f'right:{2600 - lx + 18}px;text-align:right' if side == 'L' else f'left:{lx + 18}px')
        lab.append(f'<div class="lab" style="{style};color:{col}"><b>{words}</b><br><span>{place}</span></div>')
# ---- bars
def hbar(label, v, col, w=760, note='', lw=420, h=56, rh=76):
    return f'''<div class="hb" style="height:{rh}px"><div class="hl" style="width:{lw}px">{label}</div><div class="ht" style="width:{w}px;height:{rh}px">
      <div style="position:absolute;left:0;top:{(rh - h) // 2}px;height:{h}px;width:{max(v, .006) * w:.0f}px;background:{col};border-radius:0 6px 6px 0"></div>
      <div class="hv" style="top:{(rh - 56) // 2}px;left:{max(v, .006) * w + 14:.0f}px">{round(v * 100)}%{note}</div></div></div>'''
def grid(w=760, lw=420, h=100, marks=(0, .5, 1)):
    return ''.join(f'<div style="position:absolute;left:{lw + m * w:.0f}px;top:0;bottom:0;border-left:3px solid #E3E5E8"></div>' for m in marks)
aucs = [('“You kick the ball.”', perm['warm_L28_self']['auc']), ('“She kicks the ball.”', perm['warm_L28_third']['auc']),
        ('“Imagine yourself kicking…”', perm['warm_L28_imagine']['auc']), ('“You watch a man kick…”', perm['warm_L28_watch']['auc'])]
p1 = ''.join(hbar(l, v, BLUE, w=480, lw=700) for l, v in aucs) + hbar('word similarity, kick ~ foot', perm['wemb']['auc'], '#8D939B', w=480, lw=700)
FWD = [('mouth', 'bit, licked lips'), ('eye', 'eyelid twitched'), ('hand', 'touched, picked up'), ('foot', 'tucked, shifted'), ('shoulder', 'brushed'), ('knee', '')]
p2 = ''
for n, verbs in FWD:
    v = FW[f'warm|{n}|1.0']['row'][n]['part']; r = FW[f'warm|random|1.0']['row'][n]['part']
    p2 += f'''<div class="pair"><div class="hl" style="width:300px">+ {n}</div><div>
      {hbar('', v, BLUE, w=420, lw=0, h=46, rh=54, note=f'<span class="vb">{verbs}</span>' if verbs else '')}{hbar('', r, GRAY, w=420, lw=0, h=46, rh=54)}</div></div>'''
REV = [('wink', 'eye'), ('lift', 'hand or arm'), ('punch', 'hand or arm'), ('bite', 'mouth')]
p3 = ''
for a, t in REV:
    k = ST[f'warm|{a}|1.0']; p3 += f'''<div class="pair"><div class="hl" style="width:520px">+ {a} <span class="arr">→</span> {t}</div><div>
      {hbar('', k['hit'], BLUE, w=420, lw=0, h=46, rh=54)}{hbar('', k['rand'], GRAY, w=420, lw=0, h=46, rh=54)}</div></div>'''
rest = max(ST[f'warm|{a}|1.0']['hit'] for a in json.load(open(f'{B}/stim.json'))['actions'] if a not in dict(REV))
img = base64.b64encode(open('body.png', 'rb').read()).decode()
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}} body{{background:#fff;width:2600px;min-height:2000px;position:relative;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:#111}}
.lab{{position:absolute;white-space:nowrap;font-size:50px;line-height:1.08}} .lab span{{font-size:44px;font-weight:500}}
.head{{font-size:52px;font-weight:800;line-height:1.12}}
.bub{{display:inline-block;background:#1E7CF2;color:#fff;font-size:48px;border-radius:34px;padding:16px 30px;font-weight:500}}
.mbub{{display:inline-block;background:#F0F1F3;color:#111;font-size:46px;border-radius:34px;padding:16px 30px;font-style:italic}}
.hb{{display:flex;align-items:center;height:76px;position:relative}} .hl{{flex-shrink:0;font-size:46px;font-weight:700;padding-right:20px;text-align:right;white-space:nowrap}}
.ht{{position:relative;height:76px}} .hv{{position:absolute;line-height:56px;font-size:46px;font-weight:800;white-space:nowrap}} .vb{{font-weight:500;font-style:italic;margin-left:22px}}
.pair{{display:flex;align-items:center;margin-bottom:18px}} .arr{{color:#111}}
.panel{{position:absolute;left:1480px;width:1090px}}
.key{{font-size:44px;font-weight:700;display:flex;gap:30px;align-items:center}} .sw{{display:inline-block;width:46px;height:34px;border-radius:6px;vertical-align:middle;margin-right:10px}}
</style></head><body>
<div style="position:absolute;left:50px;top:40px;width:1400px">
 <div class="head">Read Qwen3-8B’s state after an action sentence; mark the body place whose touch state is closest</div>
 <div style="display:flex;align-items:center;gap:24px;margin-top:26px"><span class="bub">You kick the ball.</span><span style="font-size:46px;font-weight:700">→ layer 28 →</span>
 <span style="font-size:46px;font-weight:700;color:{BLUE}">foot</span></div>
</div>
<img src="data:image/png;base64,{img}" style="position:absolute;left:{bx}px;top:{by}px;width:{840 * s:.0f}px">
<svg width="2600" height="2440" style="position:absolute;left:0;top:0">{''.join(svg)}</svg>
{''.join(lab)}
<div class="key" style="position:absolute;left:60px;top:2150px"><span><span class="sw" style="background:{BLUE}"></span>a body part the action uses</span><span><span class="sw" style="background:{INK}"></span>another place</span></div>

<div style="position:absolute;left:1480px;top:40px;width:1090px;display:flex;flex-direction:column;gap:70px"><div>
 <div class="head">The action’s state sits nearer the body parts that do the action than other places</div>
 <div style="font-size:42px;font-weight:600;margin:12px 0 10px">share of (own part, other place) pairs where the own part is closer</div>
 <div style="position:relative">{grid(480, 700, marks=(0, .5, 1))}{p1}
 <div style="display:flex;margin-left:660px;width:560px;justify-content:space-between;font-size:40px;font-weight:600;color:#111"><span>0%</span><span>chance</span><span>100%</span></div></div>
</div>
<div>
 <div class="head">Add a place’s vector at layer 24; ask: <span class="bub" style="font-size:44px;padding:8px 22px">Describe something you just did with your body.</span></div>
 <div style="font-size:42px;font-weight:600;margin:14px 0 10px">answers that use the added place</div>
 <div class="key" style="margin-bottom:12px"><span><span class="sw" style="background:{BLUE}"></span>place vector</span><span><span class="sw" style="background:{GRAY}"></span>random vector, same size</span></div>
 {p2}
</div>
<div>
 <div class="head">Add an action’s vector; ask where Qwen feels something</div>
 <div style="font-size:42px;font-weight:600;margin:12px 0 10px">answers that name the action’s body part</div>
 {p3}
 <div style="font-size:44px;font-weight:700;margin-top:6px">other 12 actions: at most {round(rest * 100)}%</div>
</div></div>
</body></html>'''
open('fig.html', 'w').write(html)
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=CHROME, headless=True); pg = b.new_page(viewport={'width': 2600, 'height': 2440})
    pg.goto('file://' + __import__('os').path.abspath('fig.html')); pg.wait_for_timeout(500)
    pg.screenshot(path='$BODY_WORK/fig/body_actions.png', full_page=True); b.close()
print('ok', rest)
