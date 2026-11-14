"""Figure: Qwen3-8B in the MuJoCo body, practice on own mistakes (DAgger) with the old vs the new scripted teacher.
Data: v1 $BODY_RUNS/b3s_Qwen3-8B_named_dagger.json, v2 ~/body3d/scale/v2/x8abs/b3s_Qwen3-8B_named_dagger.json
(192 held-out scenes per task, varied start poses). 'before' = after imitation (r0), 'after' = after 3 rounds of practice (r3) in both.
Optional 32B (env X32=path): its r3. Output ../body3d_scale_v4.png (+ .jpg)"""
import os, json, base64, io, numpy as np
from PIL import Image
from scipy.stats import fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
V1 = json.load(open("$BODY_RUNS/b3s_Qwen3-8B_named_dagger.json"))
V2 = json.load(open(os.path.expanduser("~/body3d/scale/v2/x8abs/b3s_Qwen3-8B_named_dagger.json")))
INK, ORANGE, BLUE = '#1D1D1F', '#E07B24', '#2B5FA8'
FS = 50; W = 2400; PAD = 46
TASKS = [("reach", "Touch a named<br>body part"), ("poke_reach", "Touch where you<br>were touched"), ("grasp", "Grab<br>the handle")]
def ev(R, st, t): return np.array([s for s in R["stages"] if s["stage"] == st][0]["eval"][f"new_{t}"]["ok"], bool)
ROWS = [("old teacher, before practice", lambda t: ev(V1, "r0", t), ORANGE, False), ("old teacher, after practice", lambda t: ev(V1, "r3", t), ORANGE, True),
        ("new teacher, before practice", lambda t: ev(V2, "r0", t), BLUE, False), ("new teacher, after practice", lambda t: ev(V2, "r3", t), BLUE, True)]
ROWS = [("before practice", ROWS[0][1], ORANGE, False), ("after practice", ROWS[1][1], ORANGE, True), ("before practice", ROWS[2][1], BLUE, False), ("after practice", ROWS[3][1], BLUE, True)]
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return p, c - h, c + h
def pfmt(p): return "p<.001" if p < .001 else (f"p={p:.3f}" if p < .01 else f"p={p:.2f}").replace("0.", ".")
def b64(p):
    im = Image.open(p); b = io.BytesIO(); im.save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
RH, GAP, LW = 128, 40, 400
def ys():
    y = []; acc = 0
    for i in range(len(ROWS)):
        if i == 2: acc += GAP
        y.append(acc + RH * (i + .5))
    return y, acc + RH * len(ROWS)
def panel(t, w):
    L, R, T = 16, 190, 70; pw = w - L - R; Y, hh = ys(); h = T + hh + 76; x = lambda v: L + pw * v; out = []
    ex = np.mean(V2["expert"][f"new_{t}"]["new_expert"])
    for (lab, f, col, filled), yy in zip(ROWS, Y):
        a = f(t); p, lo, hi = wilson(a.sum(), len(a)); yc = T + yy; bh = RH * .72
        if filled: out.append(f'<rect x="{x(0)}" y="{yc-bh/2}" width="{x(p)-x(0)}" height="{bh}" fill="{col}"/>')
        else: out.append(f'<rect x="{x(0)+3}" y="{yc-bh/2+3}" width="{max(0,x(p)-x(0)-6)}" height="{bh-6}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="16 10"/>')
        out.append(f'<line x1="{x(lo)}" x2="{x(hi)}" y1="{yc}" y2="{yc}" stroke="{INK}" stroke-width="5"/>')
        for v in (lo, hi): out.append(f'<line x1="{x(v)}" x2="{x(v)}" y1="{yc-16}" y2="{yc+16}" stroke="{INK}" stroke-width="5"/>')
    yb = T + hh; out.append(f'<line x1="{x(0)}" x2="{x(0)}" y1="{T}" y2="{yb}" stroke="{INK}" stroke-width="3"/><line x1="{x(0)}" x2="{x(1)}" y1="{yb}" y2="{yb}" stroke="{INK}" stroke-width="3"/>')
    for v in (0, .5, 1): out.append(f'<text x="{x(v)}" y="{yb+56}" text-anchor="{"start" if v == 0 else "end" if v == 1 else "middle"}">{int(v*100)}%</text>')
    out.append(f'<line x1="{x(ex)}" x2="{x(ex)}" y1="{T-6}" y2="{yb}" stroke="{BLUE}" stroke-width="5" stroke-dasharray="4 12" stroke-linecap="round"/>')
    out.append(f'<text x="{x(ex)}" y="{T-20}" text-anchor="middle" font-weight="700" fill="{BLUE}">teacher alone</text>')
    a, b = ROWS[1][1](t), ROWS[3][1](t); pv = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
    xb = x(1) + 22; y1, y2 = T + Y[1], T + Y[3]
    out.append(f'<path d="M{xb-18},{y1} H{xb} V{y2} H{xb-18}" fill="none" stroke="{INK}" stroke-width="4"/><text x="{xb+12}" y="{(y1+y2)/2+18}">{pfmt(pv)}</text>')
    return f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" fill="{INK}">{"".join(out)}</svg>'

def curve(kind, col):  # finger closing vs grip-to-handle distance, as the teacher computes it (expert2.py)
    w, h = 330, 200; X = lambda d: 30 + (1 - d / 0.10) * (w - 50); Yv = lambda c: h - 30 - c * (h - 60)
    if kind == "old": pts = [(0.10, 0), (0.025, 0), (0.025, 1), (0, 1)]
    else: pts = [(0.10, 0), (0.07, 0), (0.04, 1), (0, 1)]
    d = "M" + " L".join(f"{X(a):.0f},{Yv(c):.0f}" for a, c in pts)
    return (f'<svg width="{w}" height="{h}"><path d="{d}" fill="none" stroke="{col}" stroke-width="9" stroke-linejoin="round"/>'
            f'<line x1="30" x2="{w-20}" y1="{h-30}" y2="{h-30}" stroke="{INK}" stroke-width="3"/><line x1="30" x2="30" y1="20" y2="{h-30}" stroke="{INK}" stroke-width="3"/></svg>')
def curvebox(kind, col):
    return (f'<div style="display:grid;grid-template-columns:auto auto;align-items:center;column-gap:10px;font-size:40px">'
            f'<div style="display:flex;flex-direction:column;justify-content:space-between;height:170px;text-align:right">shut<span>open</span></div>{curve(kind, col)}'
            f'<div></div><div style="display:flex;justify-content:space-between;padding:0 10px 0 20px">far<span>at handle</span></div></div>')

def html():
    pw = (W - 2 * PAD - LW) // 3; Y, _ = ys()
    labels = ''.join(f'<div style="position:absolute;top:{70+yy-RH/2}px;height:{RH}px;right:0;display:flex;align-items:center;color:{c};font-weight:700">{n}</div>' for (n, _, c, _), yy in zip(ROWS, Y))
    heads = ''.join(f'<div style="width:{pw}px;text-align:center;font-weight:700">{n}</div>' for _, n in TASKS)
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}}
    h1{{font-size:{FS+4}px;font-weight:700;margin:0 0 24px 0;line-height:1.2}}
    .row{{display:flex;gap:26px;align-items:center}}
    .card{{border-radius:28px;padding:18px 24px;background:#F4F4F6}}
    .q{{background:#2B7BF6;color:#fff;border-radius:36px;padding:10px 24px;display:inline-block}}
    .arrow{{font-size:84px;font-weight:300}}
    .chip{{color:#fff;border-radius:18px;padding:6px 18px;font-weight:700;display:inline-block;background:{INK}}}
    </style></head><body><div class="page">
    <div class="row">
      <img src="{b64('grasp.png')}" style="height:600px">
      <div style="display:flex;flex-direction:column;gap:20px">
        <div class="card"><span class="q">Grab the handle.</span><br><div style="margin-top:10px">Qwen reads body sensors,<br>outputs joint targets</div></div>
        <div class="card">practice: Qwen moves,<br>teacher corrects every step <span class="chip">× 3</span></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:20px;flex:1">
        <div class="card" style="background:#FBEBDD;display:flex;gap:18px;align-items:center;justify-content:space-between"><div><b style="color:{ORANGE}">old teacher</b><br>fingers snap<br>shut</div>{curvebox("old", ORANGE)}</div>
        <div class="card" style="background:#EAF0F8;display:flex;gap:18px;align-items:center;justify-content:space-between"><div><b style="color:{BLUE}">new teacher</b><br>fingers close<br>as the hand<br>nears</div>{curvebox("new", BLUE)}</div>
      </div>
    </div>
    <div style="height:30px"></div>
    <div style="display:flex;margin-left:{LW}px">{heads}</div>
    <div style="display:flex;margin-top:10px">
      <div style="width:{LW-20}px;position:relative;margin-right:20px">{labels}</div>
      {''.join(panel(t, pw) for t, _ in TASKS)}
    </div>
    </div></body></html>'''

open('fig.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_load_state("load"); pg.wait_for_timeout(800)
    h = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height")
    pg.locator(".page").screenshot(path="../body3d_scale_v4.png"); br.close()
Image.open('../body3d_scale_v4.png').convert('RGB').save('../body3d_scale_v4.jpg', quality=92)
print('height', h)
