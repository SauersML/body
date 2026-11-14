"""Figure: Qwen3-8B practising in the MuJoCo body, judged by the simulator or by themselves (raw yes, stricter yes, hidden-state probe), vs more expert training.
Data: ~/mpd-data/runpod/hypha-prod-b3d-sp2-{sim,self,ctrl}/ and ~/mpd-data/vast/hypha-prod-b3d-sp3-{vcal,probe}/ selfplay_Qwen3-8B_named_*.json
(body3d/selfplay.py; stats body3d/sp3_stats.py). Renders: assets.py. Previous version: make_v1_sp2.py. Output: ../body3d_practice.png"""
import os, json, base64, io, numpy as np
from PIL import Image
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
RP = "$BODY_RUNS{m}/selfplay_Qwen3-8B_named_{m}.json"
VA = "$BODY_RUNS{m}/selfplay_Qwen3-8B_named_{m}.json"
J = {m: json.load(open(RP.format(m=m))) for m in ("ctrl", "sim", "self")}
J.update({m: json.load(open(VA.format(m=m))) for m in ("vcal", "probe")})
INK, ORANGE, BLUE, LBLUE, GREEN, RED = '#1D1D1F', '#E07B24', '#2B5FA8', '#7FA7DB', '#1A9E77', '#D2432F'
FS = 50; W = 2600; PAD = 50
TASKS = [("reach", "Touch a named<br>body part"), ("poke_reach", "Touch where<br>you were touched"), ("grasp", "Grasp<br>the handle")]
ROWS = [("before", "before", INK), ("ctrl", "control", ORANGE), ("sim", "simulator", BLUE),
        ("self", "Qwen's yes", '#5E8BC8'), ("vcal", "stricter", GREEN), ("probe", "probe", RED)]
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return p, c - h, c + h
def mcn(a, b):
    x = int((a & ~b).sum()); y = int((~a & b).sum()); return binomtest(x, x + y, .5).pvalue if x + y else 1.0
def pfmt(p): return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".") if p < .01 else f"p = {p:.2f}".replace("0.", ".")
def ok(m, t, r=-1): return np.array(J["ctrl" if m == "before" else m]["rounds"][0 if m == "before" else r]["eval"][t]["ok"], bool)
def b64(p):
    im = Image.open(p); b = io.BytesIO(); im.save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

RH, LW = 112, 330  # row height, label column width
def panel(t, w):
    L, R, T = 20, 235, 75; pw = w - L - R; h = T + RH * len(ROWS) + 80; x = lambda v: L + pw * v; out = []
    ex = np.mean(J["ctrl"]["rounds"][0]["eval"][t]["expert"])
    for i, (m, _, col) in enumerate(ROWS):
        a = ok(m, t); p, lo, hi = wilson(a.sum(), len(a)); yc = T + RH * (i + .5); bh = RH * .7
        if m == "before": out.append(f'<rect x="{x(0)}" y="{yc-bh/2+3}" width="{x(p)-x(0)-3}" height="{bh-6}" fill="none" stroke="{INK}" stroke-width="6" stroke-dasharray="16 10"/>')
        else: out.append(f'<rect x="{x(0)}" y="{yc-bh/2}" width="{x(p)-x(0)}" height="{bh}" fill="{col}"/>')
        out.append(f'<line x1="{x(lo)}" x2="{x(hi)}" y1="{yc}" y2="{yc}" stroke="{INK}" stroke-width="5"/>')
        for v in (lo, hi): out.append(f'<line x1="{x(v)}" x2="{x(v)}" y1="{yc-16}" y2="{yc+16}" stroke="{INK}" stroke-width="5"/>')
    yb = T + RH * len(ROWS); out.append(f'<line x1="{x(0)}" x2="{x(0)}" y1="{T}" y2="{yb}" stroke="{INK}" stroke-width="3"/>')
    out.append(f'<line x1="{x(0)}" x2="{x(1)}" y1="{yb}" y2="{yb}" stroke="{INK}" stroke-width="3"/>')
    for v in (0, .5, 1): out.append(f'<text x="{x(v)}" y="{yb+56}" text-anchor="{"start" if v == 0 else "end" if v == 1 else "middle"}">{int(v*100)}%</text>')
    out.append(f'<line x1="{x(ex)}" x2="{x(ex)}" y1="{T-8}" y2="{yb}" stroke="{INK}" stroke-width="5" stroke-dasharray="4 12" stroke-linecap="round"/>')
    # bracket: probe vs Qwen's raw yes (paired scenes, exact McNemar)
    i1, i2 = [k for k, (m, _, _) in enumerate(ROWS) if m in ("self", "probe")]; xb = x(1) + 22; y1, y2 = T + RH * (i1 + .5), T + RH * (i2 + .5)
    out.append(f'<path d="M{xb-18},{y1} H{xb} V{y2} H{xb-18}" fill="none" stroke="{INK}" stroke-width="4"/>')
    pv = pfmt(mcn(ok("probe", t), ok("self", t))); a, b = pv.split(" ", 1)
    out.append(f'<text x="{xb+14}" y="{(y1+y2)/2+18}">{pv.replace(" ", "")}</text>')
    out.append(f'<text x="{x(ex)}" y="{T-22}" text-anchor="middle" font-weight="700">expert</text>')
    return f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" fill="{INK}">{"".join(out)}</svg>'

def html():
    mh = 470; pw = (W - 2 * PAD - LW) // 3
    labels = ''.join(f'<div style="height:{RH}px;display:flex;align-items:center;justify-content:flex-end;color:{c};font-weight:700">{n}</div>' for _, n, c in ROWS)
    heads = ''.join(f'<div style="width:{pw}px;text-align:center;font-weight:700">{n}</div>' for _, n in TASKS)
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}}
    h1{{font-size:{FS+4}px;font-weight:700;margin:0 0 26px 0;line-height:1.2}}
    .row{{display:flex;gap:30px;align-items:center}}
    .card{{border-radius:28px;padding:20px 26px;background:#F4F4F6}}
    .bubble{{border-radius:36px;padding:12px 26px;display:inline-block}}
    .q{{background:#2B7BF6;color:#fff}} .a{{background:#E9E9EB}}
    .arrow{{font-size:90px;font-weight:300}}
    .chip{{color:#fff;border-radius:18px;padding:8px 20px;font-weight:700;display:inline-block}}
    </style></head><body><div class="page">
    <h1>A probe on Qwen's hidden state stops self-practice from hurting;<br>more expert examples still win</h1>
    <div class="row">
      <div style="text-align:center"><div style="display:flex;gap:16px;align-items:flex-end"><img src="{b64('r4_reach.png')}" style="height:{mh}px"><img src="{b64('r4_grasp.png')}" style="height:{mh}px"></div>
        <div style="margin-top:8px;white-space:nowrap">Qwen3-8B tries,<br>with small noise</div></div>
      <div class="arrow">→</div>
      <div style="display:grid;grid-template-columns:auto auto;gap:22px">
        <div class="card"><b style="color:{BLUE}">simulator</b><br>task done?</div>
        <div class="card"><b style="color:#5E8BC8">Qwen's yes</b> <span class="bubble q">Did you succeed?</span><br><span class="bubble a" style="margin-top:10px;margin-left:auto">Yes.</span></div>
        <div class="card"><b style="color:{GREEN}">stricter</b><br>only Qwen's surest yes</div>
        <div class="card"><b style="color:{RED}">probe</b><br>reads Qwen's hidden state</div>
      </div>
      <div class="arrow">→</div>
      <div style="display:flex;flex-direction:column;gap:22px"><div class="card" style="background:#EAF0F8">train on tries<br>judged successful<br>+ expert examples<br><span class="chip" style="background:{INK};margin-top:10px">× 4 rounds</span></div>
        <div class="card" style="background:#FBEBDD"><b style="color:{ORANGE}">control</b><br>only expert examples</div></div>
    </div>
    <div style="height:34px"></div>
    <div style="display:flex;margin-left:{LW}px">{heads}</div>
    <div style="display:flex;margin-top:14px">
      <div style="width:{LW}px;padding-top:75px">{labels}</div>
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
    pg.locator(".page").screenshot(path="../body3d_practice.png"); br.close()
print('height', h)
