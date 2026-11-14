"""Figure: Qwen3-8B practising in the MuJoCo body vs more expert training.
Data: ~/mpd-data/runpod/hypha-prod-b3d-sp2-{sim,self,ctrl}/selfplay_Qwen3-8B_named_*.json (selfplay.py). Renders: assets.py.
Output: ../body3d_practice.png"""
import os, json, base64, io, numpy as np
from PIL import Image
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
RP = "$BODY_RUNS{m}/selfplay_Qwen3-8B_named_{m}.json"
J = {m: json.load(open(RP.format(m=m))) for m in ("ctrl", "sim", "self")}
BLUE, LBLUE, ORANGE, INK = '#2B5FA8', '#7FA7DB', '#E07B24', '#1D1D1F'
FS = 50; W = 2600; PAD = 50
TASKS = [("reach", "Touch a named body part"), ("poke_reach", "Touch where you were touched"), ("grasp", "Grasp the handle")]
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return p, c - h, c + h
def mcn(a, b):
    x = int((a & ~b).sum()); y = int((~a & b).sum()); return binomtest(x, x + y, .5).pvalue if x + y else 1.0
def pfmt(p): return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".") if p < .01 else f"p = {p:.2f}".replace("0.", ".")
S = {}
for t, _ in TASKS:
    ok = lambda m, r: np.array(J[m]["rounds"][r]["eval"][t]["ok"], bool)
    base = ok("ctrl", 0); fin = {m: ok(m, -1) for m in J}; ex = np.array(J["ctrl"]["rounds"][0]["eval"][t]["expert"], bool)
    S[t] = dict(bars=[("before", base, None), ("ctrl", fin["ctrl"], ORANGE), ("sim", fin["sim"], BLUE), ("self", fin["self"], LBLUE)],
                expert=ex.mean(), p_sim=mcn(fin["sim"], fin["ctrl"]), p_self=mcn(fin["self"], fin["ctrl"]),
                rounds={m: len(J[m]["rounds"]) - 1 for m in J})
def b64(p):
    im = Image.open(p); b = io.BytesIO(); im.save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

def panel(t, title, w, h, first):
    d = S[t]; L, R, T, B = 150 if first else 30, 20, 230, 75; pw = w - L - R; ph = h - T - B
    y = lambda v: T + ph * (1 - v); bw = pw / 4 * 0.72; out = []
    out.append(f'<text x="{L + pw/2}" y="56" text-anchor="middle" font-weight="700">{title}</text>')
    if first:
        for v in (0, .5, 1):
            out.append(f'<text x="{L-16}" y="{y(v)+17}" text-anchor="end">{int(v*100)}%</text>')
        out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T+ph}" stroke="{INK}" stroke-width="3"/>')
    for i, (k, a, col) in enumerate(d["bars"]):
        p, lo, hi = wilson(a.sum(), len(a)); cx = L + pw / 4 * (i + .5); x0 = cx - bw / 2
        if first and not col: out.append(f'<text x="{cx}" y="{y(0)+58}" text-anchor="middle">before</text>')
        if col: out.append(f'<rect x="{x0}" y="{y(p)}" width="{bw}" height="{y(0)-y(p)}" fill="{col}"/>')
        else: out.append(f'<rect x="{x0+3}" y="{y(p)}" width="{bw-6}" height="{y(0)-y(p)}" fill="none" stroke="{INK}" stroke-width="6" stroke-dasharray="16 10"/>')
        out.append(f'<line x1="{cx}" x2="{cx}" y1="{y(hi)}" y2="{y(lo)}" stroke="{INK}" stroke-width="5"/>')
        for yy in (hi, lo): out.append(f'<line x1="{cx-16}" x2="{cx+16}" y1="{y(yy)}" y2="{y(yy)}" stroke="{INK}" stroke-width="5"/>')
    out.append(f'<line x1="{L}" x2="{L+pw}" y1="{y(0)}" y2="{y(0)}" stroke="{INK}" stroke-width="3"/>')
    ye = y(d["expert"]); out.append(f'<line x1="{L}" x2="{L+pw}" y1="{ye}" y2="{ye}" stroke="{INK}" stroke-width="5" stroke-dasharray="4 12" stroke-linecap="round"/>')
    out.append(f'<text x="{L+4}" y="{ye-16}" text-anchor="start" font-weight="700">expert</text>')
    # brackets: each practice bar vs more expert training
    top = max(wilson(a.sum(), len(a))[2] for _, a, _ in d["bars"]); top = max(top, d["expert"])
    for j, (i2, pk) in enumerate(((2, "p_sim"), (3, "p_self"))):
        x1 = L + pw / 4 * 1.5; x2 = L + pw / 4 * (i2 + .5); yb = y(top) - 40 - j * 100 - 30
        out.append(f'<path d="M{x1},{yb+18} V{yb} H{x2} V{yb+18}" fill="none" stroke="{INK}" stroke-width="4"/>')
        out.append(f'<text x="{(x1+x2)/2}" y="{yb-14}" text-anchor="middle">{pfmt(d[pk])}</text>')
    return f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" fill="{INK}">{"".join(out)}</svg>'

def legend_svg(w, h):  # bar names written once, aligned to the first panel's bars
    names = [("before", INK), ("more expert<br>training", ORANGE), ("practice,<br>simulator<br>judges", BLUE), ("practice,<br>Qwen<br>judges", LBLUE)]
    return names

def html():
    mh = 520; pw = (W - 2 * PAD) // 3
    labs = [("before practice", INK), ("more expert examples", ORANGE), ("practice: simulator says success", BLUE), ("practice: Qwen says success", '#5E8BC8')]
    lab_html = ''.join(f'<div class="lg"><span class="sw" style="{"border:6px dashed "+INK+";background:none" if c==INK else "background:"+(LBLUE if c=="#5E8BC8" else c)}"></span><span style="color:{c}">{n}</span></div>' for n, c in labs)
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}}
    h1{{font-size:{FS+4}px;font-weight:700;margin:0 0 26px 0;line-height:1.2}}
    .row{{display:flex;gap:34px;align-items:center}}
    .card{{border-radius:28px;padding:24px 30px;background:#F4F4F6}}
    .bubble{{border-radius:36px;padding:16px 28px;display:inline-block}}
    .q{{background:#2B7BF6;color:#fff}} .a{{background:#E9E9EB}}
    .arrow{{font-size:90px;font-weight:300}}
    .lg{{display:flex;align-items:center;gap:18px}} .sw{{width:60px;height:60px;display:inline-block;box-sizing:border-box}}
    .chip{{color:#fff;border-radius:18px;padding:10px 22px;font-weight:700;display:inline-block}}
    </style></head><body><div class="page">
    <h1>Practice on their own successful tries never beat more expert examples</h1>
    <div style="font-weight:700;margin:0 0 10px 0">Qwen3-8B first learned to move this body by copying an expert, then practised:</div>
    <div class="row">
      <div style="text-align:center"><div style="display:flex;gap:20px;align-items:flex-end"><img src="{b64('r4_reach.png')}" style="height:{mh}px"><img src="{b64('r4_grasp.png')}" style="height:{mh}px"></div>
        <div style="margin-top:8px">4 tries per task,<br>small random noise<br>on the movements</div></div>
      <div class="arrow">→</div>
      <div style="display:flex;flex-direction:column;gap:30px">
        <div class="card"><b style="color:{BLUE}">simulator</b><br>checks: task done?</div>
        <div class="card"><b style="color:#4F7FC4">Qwen</b><br><span class="bubble q" style="margin-top:12px">Did you succeed?</span><br><span class="bubble a" style="margin-top:12px">Yes.</span></div>
      </div>
      <div class="arrow">→</div>
      <div style="flex:1;display:flex;flex-direction:column;gap:18px">
        <div class="card" style="background:#EAF0F8">train on the successful tries<br>+ expert examples <span class="chip" style="background:{INK}">× 4 rounds</span></div>
        <div class="card" style="background:#FBEBDD"><b style="color:{ORANGE}">control</b><br>same training, expert examples only</div>
      </div>
    </div>
    <div style="height:30px"></div>
    <div style="display:flex">{''.join(panel(t, n, pw, 900, i == 0) for i, (t, n) in enumerate(TASKS))}</div>
    </div></body></html>'''

open('fig.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_load_state("load"); pg.wait_for_timeout(800)
    h = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height")
    pg.locator(".page").screenshot(path="../body3d_practice.png"); br.close()
print({t: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k != "bars"} for t, d in S.items()}, 'height', h)
