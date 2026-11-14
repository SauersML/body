"""Figure: does Qwen3-8B know inside whether their own movement worked? Answer vs linear readout of hidden state.
Data: ~/mpd-data/vast/hypha-prod-b3d-selfjudge/{sj_scores.npz,selfjudge.npz} (sj_collect.py, sj_analyze.py). Output: ../self_judge.png"""
import os, base64, io, numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DD = "$BODY_RUNS"
D = np.load(f"{DD}/sj_scores.npz"); scene = np.load(f"{DD}/selfjudge.npz")["scene"]
v, p, y, task = D["verbal"], D["probe"], D["y"].astype(bool), D["task"]
BLUE, ORANGE, INK = '#2B5FA8', '#E07B24', '#1D1D1F'; FS = 50; W = 2400; PAD = 50
TASKS = [("reach", "Touch a named<br>body part"), ("poke_reach", "Touch where you<br>were touched"), ("grasp", "Grasp the handle")]
def auc(yy, s):
    r = np.argsort(np.argsort(s)) + 1.0
    for u in np.unique(s):  # average ties
        m = s == u
        if m.sum() > 1: r[m] = r[m].mean()
    n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(0); S = {}
for t, _ in TASKS:
    i = np.where(task == t)[0]; sc = np.unique(scene[i]); by = {s: i[scene[i] == s] for s in sc}; bv, bp = [], []
    for _ in range(2000):
        idx = np.concatenate([by[s] for s in rng.choice(sc, len(sc))]); bv.append(auc(y[idx], v[idx])); bp.append(auc(y[idx], p[idx]))
    bv, bp = np.array(bv), np.array(bp); d = bp - bv
    S[t] = dict(v=auc(y[i], v[i]), p=auc(y[i], p[i]), v_ci=np.percentile(bv, [2.5, 97.5]), p_ci=np.percentile(bp, [2.5, 97.5]),
                pval=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), n=len(i))
def pfmt(x): return "p < .001" if x < .001 else (f"p = {x:.3f}" if x < .01 else f"p = {x:.2f}").replace("0.", ".")
def b64(f):
    b = io.BytesIO(); Image.open(f).save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
def panel(t, title, w, h, first):
    d = S[t]; L, R, T, B = (190 if first else 40), 20, 250, 20; pw = w - L - R; ph = h - T - B; lo0 = 0.5
    Y = lambda val: T + ph * (1 - (val - lo0) / (1 - lo0)); bw = 230; o = []
    o.append(f'<foreignObject x="0" y="0" width="{w}" height="140"><div xmlns="http://www.w3.org/1999/xhtml" style="text-align:center;font-weight:700;font-size:{FS}px;line-height:1.15;padding-left:{L-20}px">{title}</div></foreignObject>')
    if first:
        for val, lab in ((0.5, "50%"), (0.75, "75%"), (1.0, "100%")): o.append(f'<text x="{L-16}" y="{Y(val)+17}" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T+ph}" stroke="{INK}" stroke-width="3"/>')
    for k, (key, col) in enumerate((("v", ORANGE), ("p", BLUE))):
        cx = L + pw / 2 + (k - 0.5) * 280; val = d[key]; lo, hi = d[key + "_ci"]
        o.append(f'<rect x="{cx-bw/2}" y="{Y(val)}" width="{bw}" height="{Y(lo0)-Y(val)}" fill="{col}"/>')
        for j, wd in enumerate(("answer",) if key == "v" else ("hidden", "state")):
            o.append(f'<text x="{cx}" y="{Y(lo0) - 30 - 58 * ((1 if key == "p" else 0) - j)}" text-anchor="middle" fill="#fff" font-weight="700">{wd}</text>')
        o.append(f'<line x1="{cx}" x2="{cx}" y1="{Y(hi)}" y2="{Y(lo)}" stroke="{INK}" stroke-width="5"/>')
        for yy in (hi, lo): o.append(f'<line x1="{cx-16}" x2="{cx+16}" y1="{Y(yy)}" y2="{Y(yy)}" stroke="{INK}" stroke-width="5"/>')
    x1, x2 = L + pw / 2 - 140, L + pw / 2 + 140; yb = Y(max(d["p_ci"][1], d["v_ci"][1])) - 40
    o.append(f'<path d="M{x1},{yb+18} V{yb} H{x2} V{yb+18}" fill="none" stroke="{INK}" stroke-width="4"/><text x="{(x1+x2)/2}" y="{yb-14}" text-anchor="middle">{pfmt(d["pval"])}</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(lo0)}" y2="{Y(lo0)}" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" fill="{INK}">{"".join(o)}</svg>'
def html():
    pw = (W - 2 * PAD) // 3; mh = 470
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}} h1{{font-size:{FS+4}px;font-weight:700;margin:0 0 24px 0;line-height:1.2}}
    .row{{display:flex;gap:30px;align-items:center}} .card{{border-radius:28px;padding:24px 30px;background:#F4F4F6}}
    .bubble{{border-radius:36px;padding:14px 28px;display:inline-block}} .q{{background:#2B7BF6;color:#fff}}
    .arrow{{font-size:90px;font-weight:300}} .chip{{color:#fff;border-radius:18px;padding:8px 20px;font-weight:700;display:inline-block}}
    </style></head><body><div class="page">
    <h1>Qwen's hidden state judges their own moves better than Qwen's answer</h1>
    <div class="row">
      <div style="text-align:center"><div style="display:flex;gap:16px;align-items:flex-end"><img src="{b64('r4_reach.png')}" style="height:{mh}px"><img src="{b64('r4_grasp.png')}" style="height:{mh}px"></div>
        <div style="margin-top:6px">Qwen3-8B moves<br>a simulated body</div></div>
      <div class="arrow">→</div>
      <div class="card" style="flex:1">
        <span class="bubble q">Did you succeed?</span>
        <div style="margin-top:26px;display:flex;flex-direction:column;gap:22px">
          <div><span class="chip" style="background:{ORANGE}">Qwen's answer</span> how strongly Qwen says "yes"</div>
          <div><span class="chip" style="background:{BLUE}">hidden state</span> a line fit to Qwen's layer-27 state, tested on new scenes</div>
        </div></div>
    </div>
    <div style="margin:34px 0 0 0;font-weight:700">How often a real success scores above a failure (50% = coin flip)</div>
    <div style="display:flex">{''.join(panel(t, n, pw, 800, i == 0) for i, (t, n) in enumerate(TASKS))}</div>
    </div></body></html>'''
open('fig.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as pl:
    br = pl.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_load_state("load"); pg.wait_for_timeout(800)
    h = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height"); pg.locator(".page").screenshot(path="../self_judge.png"); br.close()
print({t: {k: (np.round(x, 3).tolist() if not isinstance(x, int) else x) for k, x in d.items()} for t, d in S.items()}, 'h', h)
