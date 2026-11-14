"""body3d_steer_v3: setup (how Qwen3-8B drives the body; how the place signal is made), fingertip paths steered vs not from
the same 24 start poses, per-place distances (paired, mean + 95% CI, Holm-corrected Wilcoxon over 8 places).
Data: $BODY_RUNS/steer_confirm_Qwen3-8B_named.pkl (steer_confirm.py), a3_*.png (assets3.py)."""
import os, sys, pickle, base64, io, json, numpy as np
from PIL import Image
from scipy import stats
from rend import *
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
M.light_castshadow[:] = 0
PK = sys.argv[1] if len(sys.argv) > 1 else '$BODY_RUNS/steer_confirm_Qwen3-8B_named.pkl'
Dd = pickle.load(open(PK, 'rb')); RUNS = Dd['runs']; POSES = Dd['poses']
q0 = np.load('a3_q0.npy')
BLUE, INK, RED = '#2B5FA8', '#1D1D1F', '#D2432F'
DOT = [0.13, 0.13, 0.13, 1]; REDF = [0.82, 0.26, 0.18, 1]
FS = 50; W = 2400

def b64(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
def crop_alpha(im, pad=6):
    a = np.array(im)[..., 3]; ys, xs = np.where(a > 0)
    return (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)

# ---------- distances + stats ----------
def full(q):
    x = q0.copy(); x[H.JQ] = q; return x
def tip_dist(qpos, p):
    setpose(qpos); Pw, _ = H.taxel_world(D)
    return {h: float(np.linalg.norm(Pw[H.TPART == p] - D.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right")}
PLACES = Dd['targets']; ST = {}
for nm in PLACES:
    p = H.PARTS.index(nm)
    ds = [tip_dist(r['qpos_final'], p) for r in RUNS[nm]]; du = [tip_dist(r['qpos_final'], p) for r in RUNS['none']]
    s = np.array([min(x.values()) for x in ds]) * 100; u = np.array([min(x.values()) for x in du]) * 100
    hand = [min(x, key=x.get) for x in ds]
    n = len(s); tq = stats.t.ppf(.975, n - 1)
    ci = lambda x: (x.mean() - tq * x.std(ddof=1) / np.sqrt(n), x.mean() + tq * x.std(ddof=1) / np.sqrt(n))
    ST[nm] = dict(s=s, u=u, hand=hand, sci=ci(s), uci=ci(u), dci=ci(u - s), p=float(stats.wilcoxon(u, s).pvalue))
PRE = ['head', 'chest', 'belly', 'right shin']
# Holm over the 4 preregistered (the test), and over all 8 (printed on the figure)
def holm(keys):
    o = sorted(keys, key=lambda k: ST[k]['p']); m = 0; out = {}
    for i, k in enumerate(o): m = max(m, min(1, ST[k]['p'] * (len(o) - i))); out[k] = m
    return out
H4, H8 = holm(PRE), holm(PLACES)
for k in PLACES: ST[k]['h4'] = H4.get(k); ST[k]['h8'] = H8[k]
for k in PLACES:
    v = ST[k]; print(f"{k:13s} steered {v['s'].mean():5.1f} unsteered {v['u'].mean():5.1f} diff {(v['u']-v['s']).mean():5.1f} [{v['dci'][0]:.1f},{v['dci'][1]:.1f}] closer {(v['s']<v['u']).sum()}/24 p {v['p']:.2g} holm4 {v['h4']} holm8 {v['h8']:.2g}")
passed = [k for k in PRE if H4[k] < .05 and ST[k]['s'].mean() < ST[k]['u'].mean()]
print('PASSED', passed, 'CONFIRMED' if len(passed) >= 3 else 'NOT CONFIRMED')
json.dump({k: dict(steered=v['s'].tolist(), unsteered=v['u'].tolist(), p_wil=v['p'], holm4=v['h4'], holm8=v['h8'], diff_ci=list(v['dci'])) for k, v in ST.items()}, open('stats3.json', 'w'))
if os.environ.get('STATS_ONLY'): sys.exit()

# ---------- panel B: fingertip paths, steered vs not, same starts ----------
def smooth(pts, k=6):
    pts = np.asarray(pts); P = np.vstack([pts[:1], pts, pts[-1:]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1]); return np.array(out)
NB = 3  # first 3 start poses (seeds 7100-7102), not picked
camT = Cam(800, 1100, [0, 0, 0.95], 2.45, 180, -3)
def tip_path(qs, start, hand):
    out = []
    for q in [full(start)] + list(qs):
        setpose(q); out.append(D.site(f"tip_{hand}").xpos.copy())
    return np.array(out)
SHOWB = [k for k in sorted(PRE, key=lambda k: -(ST[k]['u'] - ST[k]['s']).mean()) if k in passed]
tilesB = []
for nm in SHOWB:
    p = H.PARTS.index(nm); setpose(q0); Pw, Nw = H.taxel_world(D); m = H.TPART == p
    im = Image.fromarray(camT.render(D, dots=[(Pw[~m] + Nw[~m] * .003, DOT, .0035), (Pw[m] + Nw[m] * .004, REDF, .012)]), 'RGBA')
    paths = []
    for i in range(NB):
        h = ST[nm]['hand'][i]
        paths.append((camT.proj(smooth(tip_path(RUNS['none'][i]['qpos'], POSES[i], h))), INK))
        paths.append((camT.proj(smooth(tip_path(RUNS[nm][i]['qpos'], POSES[i], h))), BLUE))
    tilesB.append(dict(name=nm, im=im, paths=paths))
bx = np.array([crop_alpha(d['im'], 10) for d in tilesB]); X0, Y0, X1, Y1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
for d in tilesB:
    for xy, _ in d['paths']: X0, X1, Y0, Y1 = min(X0, xy[:, 0].min() - 14), max(X1, xy[:, 0].max() + 14), min(Y0, xy[:, 1].min() - 14), max(Y1, xy[:, 1].max() + 14)
X0, Y0, X1, Y1 = int(max(0, X0)), int(max(0, Y0)), int(min(800, X1)), int(min(1100, Y1)); TW, TH = X1 - X0, Y1 - Y0
for d in tilesB:
    d['im'] = d['im'].crop((X0, Y0, X1, Y1)); d['paths'] = [(xy - [X0, Y0], c) for xy, c in d['paths']]

def tileB_html(d, w):
    s = w / TW; h = TH * s; o = []
    for xy, c in d['paths']:
        pts = ' '.join(f'{x * s:.1f},{y * s:.1f}' for x, y in xy); ex, ey = xy[-1] * s
        o.append(f'<polyline points="{pts}" fill="none" stroke="white" stroke-width="13" stroke-linejoin="round" stroke-linecap="round" opacity=".85"/>'
                 f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/>'
                 f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="14" fill="{c}" stroke="white" stroke-width="4"/>')
    for xy, c in d['paths'][::2]:
        sx, sy = xy[0] * s; o.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="13" fill="white" stroke="{INK}" stroke-width="5"/>')
    return (f'<div class="tile" style="width:{w}px"><div class="tl">{d["name"]}</div>'
            f'<div style="position:relative;width:{w}px;height:{h:.0f}px"><img src="{b64(d["im"])}" style="width:{w}px;height:{h:.0f}px;position:absolute">'
            f'<svg width="{w}" height="{h:.0f}" style="position:absolute;overflow:visible">{"".join(o)}</svg></div></div>')

# ---------- panel C: per place distances ----------
def fmtp(p): return 'p < .001' if p < .001 else ('p = ' + (f'{p:.3f}' if p < .01 else f'{p:.2f}').lstrip('0'))
def distC_svg(w, h):
    blocks = [('predicted to work', sorted(PRE, key=lambda k: -(ST[k]['u'] - ST[k]['s']).mean())),
              ('other places', sorted([k for k in PLACES if k not in PRE], key=lambda k: -(ST[k]['u'] - ST[k]['s']).mean()))]
    L, Rr, Tp, B = 150, 20, 150, 150; gap = 70
    ymax = max(max(v['s'].max(), v['u'].max()) for v in ST.values()); ymax = int(np.ceil(ymax / 10) * 10)
    Y = lambda v: Tp + (1 - v / ymax) * (h - Tp - B)
    slot = (w - L - Rr - gap) / 8
    o = [f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="inherit" fill="{INK}">']
    o.append(f'<line x1="{L}" y1="{Tp - 20}" x2="{L}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
    for v in range(0, ymax + 1, 20):
        o.append(f'<line x1="{L - 14}" y1="{Y(v)}" x2="{L}" y2="{Y(v)}" stroke="{INK}" stroke-width="3"/><text x="{L - 24}" y="{Y(v) + 17}" text-anchor="end">{v}</text>')
    o.append(f'<text transform="translate(50,{(Tp + Y(0)) / 2}) rotate(-90)" text-anchor="middle">fingertip to the place (cm)</text>')
    rng = np.random.default_rng(1); x = L + 10
    for bi, (title, keys) in enumerate(blocks):
        bx0 = x; 
        for k in keys:
            v = ST[k]; cx = x + slot / 2
            for col, arr, off, ci in ((INK, v['u'], -slot * .2, v['uci']), (BLUE, v['s'], slot * .2, v['sci'])):
                xc = cx + off
                for val in arr: o.append(f'<circle cx="{xc + rng.uniform(-slot * .1, slot * .1):.1f}" cy="{Y(val):.1f}" r="8" fill="{col}" opacity=".35"/>')
                o.append(f'<line x1="{xc}" y1="{Y(ci[0])}" x2="{xc}" y2="{Y(ci[1])}" stroke="{col}" stroke-width="8" stroke-linecap="round"/>')
                o.append(f'<line x1="{xc - slot * .17}" y1="{Y(arr.mean())}" x2="{xc + slot * .17}" y2="{Y(arr.mean())}" stroke="{col}" stroke-width="9" stroke-linecap="round"/>')
            o.append(f'<line x1="{x + 8}" y1="{Y(0)}" x2="{x + slot - 8}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
            words = k.split(' '); 
            for wi, wd in enumerate(words): o.append(f'<text x="{cx}" y="{Y(0) + 62 + wi * 56}" text-anchor="middle">{wd}</text>')
            top = Y(max(v['u'].max(), v['s'].max())) - 22
            o.append(f'<text x="{cx}" y="{min(top, Y(0) - 10)}" text-anchor="middle" font-size="{FS - 6}">{fmtp(v["h8"])}</text>')
            x += slot
        o.append(f'<text x="{(bx0 + x) / 2}" y="{Tp - 80}" text-anchor="middle" font-weight="700">{title}</text>')
        o.append(f'<path d="M{bx0 + 10} {Tp - 40} V{Tp - 60} H{x - 10} V{Tp - 40}" fill="none" stroke="{INK}" stroke-width="3"/>')
        x += gap
    o.append('</svg>'); return ''.join(o)

# ---------- page ----------
POK = np.load('pokes400.npy'); NCH = int((POK[:, 1] == H.PARTS.index('chest')).sum()); NALL = len(POK)
A = {k: Image.open(f'a3_{k}.png') for k in ('sensors', 'chest', 'all', 'plain')}
A['moved'] = Image.open('a3_moved.png')
def chip(txt, col=BLUE, fg='#fff'): return f'<span class="chip" style="background:{col};color:{fg}">{txt}</span>'
def img(im, h): return f'<img src="{b64(im)}" style="height:{h}px;display:block">'
def html():
    mh = 400; tw = (W - 80 - 3 * 30) // max(1, len(tilesB))
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.12}}
    .page{{width:{W}px;padding:36px 40px;box-sizing:border-box}}
    .hd{{font-weight:700;margin:0 0 14px 0}}
    .row{{display:flex;align-items:center;justify-content:space-between}}
    .chip{{border-radius:18px;padding:10px 22px;font-weight:700;display:inline-block;text-align:center}}
    .tok{{border:4px solid {INK};border-radius:16px;padding:8px 18px;display:inline-block;text-align:center;background:#fff}}
    .ar{{font-size:90px;font-weight:300;margin:0 4px}}
    .qwen{{background:{INK};color:#fff;border-radius:26px;padding:26px 34px;font-weight:700;text-align:center}}
    .bubble{{background:#E9E9EB;border-radius:36px;padding:16px 28px;display:inline-block}}
    .col{{display:flex;flex-direction:column;gap:16px;align-items:center}}
    .lab{{text-align:center}}
    .tile .tl{{text-align:center;font-weight:700}}
    </style></head><body><div class="page">
    <div class="hd">Qwen3-8B moves a simulated body, trained to copy an expert reaching, grasping and balancing</div>
    <div class="row">
      {img(A['sensors'], mh)}
      <div class="ar">→</div>
      <div class="col" style="align-items:stretch">
        <div class="tok">41 joint angles</div><div class="tok">982 skin sensors</div><div class="tok">ball and handle position</div><div class="lab">one input token each</div>
      </div>
      <div class="ar">→</div>
      <div class="col">
        <div class="bubble">Touch where<br>you were touched.</div>
        <div class="qwen">Qwen3-8B</div>
        <div class="lab">every 0.1 s</div>
      </div>
      <div class="ar">→</div>
      <div class="tok" style="border-color:{BLUE};color:{BLUE};font-weight:700">41 joint<br>targets</div>
      <div class="ar">→</div>
      {img(A['moved'], mh)}
    </div>
    <div class="hd" style="margin-top:34px">Make a chest signal from real touches, then add the signal with no touch</div>
    <div class="row">
      <div class="col">{img(A['chest'], mh)}<div class="lab">{NCH} real touches<br>on the chest</div></div>
      <div class="ar">→</div>
      <div class="col"><div class="qwen" style="font-size:{FS}px">Qwen3-8B<br>layer 18<br>skin token</div><div class="lab">average</div></div>
      <div class="ar">−</div>
      <div class="col">{img(A['all'], mh)}<div class="lab">same for {NALL}<br>touches anywhere</div></div>
      <div class="ar">=</div>
      {chip("chest<br>signal")}
      <div class="ar">→</div>
      <div class="col"><div class="bubble">Touch where<br>you were touched.</div><div class="lab">no touch</div>{chip("+ chest signal × 1.5")}</div>
    </div>
    <div class="hd" style="margin-top:34px">Fingertip paths from the same 3 starting poses: <span style="color:{INK}">no signal</span> vs <span style="color:{BLUE}">signal added</span></div>
    <div class="row" style="align-items:flex-start">{''.join(tileB_html(d, tw) for d in tilesB)}</div>
    <div class="hd" style="margin-top:30px">With the signal, the fingertip ends closer at all 4 predicted places (24 starting poses)</div>
    {distC_svg(W - 80, 860)}
    </div></body></html>'''
open('fig3.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig3.html")); pg.wait_for_timeout(800)
    hh = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height")
    pg.locator(".page").screenshot(path="../body3d_steer_v3.png"); br.close()
print('height', hh, 'ratio', hh / W)
