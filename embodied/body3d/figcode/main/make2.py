"""Figure: Qwen3-8B steered by its own touch signal moves the hand to the place.
Data: steer_dump_Qwen3-8B_named.pkl (vast run hypha-prod-b3d-steerdump-named, steer_dump.py), pokes400.npy (pokes.py),
paths.pkl (paths.py). Renders: MuJoCo (rend.py). Output: ../body3d_steer_v2.png (v2: no title, no random control, no hand-path grid)"""
import os, pickle, itertools, base64, io, numpy as np, json
from PIL import Image
from rend import *
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
M.light_castshadow[:] = 0
PP = pickle.load(open('paths.pkl', 'rb')); R = PP['R']; q0 = PP['q0']
POKES = np.load('pokes400.npy')
BLUE, ORANGE, RED, INK = '#2B5FA8', '#E07B24', '#D2432F', '#1D1D1F'
DOT = [0.13, 0.13, 0.13, 1]; REDF = [0.82, 0.26, 0.18, 1]
FS = 50

def b64(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

def crop_alpha(im, pad=6):
    a = np.array(im)[..., 3]; ys, xs = np.where(a > 0)
    return (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)

def body(q, red_mask, cam, red_rad=0.009, dot_rad=0.0045):
    setpose(q); Pw, Nw = H.taxel_world(D)
    dots = [(Pw[~red_mask] + Nw[~red_mask] * 0.003, DOT, dot_rad)]
    if red_mask.any(): dots.append((Pw[red_mask] + Nw[red_mask] * 0.004, REDF, red_rad))
    return Image.fromarray(cam.render(D, dots=dots), 'RGBA')

# ---------- method renders ----------
camM = Cam(900, 1300, [0, 0, 0.98], 2.55, 180, -3)
chest = H.PARTS.index('chest')
chest_tax = np.zeros(H.NT, bool); chest_tax[POKES[POKES[:, 1] == chest, 0]] = True
all_tax = np.zeros(H.NT, bool); all_tax[POKES[:, 0]] = True
n_chest = int((POKES[:, 1] == chest).sum()); n_all = len(POKES)
imA = body(q0, chest_tax, camM, 0.011); imB = body(q0, all_tax, camM, 0.011); imS = body(q0, np.zeros(H.NT, bool), camM)
box = crop_alpha(imB); imA, imB, imS = (x.crop(box) for x in (imA, imB, imS))

# ---------- tiles ----------
camT = Cam(760, 1000, [0, 0, 0.93], 2.6, 180, -3)
ORDER = sorted(R, key=lambda t: R[t]['a1.5']['dist'])
tiles = []
for t in ['nothing added'] + ORDER + ['chest']:
    cond = 'none' if t == 'nothing added' else 'a1.5'; key = 'head' if t == 'nothing added' else t
    r = R[key][cond]; mask = np.zeros(H.NT, bool) if t == 'nothing added' else (H.TPART == H.PARTS.index(t))
    im = body(r['qend'], mask, camT, 0.013, 0.0035); xy = camT.proj(r['path'])
    tiles.append(dict(name=t, im=im, xy=xy, steer=cond != 'none'))
# common crop over all tiles (same scale for every tile)
bx = np.array([crop_alpha(d['im'], 10) for d in tiles]); X0, Y0 = bx[:, 0].min(), bx[:, 1].min(); X1, Y1 = bx[:, 2].max(), bx[:, 3].max()
for d in tiles:
    xs = d['xy'][:, 0]; ys = d['xy'][:, 1]
    X0 = min(X0, xs.min() - 12); X1 = max(X1, xs.max() + 12); Y0 = min(Y0, ys.min() - 12); Y1 = max(Y1, ys.max() + 12)
X0, Y0, X1, Y1 = int(max(0, X0)), int(max(0, Y0)), int(min(760, X1)), int(min(1000, Y1))
for d in tiles: d['im'] = d['im'].crop((X0, Y0, X1, Y1)); d['xy'] = d['xy'] - [X0, Y0]
TW, TH = X1 - X0, Y1 - Y0

tiles = tiles[1:]; big = dict(tiles[-1]); tiles = tiles[:-1]   # big = chest tile for the method strip
bb = crop_alpha(big['im'], 8); xy = big['xy']
bb = (int(min(bb[0], xy[:, 0].min() - 12)), int(min(bb[1], xy[:, 1].min() - 12)), int(max(bb[2], xy[:, 0].max() + 12)), int(max(bb[3], xy[:, 1].max() + 12)))
big['im'] = big['im'].crop(bb); big['xy'] = xy - [bb[0], bb[1]]
DN = {t: 100 * R[t]['none']['dist'] for t in ORDER}; DS = {t: 100 * R[t]['a1.5']['dist'] for t in ORDER}
json.dump(dict(order=ORDER, not_steered=DN, steered=DS, n_chest=n_chest, n_all=n_all), open('stats2.json', 'w'), indent=1)

W = 2300; PAD = 40; GRAY = '#3A3A3C'
def path_svg(xy, color, sw, scale):
    pts = ' '.join(f'{x * scale:.1f},{y * scale:.1f}' for x, y in xy); ex, ey = xy[-1] * scale
    return (f'<polyline points="{pts}" fill="none" stroke="white" stroke-width="{sw + 6}" stroke-linejoin="round" stroke-linecap="round" opacity=".9"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{sw * 1.6}" fill="{color}" stroke="white" stroke-width="4"/>')
def tile_img(d, w):
    tw_, th_ = d['im'].size; s = w / tw_; h = th_ * s
    return (f'<div style="position:relative;width:{w}px;height:{h:.0f}px"><img src="{b64(d["im"])}" style="width:{w}px;height:{h:.0f}px;position:absolute">'
            f'<svg width="{w}" height="{h:.0f}" style="position:absolute">{path_svg(d["xy"], BLUE, 6, s)}</svg></div>')
CW = 255  # column width per place
def bars_svg(h):
    L = 150; w = L + CW * 8; Tp, B = 120, 20; ymax = 55
    Y = lambda v: Tp + (1 - v / ymax) * (h - Tp - B)
    o = [f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="inherit" fill="{INK}">']
    for v in range(0, ymax + 1, 10):
        o.append(f'<line x1="{L - 14}" y1="{Y(v)}" x2="{L}" y2="{Y(v)}" stroke="{INK}" stroke-width="3"/><text x="{L - 24}" y="{Y(v) + 17}" text-anchor="end">{v}</text>')
    o.append(f'<line x1="{L}" y1="{Y(0)}" x2="{w}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/><line x1="{L}" y1="{Tp - 10}" x2="{L}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<text transform="translate(48,{(Tp + Y(0)) / 2}) rotate(-90)" text-anchor="middle">fingertip to place (cm)</text>')
    bw = 88
    for i, t in enumerate(ORDER):
        cx = L + CW * i + CW / 2
        for k, (v, col) in enumerate(((DN[t], INK), (DS[t], BLUE))):
            x = cx - bw - 4 + k * (bw + 8)
            o.append(f'<rect x="{x}" y="{Y(v)}" width="{bw}" height="{Y(0) - Y(v)}" fill="{col}" rx="6"/>')
        if i == 0:
            xn = cx - bw / 2 - 4; o.append(f'<text transform="translate({xn + 17},{Y(DN[t]) + 18}) rotate(-90)" text-anchor="end" fill="white" font-weight="700">not steered</text>')
            xs = cx + bw / 2 + 4; o.append(f'<text transform="translate({xs + 17},{Y(DS[t]) - 18}) rotate(-90)" text-anchor="start" fill="{BLUE}" font-weight="700">steered</text>')
    # brackets: reaches (steered < 10 cm) vs misses
    k = sum(DS[t] < 10 for t in ORDER); yb = 62
    for a, b, txt in ((0, k, 'hand reaches the place'), (k, 8, 'hand misses')):
        x0, x1 = L + CW * a + 20, L + CW * b - 20
        o.append(f'<path d="M{x0} {yb + 22} V{yb} H{x1} V{yb + 22}" fill="none" stroke="{INK}" stroke-width="4"/><text x="{(x0 + x1) / 2}" y="{yb - 16}" text-anchor="middle" font-weight="700">{txt}</text>')
    o.append(f'<text x="{w - 10}" y="{Tp + 90}" text-anchor="end">1 run per place</text>')
    o.append('</svg>'); return ''.join(o)
def chip(txt, col=BLUE): return f'<span class="chip" style="background:{col}">{txt}</span>'
ARW = '<div class="op">→</div>'
def html():
    mh = 420; tw = CW - 10
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.12}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}}
    .chip{{color:#fff;border-radius:18px;padding:12px 22px;font-weight:700;display:inline-block;text-align:center}}
    .op{{font-size:80px;font-weight:300;align-self:center;margin:0 0px;color:{INK}}}
    .step{{display:flex;flex-direction:column;align-items:center;text-align:center}}
    .cap{{font-weight:700;margin-bottom:12px;height:120px;display:flex;align-items:flex-end;justify-content:center}}
    </style></head><body><div class="page">
    <div style="display:flex;justify-content:space-between;align-items:stretch">
      <div class="step"><div class="cap">real touch<br>on the chest</div><img src="{b64(imA)}" style="height:{mh}px"><div>× {n_chest} touches</div></div>
      {ARW}
      <div class="step" style="justify-content:center;width:300px;white-space:nowrap"><div style="margin-top:130px">{chip("Qwen3-8B<br>activation<br>layer 18", INK)}</div><div style="margin-top:28px">averaged</div></div>
      <div class="op">−</div>
      <div class="step"><div class="cap">all touches<br>anywhere</div><img src="{b64(imB)}" style="height:{mh}px"><div>× {n_all}<br>touches</div></div>
      <div class="op">=</div>
      <div class="step" style="justify-content:center"><div style="margin-top:130px">{chip("chest<br>signal")}</div></div>
      {ARW}
      <div class="step"><div class="cap">no touch;<br>add chest signal</div><img src="{b64(imS)}" style="height:{mh}px"></div>
      {ARW}
      <div class="step"><div class="cap">hand moves<br>to the chest</div>{tile_img(big, int(mh * big['im'].size[0] / big['im'].size[1]))}</div>
    </div>
    <div style="margin-top:40px;font-weight:700">Same steps for 8 places, signal strength 1.5</div>
    <div style="display:flex;margin-left:150px;margin-top:20px">{''.join(f'<div style="width:{CW}px;display:flex;flex-direction:column;align-items:center"><div style="font-weight:700;height:118px;display:flex;align-items:flex-end;text-align:center">{d["name"]}</div>{tile_img(d, tw)}</div>' for d in tiles)}</div>
    {bars_svg(900)}
    </div></body></html>'''
open('fig2.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig2.html")); pg.wait_for_load_state("load"); pg.wait_for_timeout(800)
    print('rights', pg.evaluate("[...document.querySelectorAll('.step')].map(e=>Math.round(e.getBoundingClientRect().right))"))
    h = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height")
    pg.locator(".page").screenshot(path="../body3d_steer_v2.png"); br.close()
print(ORDER, 'height', h)
