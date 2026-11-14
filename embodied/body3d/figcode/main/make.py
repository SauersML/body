"""Figure: Qwen3-8B steered by its own touch signal moves the hand to the place.
Data: steer_dump_Qwen3-8B_named.pkl (vast run hypha-prod-b3d-steerdump-named, steer_dump.py), pokes400.npy (pokes.py),
paths.pkl (paths.py). Renders: MuJoCo (rend.py). Output: ../body3d_steer.png"""
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
ORDER = ['head', 'chest', 'belly', 'right shin', 'left thigh', 'right thigh', 'left forearm', 'left shin']
tiles = []
for t in ['nothing added'] + ORDER:
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

# ---------- stats ----------
T = list(R); g = lambda c: np.array([R[t][c]['dist'] for t in T]) * 100
ALPH = [0, 0.25, 0.5, 1.0, 1.5, 2.0]; cond = lambda a: 'none' if a == 0 else f'a{a}'
rng = np.random.default_rng(0)
def boot(x): return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)], [2.5, 97.5])
def perm_p(d): s = [np.mean(d * np.array(z)) for z in itertools.product([1, -1], repeat=len(d))]; return float(np.mean(np.abs(s) >= abs(d.mean()) - 1e-9))
ST = {a: dict(x=g(cond(a)), m=g(cond(a)).mean(), ci=boot(g(cond(a)))) for a in ALPH}
RA = dict(x=g('rand1.5'), m=g('rand1.5').mean(), ci=boot(g('rand1.5')))
p_steer = perm_p(g('a1.5') - g('none')); p_rand = perm_p(g('a1.5') - g('rand1.5'))
stats = dict(mean={a: round(float(ST[a]['m']), 1) for a in ALPH}, ci={a: ST[a]['ci'].round(1).tolist() for a in ALPH},
             random=round(float(RA['m']), 1), random_ci=RA['ci'].round(1).tolist(), p_steer_vs_none=p_steer, p_steer_vs_random=p_rand,
             per_target={t: {c: round(100 * R[t][c]['dist'], 1) for c in R[t]} for t in T}, n_chest=n_chest, n_all=n_all)
json.dump(stats, open('stats.json', 'w'), indent=1)

# ---------- layout ----------
W = 2300; PAD = 40
def path_svg(xy, color, sw, scale):
    pts = ' '.join(f'{x * scale:.1f},{y * scale:.1f}' for x, y in xy)
    ex, ey = xy[-1] * scale
    return (f'<polyline points="{pts}" fill="none" stroke="white" stroke-width="{sw + 6}" stroke-linejoin="round" stroke-linecap="round" opacity=".9"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{sw * 1.6}" fill="{color}" stroke="white" stroke-width="4"/>')

def tile_html(d, w):
    s = w / TW; h = TH * s; col = BLUE if d['steer'] else INK
    return (f'<div class="tile" style="width:{w}px"><div class="tl">{d["name"]}</div>'
            f'<div style="position:relative;width:{w}px;height:{h:.0f}px"><img src="{b64(d["im"])}" style="width:{w}px;height:{h:.0f}px;position:absolute">'
            f'<svg width="{w}" height="{h:.0f}" style="position:absolute">{path_svg(d["xy"], col, 7, s)}</svg></div></div>')

def dose_svg(w, h):
    L, Rr, Tp, B = 150, 60, 30, 190
    xmax = 2.15; ymax = 60
    X = lambda a: L + (a + 0.12) / (xmax + 0.12) * (w - L - Rr); Y = lambda v: Tp + (1 - v / ymax) * (h - Tp - B)
    o = [f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="inherit" fill="{INK}">']
    o.append(f'<line x1="{L}" y1="{Y(0)}" x2="{w - Rr}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/><line x1="{L}" y1="{Tp}" x2="{L}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
    for v in range(0, ymax + 1, 10):
        o.append(f'<line x1="{L - 14}" y1="{Y(v)}" x2="{L}" y2="{Y(v)}" stroke="{INK}" stroke-width="3"/><text x="{L - 24}" y="{Y(v) + 17}" text-anchor="end">{v}</text>')
    for a in ALPH:
        o.append(f'<line x1="{X(a)}" y1="{Y(0)}" x2="{X(a)}" y2="{Y(0) + 14}" stroke="{INK}" stroke-width="3"/><text x="{X(a)}" y="{Y(0) + 66}" text-anchor="middle">{("%g" % a).lstrip("0") if a else "0"}</text>')
    o.append(f'<text x="{(L + w - Rr) / 2}" y="{h - 40}" text-anchor="middle">strength added (1 = one real touch)</text>')
    o.append(f'<text transform="translate(52,{(Tp + Y(0)) / 2}) rotate(-90)" text-anchor="middle">fingertip to target place (cm)</text>')
    # raw per-place dots
    for a in ALPH:
        for v in ST[a]['x']: o.append(f'<circle cx="{X(a)}" cy="{Y(v)}" r="10" fill="{BLUE}" opacity=".28"/>')
    for v in RA['x']: o.append(f'<circle cx="{X(1.5) + 70}" cy="{Y(v)}" r="10" fill="{ORANGE}" opacity=".35"/>')
    pts = ' '.join(f'{X(a)},{Y(ST[a]["m"])}' for a in ALPH)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="7" stroke-linejoin="round"/>')
    for a in ALPH:
        lo, hi = ST[a]['ci']; o.append(f'<line x1="{X(a)}" y1="{Y(lo)}" x2="{X(a)}" y2="{Y(hi)}" stroke="{BLUE}" stroke-width="7" stroke-linecap="round"/><circle cx="{X(a)}" cy="{Y(ST[a]["m"])}" r="17" fill="{BLUE}" stroke="white" stroke-width="4"/>')
    lo, hi = RA['ci']; xr = X(1.5) + 70
    o.append(f'<line x1="{xr}" y1="{Y(lo)}" x2="{xr}" y2="{Y(hi)}" stroke="{ORANGE}" stroke-width="7" stroke-linecap="round"/><rect x="{xr - 17}" y="{Y(RA["m"]) - 17}" width="34" height="34" fill="{ORANGE}" stroke="white" stroke-width="4" transform="rotate(45 {xr} {Y(RA["m"])})"/>')
    o.append(f'<text x="{xr - 12}" y="{Y(hi) - 24}" text-anchor="middle" fill="{ORANGE}">random direction</text>')
    o.append(f'<text x="{X(0.42)}" y="{Y(4)}" text-anchor="middle" fill="{BLUE}">place signal</text>')
    # p bracket nothing added vs 1.5
    yb = Y(57)
    o.append(f'<path d="M{X(0)} {yb + 18} V{yb} H{X(1.5)} V{yb + 18}" fill="none" stroke="{INK}" stroke-width="3"/><text x="{(X(0) + X(1.5)) / 2}" y="{yb - 14}" text-anchor="middle">p = {p_steer:.2f}</text>')
    o.append('</svg>'); return ''.join(o)

def chip(txt, col=BLUE): return f'<span class="chip" style="background:{col}">{txt}</span>'

def html():
    mh = 580  # method render height
    sA = mh / imA.height; wA = imA.width * sA; wS = imS.width * (mh / imS.height)
    tw = 380
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
    .page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}}
    h1{{font-size:{FS}px;font-weight:700;margin:0 0 30px 0;line-height:1.2}}
    .row{{display:flex;gap:40px;align-items:stretch}}
    .card{{border-radius:28px;padding:26px 30px}}
    .hd{{font-weight:700;margin-bottom:18px}}
    .chip{{color:#fff;border-radius:18px;padding:10px 24px;font-weight:700;display:inline-block;text-align:center}}
    .op{{font-size:120px;font-weight:300;align-self:center;margin:0 6px}}
    .lab{{text-align:center;margin-top:6px}}
    .tile .tl{{text-align:center;height:62px}}
    .grid{{display:grid;grid-template-columns:repeat(3,{tw}px);column-gap:18px;row-gap:14px}}
    .bubble{{background:#E9E9EB;border-radius:36px;padding:20px 30px;display:inline-block}}
    </style></head><body><div class="page">
    <h1>Steering Qwen3-8B with their own layer-18 signal for a touched place, with nothing touching, moves the hand to the head, chest, belly and right shin</h1>
    <div class="row">
      <div class="card" style="background:#F4F4F6;flex:1.35">
        <div class="hd">1&nbsp; Layer-18 activation, averaged over real touches</div>
        <div style="display:flex;align-items:flex-end;justify-content:center">
          <div><img src="{b64(imA)}" style="height:{mh}px"><div class="lab">{n_chest} on the chest</div></div>
          <div class="op" style="margin-bottom:{mh/2}px">−</div>
          <div><img src="{b64(imB)}" style="height:{mh}px"><div class="lab">{n_all} anywhere</div></div>
          <div class="op" style="margin-bottom:{mh/2}px">=</div>
          <div style="align-self:center;margin-bottom:60px">{chip("chest<br>signal")}</div>
        </div>
      </div>
      <div class="card" style="background:#FAF0E6;flex:0.65">
        <div class="hd">2&nbsp; No touch; Qwen3-8B moves the body</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:24px">
          <div style="display:flex;flex-direction:column;gap:40px;align-items:flex-start">
            <div class="bubble">Touch where<br>you were<br>touched.</div>
            <div>{chip("+ chest signal<br>at layer 18")}</div>
          </div>
          <img src="{b64(imS)}" style="height:{mh}px">
        </div>
      </div>
    </div>
    <div class="row" style="margin-top:34px">
      <div>
        <div class="hd">3&nbsp; Fingertip path, signal strength 1.5</div>
        <div class="grid">{''.join(tile_html(d, tw) for d in tiles)}</div>
      </div>
      <div style="flex:1">
        <div class="hd">Distance at the end of the movement</div>
        {dose_svg(1000, 1640)}
      </div>
    </div>
    </div></body></html>'''

open('fig.html', 'w').write(html())
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "$BODY_WORK")
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": 1000}).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_load_state("load"); pg.wait_for_timeout(800)
    h = pg.evaluate("document.querySelector('.page').getBoundingClientRect().height")
    pg.locator(".page").screenshot(path="../body3d_steer.png"); br.close()
print(json.dumps(stats)[:600], 'height', h)
