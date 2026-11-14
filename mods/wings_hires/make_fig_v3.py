import os
import json, numpy as np, html, sys
sys.path.insert(0, os.path.expandvars('$BODY_REPO/mods/wings_hires'))
import wing3d as W3
O = os.path.expandvars('$BODY_REPO/mods/wings_hires/out8b')
F = json.load(open(f'{O}/figdata.json'))
OUT = os.path.expandvars('$BODY_REPO/figures/out/body_wings_v3.png')
WD, HT = 2600, 3240
CF, CM, CB = (0.11, 0.33, 0.66), (0.91, 0.78, 0.36), (0.74, 0.24, 0.09)
STOPS = [(0.0, CF), (0.5, CM), (1.0, CB)]
def cmap(x):
    x = np.clip(x, 0, 1); out = np.zeros((len(x), 3))
    for (a, ca), (b, cb) in zip(STOPS[:-1], STOPS[1:]):
        m = (x >= a) & (x <= b); w = ((x[m] - a) / (b - a))[:, None]; out[m] = (1 - w) * np.array(ca) + w * np.array(cb)
    return out
hexc = lambda c: '#%02x%02x%02x' % tuple(int(255 * v) for v in c)
HF, HM, HB = hexc(CF), '#c99a1c', hexc(CB)
UC = np.array([0.04, 0.27, 0.5, 0.73, 0.96]); VC = np.array([1 / 6, 0.5, 5 / 6])
def field(G, si, U, V):
    G = np.array(G)[si]
    ui = np.clip(np.interp(U, UC, np.arange(5)), 0, 4); vi = np.clip(np.interp(V, VC, np.arange(3)), 0, 2)
    u0 = np.floor(ui).astype(int).clip(0, 3); v0 = np.floor(vi).astype(int).clip(0, 1); fu = ui - u0; fv = vi - v0
    return (1-fu)*(1-fv)*G[u0, v0] + fu*(1-fv)*G[u0+1, v0] + (1-fu)*fv*G[u0, v0+1] + fu*fv*G[u0+1, v0+1]
def painter(key):
    G = F[key]['reg_all']
    return lambda sd, U, V: cmap(field(G, 0 if sd == 1 else 1, U, V)) ** 2.2
FONT = "Inter, Helvetica Neue, Arial, sans-serif"
def T(x, y, t, sz=50, w=400, anc='start', col='#111', it=False):
    st = ' font-style="italic"' if it else ''
    return f'<text x="{x}" y="{y}" font-size="{sz}" font-weight="{w}" text-anchor="{anc}" fill="{col}"{st}>{t}</text>'
def rich(x, y, lines, sz=48, lh=62, col='#111'):
    # lines: list of lists of (text, style) ; style: '' | 'b' (bold) | color hex for bold colored
    o = []
    for i, segs in enumerate(lines):
        sp = ''
        for t, s in segs:
            a = ''
            if s == 'b': a = ' font-weight="700"'
            elif s == 'x': a = ' font-weight="700" fill="#888" text-decoration="line-through"'
            elif s: a = f' font-weight="700" fill="{s}"'
            sp += f'<tspan{a}>{html.escape(t)}</tspan>'
        o.append(f'<text x="{x}" y="{y + i * lh}" font-size="{sz}" fill="{col}">{sp}</text>')
    return ''.join(o)
def arrow(x1, y1, x2, y2, col='#333', w=6):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}" marker-end="url(#ah)"/>'
def step(n, x, y, title, col='#111'):
    return (f'<circle cx="{x+30}" cy="{y-18}" r="32" fill="#111"/>' + T(x + 30, y, str(n), 44, 700, 'middle', '#fff') + T(x + 82, y, title, 58, 700, col=col))
# flat wing pictogram from the 3-D wing's own outline (person's left wing seen from behind)
def wing_icon(x0, y0, sc, hi=None, q=False):
    u = np.linspace(0, 1, 60)
    L = W3.lead(u, 1); Tr = W3.trail(u, 1)
    def pt(p): return (x0 + p[0] * sc, y0 - (p[1] - 4.55) * sc)
    def band(v0, v1):
        A = [L + v0 * (Tr - L)][0]; B = [L + v1 * (Tr - L)][0]
        pts = [pt(p) for p in A] + [pt(p) for p in B[::-1]]
        return 'M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts) + 'Z'
    o = []
    for k, (v0, v1, c) in enumerate([(0, 1/3, HF), (1/3, 2/3, hexc(CM)), (2/3, 1, HB)]):
        op = 1.0 if (hi is None or hi == k) else 0.18
        o.append(f'<path d="{band(v0, v1)}" fill="{c}" fill-opacity="{op}" stroke="#fff" stroke-width="3"/>')
    o.append(f'<path d="{band(0, 1)}" fill="none" stroke="#333" stroke-width="3"/>')
    return ''.join(o)
def layers(x, y, w, h, n=12, hi=8):
    o = [T(x + w / 2, y - 20, 'Qwen3-8B', 48, 700, 'middle')]
    g = h / n
    for i in range(n):
        yy = y + h - (i + 1) * g + 4
        c = '#2f7d4f' if i == hi else '#d9d9d9'
        o.append(f'<rect x="{x}" y="{yy:.0f}" width="{w}" height="{g-8:.0f}" rx="6" fill="{c}"/>')
    return ''.join(o), y + h - (hi + 1) * g + 4 + (g - 8) / 2
def vec(x, y, seed):
    r = np.random.default_rng(seed); o = []
    for i in range(9):
        v = r.uniform(0.15, 0.85); c = hexc((0.18 + 0.6 * v * 0.2, 0.49 * (0.6 + 0.4 * v), 0.31 * (0.6 + 0.4 * v)))
        g = int(70 + 160 * v); o.append(f'<rect x="{x}" y="{y + i * 30}" width="44" height="26" rx="4" fill="rgb({int(g*0.55)},{int(g*0.85)},{int(g*0.62)})"/>')
    return ''.join(o)
svg = [f'<svg width="{WD}" height="{HT}" style="position:absolute;left:0;top:0;font-family:{FONT}">',
       '<defs><marker id="ah" markerWidth="5" markerHeight="5" refX="3.2" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#333"/></marker></defs>']
# ---------- headline
svg.append(T(60, 92, 'Given wings in a chat, Qwen3-8B places a touch on the right', 66, 700))
svg.append(T(60, 172, 'part of a wing even when the sentence never says “wing”', 66, 700))
# ---------- 1: two chats
y1 = 290
svg.append(step(1, 60, y1, 'Two chats'))
def bubble(x, y, w, h, fill, stroke=None, dash=False):
    st = (f' stroke="{stroke}" stroke-width="4"' + (' stroke-dasharray="18 12"' if dash else '')) if stroke else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{fill}"{st}/>'
xa, wa, xb, wb = 60, 1720, 1830, 710
svg.append(T(xa, y1 + 100, 'Chat A: Qwen gets wings', 54, 700, col='#a5401a'))
svg.append(T(xb, y1 + 100, 'Chat B: no wings', 54, 700))
by = y1 + 135
for x, w, t in ((xa, wa, 'warm-up: hello, body scan, breathing'), (xb, wb, 'same warm-up')):
    svg.append(bubble(x, by, w, 90, '#eeeeee'))
    svg.append(T(x + 36, by + 62, t, 48, 400, col='#111'))
wy = by + 110
svg.append(bubble(xa, wy, wa, 250, '#f6e7dc'))
svg.append(rich(xa + 36, wy + 70, [[('User: ', 'b'), ('“Wings are sprouting from your shoulder blades… Each wing has a', '')],
                                   [('stiff, ribbed front edge', HF), (', a ', ''), ('broad thin membrane', HM), (' dusted with fine scales…,', '')],
                                   [('and a ', ''), ('soft, thin back edge', HB), (' that flutters.”', '')]], 48, 64))
svg.append(bubble(xa + 200, wy + 270, wa - 200, 96, '#eeeeee'))
svg.append(rich(xa + 236, wy + 336, [[('Qwen: ', 'b'), ('“The stiff front edge is firm against my skin…”', '')]], 48))
svg.append(bubble(xb, wy, wb, 366, '#ffffff', '#9a9a9a', True))
svg.append(T(xb + wb / 2, wy + 200, 'no wings message', 52, 400, 'middle', '#555'))
# ---------- 2/3: train and test rows (same picture twice)
def row(n, y, title, lines, out_lbl, hi, tag):
    o = [step(n, 60, y, title)]
    cy = y + 50; hb = len(lines) * 62 + 50
    o.append(f'<rect x="60" y="{cy}" width="1120" height="{hb}" rx="22" fill="#f4f4f4"/>')
    o.append(rich(96, cy + 70, lines, 48, 62))
    tx, ty = 1222, cy + hb / 2 - 15
    o.append(f'<rect x="{tx}" y="{ty}" width="190" height="90" rx="14" fill="#fff" stroke="#2f7d4f" stroke-width="5"/>' + T(tx + 95, ty + 62, 'Qwen:', 48, 700, 'middle', '#2f7d4f'))
    o.append(T(tx + 95, ty + 145, 'reply start', 48, 400, 'middle', '#2f7d4f'))
    o.append(arrow(1182, ty + 45, 1212, ty + 45))
    L, ly = layers(1500, cy + 50, 150, max(hb - 40, 230))
    o.append(L)
    sb = cy + 50 + max(hb - 40, 230)
    o.append(f'<path d="M{tx+190},{ty+45} C1465,{ty+45} 1465,{ly} 1492,{ly}" fill="none" stroke="#2f7d4f" stroke-width="6" marker-end="url(#ah)"/>')
    o.append(T(1575, sb + 56, 'layer 24', 48, 700, 'middle', '#2f7d4f'))
    vy = ly - 135
    o.append(arrow(1660, ly, 1712, ly, '#2f7d4f'))
    o.append(vec(1726, vy, n))
    o.append(T(1748, vy + 330, 'state', 48, 400, 'middle'))
    o.append(arrow(1782, ly, 1832, ly))
    o.append(f'<rect x="1842" y="{ly-62}" width="200" height="124" rx="18" fill="#111"/>' + T(1942, ly - 8, 'linear', 46, 700, 'middle', '#fff') + T(1942, ly + 42, 'read-out', 46, 700, 'middle', '#fff'))
    o.append(T(1942, ly + 120, 'weighted sum', 48, 400, 'middle'))
    o.append(arrow(2044, ly, 2092, ly))
    o.append(wing_icon(2082, ly - 40, 46, hi))
    o.append(T(2290, ly + 205, out_lbl, 48, 700, 'middle', HF))
    o.append(T(1180, cy + hb + 58, tag, 48, 700, 'end'))
    return ''.join(o), max(cy + hb + 58, ly + 205)
y2 = wy + 366 + 125
r2, e2 = row(2, y2, 'In each chat, fit a read-out on sentences that say “wing”',
             [[('“A fingertip presses gently on the ', ''), ('stiff', HF)],
              [('front edge', HF), (' of your left ', ''), ('wing', 'b'), (', near the base,', '')],
              [('on the upper side.”', '')]], 'answer: front edge', None, '× 2,400 sentences: 60 spots × 40 wordings')
svg.append(r2)
y3 = e2 + 115
r3, e3 = row(3, y3, 'Test on sentences that never say “wing”',
             [[('“…presses gently on the ', ''), ('stiff ribbed rim', HF)],
              [('at the front', HF), (' of the thing that grows from', '')],
              [('behind your left shoulder blade, a hand’s', '')],
              [('width out from your back, on its top face.”', '')]], 'guess: front edge', 0, '× 2,400 sentences, same 60 spots')
svg.append(r3)
qy = e3 + 85
svg.append(rich(60, qy, [[('Qwen’s reply, Chat A: ', 'b'), ('“…the stiff, ribbed edge of my left wing, where it meets my back”', '#a5401a')]], 48))
# ---------- 4: result
y4 = qy + 125
svg.append(step(4, 60, y4, 'Guesses on the test sentences: clean bands only in Chat A'))
s = 66.0; Yc = 2.6; OX = 9.6
cy0 = y4 - 30; ch = 770
def scr(x, y): return WD / 2 + x * s, cy0 + ch / 2 - (y - Yc) * s
lx, _ = scr(-OX, 0); rx, _ = scr(OX, 0)
svg.append(T(lx, y4 + 100, 'Chat A: Qwen gets wings', 54, 700, 'middle', '#a5401a'))
svg.append(T(rx, y4 + 100, 'Chat B: no wings', 54, 700, 'middle'))
accW, accN = F['W']['reg_acc'], F['N']['reg_acc']
yb = cy0 + ch + 20
svg.append(T(lx, yb + 70, f'{accW*100:.0f}% right', 84, 700, 'middle', '#a5401a'))
svg.append(T(rx, yb + 70, f'{accN*100:.0f}% right', 84, 700, 'middle'))
svg.append(T(WD / 2, yb + 160, 'Random guessing: 33%.  Raw word embeddings instead of Qwen’s state: 22%', 48, 400, 'middle'))
# key
svg.append(T(WD / 2, yb + 255, 'color at each spot = average guess over that spot’s 40 test sentences', 48, 700, 'middle'))
kx, ky, kw, kh = 600, yb + 290, 1400, 72
grad = ''.join(f'<stop offset="{a}" stop-color="{hexc(c)}"/>' for a, c in STOPS)
svg.append(f'<defs><linearGradient id="g">{grad}</linearGradient></defs><rect x="{kx}" y="{ky}" width="{kw}" height="{kh}" rx="14" fill="url(#g)"/>')
svg.append(T(kx + 28, ky + 52, 'front edge', 48, 700, 'start', '#fff'))
svg.append(T(kx + kw / 2, ky + 52, 'membrane', 48, 700, 'middle', '#3a2c00'))
svg.append(T(kx + kw - 28, ky + 52, 'back edge', 48, 700, 'end', '#fff'))
cav = ky + kh + 110
pass
svg.append('</svg>')
H = cav + 85
svg[0] = svg[0].replace(f'height="{HT}"', f'height="{H}"')
print('height', H)
W3.render(WD, int(H), ''.join(svg), (0, cy0, WD, ch), [(-OX, painter('W')), (OX, painter('N'))], OUT, Yc, s)
print(OUT)
