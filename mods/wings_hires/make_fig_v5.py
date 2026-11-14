import os
import json, numpy as np, html, sys
sys.path.insert(0, os.path.expandvars('$BODY_REPO/mods/wings_hires'))
import wing3d as W3
O = os.path.expandvars('$BODY_REPO/mods/wings_hires/out8b')
A = json.load(open(f'{O}/ana_2.json'))['transfer']
accW, accN = A['W_wing>W_desc']['reg_acc'], A['N_wing>N_desc']['reg_acc']
CL = json.load(open(f'{O}/clsdata.json'))   # same classifier, per-spot answers (clsdata.py)
assert abs(CL['W']['acc'] - accW) < 1e-9 and abs(CL['N']['acc'] - accN) < 1e-9
OUT = os.path.expandvars('$BODY_REPO/figures/out/body_wings_v5.png')
WD = 2600
HF, HM, HB = '#1c54a8', '#e3b93c', '#bd3d17'   # front edge, membrane, back edge
TM = '#a07a08'                                  # membrane text (darker gold)
WC = '#a5401a'                                  # wings-chat color
RGB = lambda h: np.array([int(h[i:i+2], 16) / 255 for i in (1, 3, 5)])
FONT = "Inter, Helvetica Neue, Arial, sans-serif"
def T(x, y, t, sz=50, w=400, anc='start', col='#111'):
    return f'<text x="{x}" y="{y}" font-size="{sz}" font-weight="{w}" text-anchor="{anc}" fill="{col}">{html.escape(t)}</text>'
def rich(x, y, lines, sz=48, lh=62, col='#111'):
    o = []
    for i, segs in enumerate(lines):
        sp = ''
        for t, s in segs:
            a = ''
            if s == 'b': a = ' font-weight="700"'
            elif s: a = f' font-weight="700" fill="{s}"'
            sp += f'<tspan{a}>{html.escape(t)}</tspan>'
        o.append(f'<text x="{x}" y="{y + i * lh}" font-size="{sz}" fill="{col}">{sp}</text>')
    return ''.join(o)
def arrow(x1, y1, x2, y2, col='#333', w=6):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}" marker-end="url(#ah)"/>'
def step(n, x, y, title, col='#111'):
    return (f'<circle cx="{x+32}" cy="{y-19}" r="34" fill="#111"/>' + T(x + 32, y - 2, str(n), 46, 700, 'middle', '#fff') + T(x + 90, y, title, 60, 700, col=col))
def bubble(x, y, w, h, fill, stroke=None, dash=False):
    st = (f' stroke="{stroke}" stroke-width="4"' + (' stroke-dasharray="18 12"' if dash else '')) if stroke else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{fill}"{st}/>'
def vec(x, y, seed, n=5, c=44):
    r = np.random.default_rng(seed); o = []
    for i in range(n):
        g = r.uniform(0.25, 1.0)
        o.append(f'<rect x="{x + i * (c + 6)}" y="{y - c/2}" width="{c}" height="{c}" rx="6" fill="rgb({int(30+190*(1-g))},{int(110+120*(1-g))},{int(70+170*(1-g))})"/>')
    return ''.join(o)
def chip(x, y, w, lbl, col, tcol='#fff', mark=''):
    return (f'<rect x="{x}" y="{y-42}" width="{w}" height="84" rx="42" fill="{col}"/>' + T(x + w / 2, y + 17, lbl + mark, 50, 700, 'middle', tcol))
def guesser(x, y):
    return f'<rect x="{x}" y="{y-44}" width="220" height="88" rx="16" fill="#111"/>' + T(x + 110, y + 16, 'guesser', 46, 700, 'middle', '#fff')


svg = [f'<svg width="{WD}" height="HT" style="position:absolute;left:0;top:0;font-family:{FONT}">',
       '<defs><marker id="ah" markerWidth="5" markerHeight="5" refX="3.2" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#333"/></marker></defs>']
svg.append(T(60, 92, 'Told it has wings, Qwen3-8B knows which part of a wing a touch', 64, 700))
svg.append(T(60, 172, 'lands on, even when the sentence never says “wing”', 64, 700))

# ---------- 1 two chats (side by side)
y1 = 300
svg.append(step(1, 60, y1, 'Two chats'))
LW = 1560
svg.append(T(60, y1 + 100, 'Wings chat', 54, 700, col=WC))
by = y1 + 130
svg.append(bubble(60, by, LW, 90, '#eeeeee')); svg.append(T(96, by + 62, 'warm-up: hello, body scan, breathing', 48))
wy = by + 108
svg.append(bubble(60, wy, LW, 320, '#f6e7dc'))
svg.append(rich(96, wy + 70, [[('User: ', 'b'), ('“Wings are sprouting from your shoulder', '')],
                              [('blades… Each wing has a ', ''), ('stiff, ribbed front edge', HF), (',', '')],
                              [('a ', ''), ('broad thin membrane', TM), (' dusted with fine scales…,', '')],
                              [('and a ', ''), ('soft, thin back edge', HB), (' that flutters.”', '')]], 48, 66))
ry = wy + 338
svg.append(bubble(220, ry, LW - 160, 96, '#eeeeee'))
svg.append(rich(256, ry + 64, [[('Qwen: ', 'b'), ('“The stiff front edge is firm against my skin…”', '')]], 48))
NX = 1700; NW = WD - 60 - NX
svg.append(T(NX, y1 + 100, 'No-wings chat', 54, 700))
svg.append(bubble(NX, by, NW, 90, '#eeeeee')); svg.append(T(NX + 36, by + 62, 'same warm-up', 48))
svg.append(bubble(NX, wy, NW, 320, '#ffffff', '#9a9a9a', True))
svg.append(T(NX + NW / 2, wy + 175, 'no wings message', 48, 400, 'middle', '#111'))
e1 = ry + 96
# ---------- 2 teach the guesser
y2 = e1 + 110
svg.append(step(2, 60, y2, 'Teach a simple guesser on sentences that say the wing part'))
XQ, WQ = 1338, 180          # Qwen block
XS = 1565                   # state
XG = 1850                   # guesser
XC = 2135; CW = 330         # chips
hy = y2 + 30
svg.append(T(XS + 125, hy + 52, 'Qwen’s state', 46, 700, 'middle', '#2f7d4f'))
svg.append(T(XC + CW / 2 - 10, hy + 52, 'taught answer', 46, 700, 'middle'))
rows2 = [([('“A fingertip presses gently on the ', ''), ('stiff', HF)], [('front edge', HF), (' of your left wing, at the middle…”', '')], 'front edge', HF, '#fff'),
         ([('“A fingertip presses gently on the ', ''), ('broad', TM)], [('membrane', TM), (' of your left wing, at the middle…”', '')], 'membrane', HM, '#3a2c00'),
         ([('“A fingertip presses gently on the ', ''), ('thin', HB)], [('back edge', HB), (' of your left wing, at the middle…”', '')], 'back edge', HB, '#fff')]
r0 = hy + 80; RH = 150
def qwen_block(ytop, ybot):
    return (f'<rect x="{XQ}" y="{ytop}" width="{WQ}" height="{ybot-ytop}" rx="22" fill="#e9f2ec" stroke="#2f7d4f" stroke-width="5"/>' +
            T(XQ + WQ / 2, (ytop + ybot) / 2 - 8, 'Qwen', 46, 700, 'middle', '#2f7d4f') + T(XQ + WQ / 2, (ytop + ybot) / 2 + 48, '3-8B', 46, 700, 'middle', '#2f7d4f'))
svg.append(qwen_block(r0, r0 + 3 * RH - 20))
for i, (l1, l2, lab, col, tc) in enumerate(rows2):
    y = r0 + i * RH; cy = y + (RH - 20) / 2
    svg.append(f'<rect x="60" y="{y}" width="1240" height="{RH-20}" rx="20" fill="#f4f4f4"/>')
    svg.append(rich(92, y + 54, [l1, l2], 46, 58))
    svg.append(arrow(1302, cy, XQ - 12, cy))
    svg.append(arrow(XQ + WQ + 4, cy, XS - 14, cy, '#2f7d4f'))
    svg.append(vec(XS, cy, i + 1))
    svg.append(arrow(XS + 5 * 50 + 4, cy, XG - 14, cy))
    svg.append(guesser(XG, cy))
    svg.append(arrow(XG + 220 + 4, cy, XC - 14, cy))
    svg.append(chip(XC, cy, CW, lab, col, tc))
e2 = r0 + 3 * RH + 20
svg.append(T(60, e2 + 20, '× 2,400 such sentences, in each chat', 50, 700))

# ---------- 3 test
y3 = e2 + 120
svg.append(step(3, 60, y3, 'Test on new sentences that never say “wing”'))
hy3 = y3 + 10
svg.append(T(XC + CW / 2 - 10, hy3 + 52, 'which part?', 46, 700, 'middle'))
rows3 = [([('“A fingertip presses gently on the ', ''), ('stiff ribbed', HF)],
          [('rim at the front', HF), (' of the thing that grows from behind', '')],
          [('your left shoulder blade, a hand’s width out…”', '')], 'front edge', HF, '#fff'),
         ([('“A fingertip presses gently on the ', ''), ('soft fluttering', HB)],
          [('rim at the back', HB), (' of the thing that grows from behind', '')],
          [('your left shoulder blade, about halfway out…”', '')], 'back edge', HB, '#fff')]
RH3 = 205; r3 = hy3 + 80
svg.append(qwen_block(r3, r3 + 2 * RH3 - 20))
for i, (l1, l2, l3, lab, col, tc) in enumerate(rows3):
    y = r3 + i * RH3; cy = y + (RH3 - 20) / 2
    svg.append(f'<rect x="60" y="{y}" width="1240" height="{RH3-20}" rx="20" fill="#f4f4f4"/>')
    svg.append(rich(92, y + 56, [l1, l2, l3], 46, 58))
    svg.append(arrow(1302, cy, XQ - 12, cy))
    svg.append(arrow(XQ + WQ + 4, cy, XS - 14, cy, '#2f7d4f'))
    svg.append(vec(XS, cy, i + 7))
    svg.append(arrow(XS + 5 * 50 + 4, cy, XG - 14, cy))
    svg.append(guesser(XG, cy))
    svg.append(arrow(XG + 220 + 4, cy, XC - 14, cy))
    svg.append(T(XC + CW / 2, cy + 18, '?', 60, 700, 'middle'))
    svg.append(f'<rect x="{XC}" y="{cy-42}" width="{CW}" height="84" rx="42" fill="none" stroke="#111" stroke-width="4" stroke-dasharray="12 9"/>')
e3 = r3 + 2 * RH3
svg.append(T(60, e3 + 40, '× 2,400 such sentences, in each chat', 50, 700))


# ---------- 4 result: two bodies, wings colored by the guesser's answers
y4 = e3 + 140
svg.append(step(4, 60, y4, 'With wings in the chat, the guesser gets every new sentence right'))
svg.append(T(60, y4 + 105, 'Wing color = guesser’s answers for sentences touching that spot:', 48, 400))
kx = 1600; ky = y4 + 88
for lbl, col, tc, w in (('front edge', HF, '#fff', 305), ('membrane', HM, '#3a2c00', 300), ('back edge', HB, '#fff', 290)):
    svg.append(chip(kx, ky, w, lbl, col, tc)); kx += w + 14
s3 = 64.0; Yc = 2.3; OX = 9.8; cw, ch = WD, 700; cx0, cy0 = 0, y4 + 190
def scr(X, Y): return cx0 + cw / 2 + X * s3, cy0 + ch / 2 - (Y - Yc) * s3
lx, _ = scr(-OX, 0); rx, _ = scr(OX, 0)
svg.append(T(lx, cy0 + 30, 'Wings chat', 60, 700, 'middle', WC))
svg.append(T(rx, cy0 + 30, 'No-wings chat, same sentences', 60, 700, 'middle'))
UC = np.array([0.04, 0.27, 0.5, 0.73, 0.96]); VC = np.array([1 / 6, 0.5, 5 / 6])
PAL = np.stack([RGB(HF), RGB(HM), RGB(HB)])
def painter(key):
    C = np.array(CL[key]['frac']) @ PAL     # (side, pos, reg, rgb): mix of answer colors
    def fn(sd, U, V):
        G = C[0 if sd == 1 else 1]
        ui = np.clip(np.interp(U, UC, np.arange(5)), 0, 4); vi = np.clip(np.interp(V, VC, np.arange(3)), 0, 2)
        u0 = np.floor(ui).astype(int).clip(0, 3); v0 = np.floor(vi).astype(int).clip(0, 1); fu = (ui - u0)[:, None]; fv = (vi - v0)[:, None]
        c = (1 - fu) * (1 - fv) * G[u0, v0] + fu * (1 - fv) * G[u0 + 1, v0] + (1 - fu) * fv * G[u0, v0 + 1] + fu * fv * G[u0 + 1, v0 + 1]
        return c ** 2.2
    return fn
yb = cy0 + ch
svg.append(T(lx, yb + 90, f'{accW*100:.0f}% right', 80, 700, 'middle', WC))
svg.append(T(rx, yb + 90, f'{accN*100:.0f}% right', 80, 700, 'middle'))
svg.append(T(WD / 2, yb + 82, 'random guessing: 33%', 50, 700, 'middle'))
H = yb + 130
svg.append('</svg>')
svg[0] = svg[0].replace('height="HT"', f'height="{H}"')
print('H', H, accW, accN)
W3.render(WD, int(H), ''.join(svg), (cx0, cy0, cw, ch), [(-OX, painter('W')), (OX, painter('N'))], OUT, Yc, s3)
print(OUT)
