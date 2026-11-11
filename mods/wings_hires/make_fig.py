import os
import json, numpy as np, html
import wing3d as W3
O = os.path.expandvars('$BODY_REPO/mods/wings_hires/out8b')
F = json.load(open(f'{O}/figdata.json'))
OUT = os.path.expandvars('$BODY_REPO/figures/out/body_wings_hires.png')
WD, HT = 2600, 2200
STOPS = [(0.0, (0.11, 0.33, 0.66)), (0.5, (0.91, 0.80, 0.42)), (1.0, (0.74, 0.24, 0.09))]
def cmap(x):
    x = np.clip(x, 0, 1); out = np.zeros((len(x), 3))
    for (a, ca), (b, cb) in zip(STOPS[:-1], STOPS[1:]):
        m = (x >= a) & (x <= b); w = ((x[m] - a) / (b - a))[:, None]; out[m] = (1 - w) * np.array(ca) + w * np.array(cb)
    return out
def hexc(c): return '#%02x%02x%02x' % tuple(int(255 * v) for v in c)
UC = np.array([0.04, 0.27, 0.5, 0.73, 0.96]); VC = np.array([1 / 6, 0.5, 5 / 6])
def field(G, side_idx, U, V):
    G = np.array(G)[side_idx]   # (5 pos, 3 reg)
    ui = np.clip(np.interp(U, UC, np.arange(5)), 0, 4); vi = np.clip(np.interp(V, VC, np.arange(3)), 0, 2)
    u0 = np.floor(ui).astype(int).clip(0, 3); v0 = np.floor(vi).astype(int).clip(0, 1); fu = ui - u0; fv = vi - v0
    g = lambda a, b: G[a, b]
    val = (1 - fu) * (1 - fv) * g(u0, v0) + fu * (1 - fv) * g(u0 + 1, v0) + (1 - fu) * fv * g(u0, v0 + 1) + fu * fv * g(u0 + 1, v0 + 1)
    return val
def painter(key):
    G = F[key]['reg_all']
    def fn(sd, U, V):   # sd=+1 is the person's left wing (side index 0)
        return cmap(field(G, 0 if sd == 1 else 1, U, V)) ** 2.2
    return fn
# ---- layout
s = 72.0; Yc = 2.1; cy0 = 1180; ch = 740; cvs = (0, cy0, WD, ch); OX = 9.0
def scr(x, y): return WD / 2 + x * s, cy0 + ch / 2 - (y - Yc) * s
T = lambda x, y, t, sz=50, w=400, anc='start', col='#111': f'<text x="{x}" y="{y}" font-size="{sz}" font-weight="{w}" text-anchor="{anc}" fill="{col}">{html.escape(t)}</text>'
svg = ['<svg width="%d" height="%d" style="position:absolute;left:0;top:0;font-family:Inter, Helvetica Neue, Arial, sans-serif">' % (WD, HT)]
# headline
svg.append(T(70, 95, "Described without the word “wing”, a touch lands on the right part of the wing", 60, 700))
svg.append(T(70, 170, "only in the chat where Qwen3-8B has wings", 60, 700, col='#a5401a'))
# method: two chats
def card(x, y, w, h, title, lines, col):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{col}" />', T(x + 36, y + 70, title, 52, 700)]
    for i, (who, txt) in enumerate(lines):
        out.append(T(x + 36, y + 145 + i * 62, who, 46, 700, col='#555') + T(x + 250, y + 145 + i * 62, txt, 46))
    return ''.join(out)
svg.append(card(70, 230, 1215, 395, "Wings chat", [("User", "Wings are sprouting from your shoulder"), ("", "blades… a stiff, ribbed front edge, a broad"), ("", "membrane, a soft back edge that flutters."), ("Model", "The stiff front edge is firm against my skin…")], '#f3ece2'))
svg.append(card(1315, 230, 1215, 395, "No-wings chat", [("User", "the same warm hello and body scan,"), ("", "no wings"), ("", "")], '#ececec'))
# train / test sentences
y0 = 690
svg.append(T(70, y0, "Learn a read-out from touch sentences that say “wing” (Qwen3-8B layer 24):", 50, 700))
svg.append(T(110, y0 + 70, "“A fingertip presses on the stiff front edge of your left wing, near the base.”", 46))
svg.append(T(70, y0 + 160, "Apply the read-out to the same spots described without the word:", 50, 700))
svg.append(T(110, y0 + 230, "“…the stiff ribbed rim at the front of the thing that grows from behind your", 46))
svg.append(T(110, y0 + 290, "left shoulder blade, a hand’s width out from your back.”", 46))
svg.append(T(110, y0 + 360, "Model, wings chat: “…the stiff, ribbed edge of my left wing…”", 46, 400, col='#a5401a'))
# result labels
lx, _ = scr(-OX, 0); rx, _ = scr(OX, 0)
svg.append(T(lx, cy0 - 10, "Wings chat", 58, 700, 'middle'))
svg.append(T(rx, cy0 - 10, "No-wings chat, same sentences", 58, 700, 'middle'))
accW, accN = F['W']['reg_acc'], F['N']['reg_acc']
yb = cy0 + ch + 10
svg.append(T(lx, yb + 60, f"front edge, membrane, back edge read correctly: {accW*100:.0f}%", 46, 400, 'middle'))
svg.append(T(rx, yb + 60, f"read correctly: {accN*100:.0f}%   (guessing: 33%)", 46, 400, 'middle'))
# colour key: a flat wing strip with three bands
kx, ky, kw, kh = 860, yb + 105, 880, 52
grad = ''.join(f'<stop offset="{a}" stop-color="{hexc(c)}"/>' for a, c in STOPS)
svg.append(f'<defs><linearGradient id="g">{grad}</linearGradient></defs><rect x="{kx}" y="{ky}" width="{kw}" height="{kh}" rx="10" fill="url(#g)"/>')
svg.append(T(kx - 24, ky + 42, "read-out says: front edge", 46, 400, 'end'))
svg.append(T(kx + kw / 2, ky + kh + 52, "membrane", 46, 400, 'middle'))
svg.append(T(kx + kw + 24, ky + 42, "back edge", 46, 400, 'start'))
svg.append('</svg>')
W3.render(WD, HT, ''.join(svg), cvs, [(-OX, painter('W')), (OX, painter('N'))], OUT, Yc, s)
print(OUT)
