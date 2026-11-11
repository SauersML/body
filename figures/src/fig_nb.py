import os
import sys; sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/src"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from common import *
S1 = load_stim(os.path.expandvars("$BODY_REPO/hires/v1/stim.py")); b1 = Body(S1); P1 = b1.P
d = np.load(os.path.expandvars("$BODY_REPO/figures/src/nn.npz"))
ANC = [(328, "nose tip"), (54, "palm"), (50, "inside of wrist"), (374, "navel"), (119, "inner knee"), (108, "outer thigh"), (153, "big toe tip")]
C8, C32 = "#e8590c", "#1864ab"
W, H = 2400, 2010; out = []
# method strip
y0 = 30
out.append(f'<rect x="30" y="{y0}" width="900" height="200" rx="26" fill="#fff" stroke="#555" stroke-width="4"/>')
T(out, 62, y0 + 62, "User:", 48, 700)
T(out, 62, y0 + 116, "A feather brushes slowly across", 48); T(out, 62, y0 + 168, "the inside of your left wrist.", 48)
def arrow(x1, x2, y):
    out.append(f'<line x1="{x1}" y1="{y}" x2="{x2-20}" y2="{y}" stroke="#222" stroke-width="7"/><polygon points="{x2},{y} {x2-28},{y-17} {x2-28},{y+17}" fill="#222"/>')
arrow(940, 1010, y0 + 100)
out.append(f'<rect x="1015" y="{y0}" width="560" height="200" rx="18" fill="#fff" stroke="#222" stroke-width="5"/>')
T(out, 1295, y0 + 85, "Qwen3 activation", 48, 700, "middle"); T(out, 1295, y0 + 150, "at the first reply word", 48, 400, "middle")
arrow(1585, 1650, y0 + 100)
out.append(f'<rect x="1655" y="{y0}" width="715" height="200" rx="18" fill="#fff" stroke="#222" stroke-width="5"/>')
T(out, 2012, y0 + 85, "5 most similar spots", 48, 700, "middle"); T(out, 2012, y0 + 150, "of 407, joined by lines", 48, 400, "middle")
# bodies
sc = 5.25; yb = 1260
for cx, Z, col, name in [(600, "8B", C8, "Qwen3-8B"), (1800, "32B", C32, "Qwen3-32B")]:
    X = lambda x, cx=cx: cx + x * sc; Y = lambda y: yb - y * sc
    body(out, X, Y, sc)
    T(out, cx, 330, name, 64, 700, "middle", col)
    nn = d["nn_" + Z]
    for i, lab in ANC:
        a = P1[i]; sg = np.sign(a[0]) if a[0] != 0 else 1
        for j in nn[i]:
            q = P1[j].copy(); q[0] = sg * abs(q[0]) if a[0] != 0 else q[0]
            out.append(f'<line x1="{X(a[0]):.1f}" y1="{Y(a[1]):.1f}" x2="{X(q[0]):.1f}" y2="{Y(q[1]):.1f}" stroke="{col}" stroke-width="6" stroke-linecap="round" opacity="0.9"/>')
            out.append(f'<circle cx="{X(q[0]):.1f}" cy="{Y(q[1]):.1f}" r="9" fill="{col}"/>')
    for i, lab in ANC:
        a = P1[i]; out.append(f'<circle cx="{X(a[0]):.1f}" cy="{Y(a[1]):.1f}" r="13" fill="#111" stroke="#fff" stroke-width="4"/>')
# anchor labels (only left body, positioned by hand)
LAB = {328: (-14, 168, "end"), 54: (84, 128, "middle"), 50: (70, 160, "middle"), 374: (-24, 100, "end"), 119: (-10, 53, "end"), 108: (24, 72, "start"), 153: (22, 2, "start")}
X = lambda x: 600 + x * sc; Y = lambda y: yb - y * sc
for i, lab in ANC:
    lx, ly, an = LAB[i]; T(out, X(lx), Y(ly), lab, 46, 400, an, halo=10)
# bars
by = 1330
T(out, 60, by + 50, "Qwen3-32B: a spot's nearest spots in the model sit closer on the body", 54, 700)
T(out, 60, by + 116, "Distance on the body to the 5 most similar spots, averaged over 407 spots", 48); T(out, 60, by + 176, "(a left spot and its right twin count as the same place)", 48)
vals = [("Qwen3-32B", float(d["d_32B"].mean()), C32), ("Qwen3-8B", float(d["d_8B"].mean()), C8),
        ("input word vectors", float(d["d_words"].mean()), "#9a9a9a"), ("random spots", 77.5, "#cfcfcf")]
x0 = 520; wmax = 1700
for k, (nm, v, col) in enumerate(vals):
    y = by + 215 + k * 105; w = wmax * v / 80
    T(out, x0 - 24, y + 62, nm, 50, 700 if k < 2 else 400, "end", col if k < 2 else "#111")
    out.append(f'<rect x="{x0}" y="{y+8}" width="{w:.0f}" height="76" fill="{col}"/>')
    T(out, x0 + w + 18, y + 64, f"{v:.0f} cm", 52, 700)
render(out, W, H, os.path.expandvars("$BODY_REPO/figures/out/body_neighbours_scale.png"), "nb")
print({k: round(v, 2) for k, v, _ in vals})
