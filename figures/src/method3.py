import os
# method strip: one sentence -> model -> state; repeat for 407 spots; 5 most similar -> cm -> colour
import sys as _s; _s.path.insert(0, os.path.expandvars("$BODY_REPO/hires/v2")); import stim as _S
EX = 38  # inside of left forearm, a hand-width above the wrist
_p2 = [l["phrase"] for l in _S.LOCS]; _k = _p2.index(str(ph[EX]))
def _cells(f, key):
    a = np.load(os.path.expandvars("$BODY_REPO/hires/v2/") + f)[key][_k].astype(np.float32)
    return a / (np.abs(a).max() + 1e-9)
C8, C32 = _cells("red8b_short.npz", "reply|27|all|A"), _cells("red32_short.npz", "reply|53|all|A")
def dcol(v):  # diverging blue-white-red for state values
    v = float(np.clip(v * 2.2, -1, 1)); a = (178, 24, 43) if v > 0 else (33, 102, 172); t = abs(v) ** 0.7
    return "#%02x%02x%02x" % tuple(round(255 + t * (c - 255)) for c in a)
def arrow(x1, y1, x2, y2, w=7):
    import math; L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; hx, hy = x2 - ux * 30, y2 - uy * 30
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="#111" stroke-width="{w}"/>')
    out.append(f'<polygon points="{x2},{y2} {hx - uy*18:.0f},{hy + ux*18:.0f} {hx + uy*18:.0f},{hy - ux*18:.0f}" fill="#111"/>')
W = 2600
# ---- row A ----
for k in (2, 1):
    out.append(f'<rect x="{40+k*16}" y="{30+k*16}" width="1000" height="230" rx="22" fill="#fff" stroke="#9aa3ad" stroke-width="3"/>')
out.append('<rect x="40" y="30" width="1000" height="230" rx="22" fill="#fff" stroke="#111" stroke-width="4"/>')
T(80, 105, "A feather brushes slowly across", 54); T(80, 170, "the inside of your left forearm,", 54, 700); T(80, 235, "a hand-width above the wrist.", 54, 700)
T(40, 340, "+ 25 more touch sentences (pain, itch, ice…)", 50)
arrow(1090, 150, 1175, 150)
out.append('<rect x="1190" y="45" width="440" height="210" rx="26" fill="#1d2b3a"/>')
T(1410, 135, "Qwen3-8B", 62, 700, "middle", "#fff"); T(1410, 215, "Qwen3-32B", 62, 700, "middle", "#fff")
arrow(1645, 150, 1730, 150)
cw = 14.6
for j, (C, lab, n) in enumerate([(C8, "4,096 numbers", 4096), (C32, "5,120 numbers", 5120)]):
    y = 22 + j * 128; ncell = round(44 * n / 4096)
    for c in range(ncell):
        out.append(f'<rect x="{1750 + c*cw:.1f}" y="{y}" width="{cw-2:.1f}" height="58" fill="{dcol(C[c])}"/>')
    out.append(f'<rect x="1750" y="{y}" width="{ncell*cw-2:.1f}" height="58" fill="none" stroke="#111" stroke-width="2"/>')
    T(1750, y + 108, ("Qwen3-8B: " if j == 0 else "Qwen3-32B: ") + lab, 50)
T(1750, 340, "internal state, average of all 26", 50)
# ---- row B ----
B0 = 390; ybB = B0 + 735; scB = 3.95; cxB = 390
_sc = sc; sc = scB
XB = lambda x: cxB + x * scB; YB = lambda y: ybB - y * scB
body(XB, YB, "#e3e7eb"); sc = _sc
nn = d["nn_32B"][EX]; tw = d["tw"][EX]
for i in np.where(front)[0]:
    out.append(f'<circle cx="{XB(PA[i,0]):.1f}" cy="{YB(PA[i,1]):.1f}" r="5.5" fill="#8a949e"/>')
for j in nn:
    out.append(f'<circle cx="{XB(PA[j,0]):.1f}" cy="{YB(PA[j,1]):.1f}" r="13" fill="#fff" stroke="#e8590c" stroke-width="6"/>')
xt, yt = XB(PA[tw,0]), YB(PA[tw,1])
out.append(f'<path d="M{xt-13},{yt-13} L{xt+13},{yt+13} M{xt-13},{yt+13} L{xt+13},{yt-13}" stroke="#111" stroke-width="6"/>')
out.append(f'<circle cx="{XB(PA[EX,0]):.1f}" cy="{YB(PA[EX,1]):.1f}" r="12" fill="#111"/>')
# legend lines, right of the body
lx = 880; ly = B0 + 90
T(lx - 40, ly, "Repeat for all 407 spots on the body", 56, 700)
out.append(f'<circle cx="{lx}" cy="{ly+82}" r="13" fill="#111"/>'); T(lx + 40, ly + 100, "the forearm spot in the sentence", 52)
out.append(f'<circle cx="{lx}" cy="{ly+172}" r="12" fill="#fff" stroke="#e8590c" stroke-width="6"/>'); T(lx + 40, ly + 190, "the 5 spots whose Qwen3-32B states are most similar", 52)
out.append(f'<path d="M{lx-13},{ly+249} L{lx+13},{ly+275} M{lx-13},{ly+275} L{lx+13},{ly+249}" stroke="#111" stroke-width="6"/>'); T(lx + 40, ly + 280, "mirror twin (same place, other side): skipped", 52)
T(lx - 40, ly + 390, "Distance on the body to those 5:", 52, 700)
T(lx - 40, ly + 460, "9, 9, 5, 5, 4 cm", 52)
arrow(lx + 330, ly + 443, lx + 420, ly + 443, 6)
T(lx + 440, ly + 460, "average 6 cm", 52, 700)
arrow(lx + 790, ly + 443, lx + 890, ly + 443, 6)
T(lx + 910, ly + 460, "dot colour", 52, 700)
out.append(f'<circle cx="{lx+1210}" cy="{ly+443}" r="24" fill="{col(6.3)}" stroke="#fff" stroke-width="3"/>')
# colour scale bar
gx0, gx1, gy = lx - 40, lx + 1200, ly + 515
out.append('<defs><linearGradient id="cs">' + "".join(f'<stop offset="{a/40:.3f}" stop-color="#%02x%02x%02x"/>' % c for a, c in STOPS) + '</linearGradient></defs>')
out.append(f'<rect x="{gx0}" y="{gy}" width="{gx1-gx0}" height="40" fill="url(#cs)"/>')
out.append(f'<line x1="{gx0 + (gx1-gx0)*6.3/40:.0f}" y1="{gy-8}" x2="{gx0 + (gx1-gx0)*6.3/40:.0f}" y2="{gy+48}" stroke="#111" stroke-width="5"/>')
for t in (0, 10, 20, 30, 40):
    T(gx0 + (gx1 - gx0) * t / 40, gy + 100, f"{t}" + (" cm or more" if t == 40 else ""), 48, 400, "middle" if t < 40 else "middle")
OFF = ybB + 50
H = OFF + 1880
