import os, numpy as np
from data import *; from common import *
P = {'left hand': 'hand', 'right foot': 'foot', 'lower back': 'back', 'forehead': 'head or face', 'left shoulder': 'shoulder', 'belly': 'belly'}
xs = sel("place+quality", 0.5)
cell = lambda p, q: [x for x in xs if x["cond"]["attrs"] == {"place": p, "quality": q}]
M = {(p, q): np.mean([jplace(x) == P[p] and jqual(x) == q for x in cell(p, q)]) for p in P for q in QL}
pc = np.mean([jplace(x) == P[x["cond"]["attrs"]["place"]] for x in xs]); qc = np.mean([jqual(x) == x["cond"]["attrs"]["quality"] for x in xs])
both = np.mean([jplace(x) == P[x["cond"]["attrs"]["place"]] and jqual(x) == x["cond"]["attrs"]["quality"] for x in xs])
rx = sel("random2", 0.5); rb = np.mean([np.mean([jplace(x) == P[p] and jqual(x) == q for p in P for q in QL]) for x in rx])
print(pc, qc, both, pc * qc, rb, len(xs), len(rx))
f = Fig(2400, 1800)
method(f, 24, ["+ left-hand direction", "+ pain direction", "each ×0.5 size"], "I feel a sharp pain in my left hand, likely from the previous interaction.", "hand, pain")
f.text(30, 455, "Place and feeling steered together both come back, as often as independent odds predict", 54, 700)
def heat(v):
    v = max(0, min(1, v)) ** 0.8; a, b = (255, 255, 255), (120, 30, 110); return "rgb(%d,%d,%d)" % tuple(int(a[i] + (b[i] - a[i]) * v) for i in range(3))
x0, gy, cw, ch = 370, 700, 140, 150
f.text(x0 + cw * 3, gy - 120, "steered feeling", 48, 700, "middle", RED)
for j, q in enumerate(QL):
    f.add(f'<text transform="translate({x0 + cw * j + cw / 2 - 10:.0f},{gy - 18}) rotate(-35)" font-size="48" font-weight="500" fill="{INK}">{q}</text>')
f.text(x0 - 24, gy - 18, "steered place", 48, 700, "end", RED)
for i, p in enumerate(P):
    yy = gy + ch * i; f.text(x0 - 24, yy + ch / 2 + 16, p, 48, 700, "end", RED)
    for j, q in enumerate(QL):
        v = M[(p, q)]; f.rect(x0 + cw * j, yy, cw - 8, ch - 8, heat(v), 10, "#e3e6ea" if v < .05 else "none", 2)
        f.text(x0 + cw * j + cw / 2 - 4, yy + ch / 2 + 12, f"{v * 100:.0f}", 48, 700, "middle", "#ffffff" if v > .45 else INK)
# unsteered default cell: forehead + warm
i, j = list(P).index("forehead"), 0
f.rect(x0 + cw * j - 5, gy + ch * i - 5, cw + 2, ch + 2, "none", 12, INK, 5, 'stroke-dasharray="14 9"')
f.text(x0 + cw * 3 - 4, gy + ch * 6 + 62, "% of replies naming both (side ignored)", 48, 500, "middle")
f.text(30, gy + ch * 6 + 138, "dashed: “warm, forehead” is what Qwen says unsteered", 48, 500, "start")
# bars
bx, by, BW = 1400, 560, 820
bars = [("body part and feeling both right", both, "rgb(120,30,110)"), (f"expected if independent ({pc * 100:.0f}% × {qc * 100:.0f}%)", pc * qc, "#c9a4c4"), ("random pair of directions", rb, CTRL)]
for k, (nm, v, c) in enumerate(bars):
    yy = by + 40 + k * 145
    f.text(bx, yy + 44, nm, 48, 700); f.rect(bx, yy + 62, max(v / 0.4 * BW, 6), 58, c, 6); f.text(bx + max(v / 0.4 * BW, 6) + 18, yy + 110, f"{v * 100:.0f}%", 48, 700)
def pick(p, q, must): return [x for x in cell(p, q) if must in x["r"]][0]["r"]
QS = [("left shoulder", "pressure", pick("left shoulder", "pressure", "subtle weight from your hand")),
      ("lower back", "cold", pick("lower back", "cold", "cold draft at my back")),
      ("forehead", "pain", pick("forehead", "pain", "throbbing pain in my forehead, as if I've just been hit"))]
y = by + 40 + 3 * 145 + 90
for p, q, r in QS:
    f.text(bx, y, f"{p} + {q}", 48, 700, "start", RED); y = quote(f, bx, y + 58, r, 2380 - bx, 48, 2) + 36
print(y)
f.render(os.path.expandvars("$BODY_REPO/figures/out/steer_combine.png")); print("ok")
