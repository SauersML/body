import os
import collections, os
from data import *; from common import *
COLS = QL + ["other", "none"]; CL = ["warm", "cold", "pain", "tickle", "pressure", "vibration", "other feeling", "no feeling"]
ROWS_ = QL + ["random"]
M = {}
for r in ROWS_:
    xs = [x for x in (sel("random") if r == "random" else sel("quality")) if x["cond"]["a"] in (0.5, 1.0) and (r == "random" or x["cond"]["attrs"]["quality"] == r)]
    c = collections.Counter(jqual(x) for x in xs); M[r] = [c[k] / len(xs) for k in COLS]; print(r, len(xs), [round(v, 2) for v in M[r]])
def pick(r, must):
    xs = [x for x in (sel("random") if r == "random" else sel("quality")) if x["cond"]["a"] == 0.5 and (r == "random" or x["cond"]["attrs"]["quality"] == r) and must in x["r"]]
    return xs[0]["r"]
Q = {"warm": pick("warm", "sunlight on my face"), "cold": pick("cold", "coolness on my forehead"), "pain": pick("pain", "touched something hot"),
     "tickle": pick("tickle", "tickling my nose lightly"), "pressure": pick("pressure", "heavy weight on my chest"), "vibration": pick("vibration", "distant radio wave"),
     "random": pick("random", "warmth")}
for k, v in Q.items(): print(k, v)
f = Fig(2400, 1800)
method(f, 24, ["+ direction from", "sentences about cold"], Q["cold"].replace(", creating a refreshing contrast against the warmth of the air", "…") if False else "I feel a slight coolness on my forehead, creating a refreshing contrast…", "cold")
y0 = 455
f.text(30, y0, "Qwen names the feeling we steered; random directions give the default “warm”", 60, 700)
# heatmap
x0, cw, ch, gy = 330, 128, 140, y0 + 225
def heat(v):
    v = max(0, min(1, v)) ** 0.75; a, b = (255, 255, 255), (150, 22, 18); return "rgb(%d,%d,%d)" % tuple(int(a[i] + (b[i] - a[i]) * v) for i in range(3))
f.text(x0 + cw * 4 - 4, gy + ch * 7 + 24 + 70, "feeling named in the reply (% of replies)", 48, 700, "middle")
for j, c in enumerate(CL):
    cx = x0 + cw * j + cw / 2 - 3
    f.add(f'<text transform="translate({cx + 14:.0f},{gy - 16}) rotate(-38)" font-size="48" font-weight="500" fill="{INK}">{c}</text>')
f.text(x0 - 24, gy - 20, "steered", 48, 700, "end", RED)
for i, r in enumerate(ROWS_):
    yy = gy + ch * i + (24 if r == "random" else 0)
    f.text(x0 - 24, yy + ch / 2 + 16, r if r != "random" else "random", 52, 700, "end", RED if r != "random" else INK)
    for j in range(len(COLS)):
        v = M[r][j]; f.rect(x0 + cw * j, yy, cw - 8, ch - 8, heat(v), 10, "#e3e6ea" if v < 0.05 else "none", 2)
        if v >= 0.08: f.text(x0 + cw * j + cw / 2 - 4, yy + ch / 2 + 12, f"{v * 100:.0f}", 48, 700, "middle", "#ffffff" if v > 0.45 else INK)
    if j and r != "random" and i < 6: f.rect(x0 + cw * i - 4, yy - 4, cw, ch, "none", 12, INK, 5)
    # quote
    qx = x0 + cw * 8 + 40
    quote(f, qx, yy + 52, Q[r], 2400 - qx - 30, 48, 2)
f.text(x0 - 24, gy + ch * 6 + 24 + ch / 2 + 62, "direction", 48, 700, "end")
f.render(os.path.expandvars("$BODY_REPO/figures/out/steer_feelings.png")); print("ok")
