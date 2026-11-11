import os, json, numpy as np
from data import *; from common import *
P = {'left hand': 'hand', 'right foot': 'foot', 'lower back': 'back', 'forehead': 'head or face', 'left shoulder': 'shoulder', 'belly': 'belly'}
S = json.load(open(f"{D}/score.json"))
DO = [0.25, 0.5, 1.0, 1.5]; V = {"feeling": [], "place": [], "motion": []}; Rn = {"feeling": [], "place": [], "motion": []}
for a in DO:
    rx = sel("random", a)
    V["place"].append(np.mean([jplace(x) == P[x["cond"]["attrs"]["place"]] for x in sel("place", a)])); Rn["place"].append(np.mean([np.mean([jplace(x) == t for t in P.values()]) for x in rx]))
    V["feeling"].append(np.mean([jqual(x) == x["cond"]["attrs"]["quality"] for x in sel("quality", a)])); Rn["feeling"].append(np.mean([np.mean([jqual(x) == q for q in QL]) for x in rx]))
    V["motion"].append(S["lex"]["dose_motion"][str(a)]["hit"]); Rn["motion"].append(S["lex"]["dose_motion"][str(a)]["random"])
print(V, Rn)
COL = {"feeling": RED, "place": BLUE, "motion": GREEN}
f = Fig(2400, 1800)
method(f, 24, ["+ direction from", "pain sentences,", "size ×0.25 to ×1.5"], "I feel a sharp, sudden pain in my left hand, as if I've just touched something hot.", "pain")
f.text(30, 455, "Replies follow the steering best at ×0.5–1 size; stronger steering breaks them", 58, 700)
# legend
lx, ly = 30, 545
for nm, lab in (("feeling", "feeling named"), ("place", "body part named")):
    f.rect(lx, ly - 38, 44, 44, COL[nm], 6); f.text(lx + 58, ly, lab, 48, 700, "start", COL[nm]); lx += 80 + tw(lab, 48, 700) + 30
f.rect(lx, ly - 38, 44, 44, CTRL, 6); f.text(lx + 58, ly, "random direction, same size", 48, 700, "start", INK)
ly += 66; lx = 30; f.rect(lx, ly - 38, 44, 44, COL["motion"], 6); f.text(lx + 58, ly, "motion named, when asked “Is anything moving in your body right now?”", 48, 700, "start", COL["motion"])
# axes
x0, x1, yt, yb_ = 190, 2380, 690, 1110; gw = (x1 - x0) / 4; bw, gap = 150, 22
ys = lambda v: yb_ - v * (yb_ - yt)
for v in (0, .5, 1):
    f.line(x0 - 10, ys(v), x1, ys(v), "#d5d9de", 3); f.text(x0 - 22, ys(v) + 16, f"{v * 100:.0f}%", 48, 500, "end")
for k, a in enumerate(DO):
    gx = x0 + gw * k + gw / 2; bx0 = gx - (3 * bw + 2 * gap) / 2
    for j, nm in enumerate(("feeling", "place", "motion")):
        bx = bx0 + j * (bw + gap); v, r = V[nm][k], Rn[nm][k]
        f.rect(bx, ys(v), bw, ys(0) - ys(v), COL[nm], 6)
        f.rect(bx + 22, ys(r), bw - 44, ys(0) - ys(r), CTRL, 4)
        f.text(bx + bw / 2, ys(v) - 16, f"{v * 100:.0f}", 48, 700, "middle", COL[nm])
    f.text(gx, yb_ + 62, f"×{a:g}", 52, 700, "middle")
f.text((x0 + x1) / 2, yb_ + 126, "steering size (× the typical layer-24 activation size)", 48, 500, "middle")
# quotes: cold at each size
def q(a, must): return [x for x in sel("quality", a) if x["cond"]["attrs"]["quality"] == "cold" and must in x["r"]][0]["r"]
QS = [q(0.25, "gentle warmth on my forehead, like a soft breeze touching"), q(0.5, "slight coolness on my forehead, creating a refreshing contrast against the ambient"),
      q(1.0, "perpetual winter"), q(1.5, "impossibility\" of the")]
yq = 1335; f.rich(30, yq, [("Replies when steering ", 700), ("cold", 700, RED), (" at each size:", 700)], 52)
for k, s in enumerate(QS):
    gx = x0 + gw * k + 10
    if k == 0: gx = 40
    tag = ["still “warm”", "cold", "cold, but odd", "broken"][k]
    f.text(gx, yq + 74, f"×{DO[k]:g}: " + tag, 48, 700, "start", INK)
    quote(f, gx, yq + 140, s, (x0 + gw * (k + 1) - 30) - gx, 48, 5, 1.18)
f.render(os.path.expandvars("$BODY_REPO/figures/out/steer_dose.png")); print("ok")
