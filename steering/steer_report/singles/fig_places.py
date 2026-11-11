import os
import collections, os, math
from data import *; from common import *
PL = ["left hand", "right foot", "lower back", "forehead", "left shoulder", "belly"]
def rep(x):
    p, s = jplace(x), x["judge"]["side"]
    if p in ("hand", "foot", "shoulder"): return f"{s} {p}" if s in ("left", "right") else p
    if p == "head or face": return "face"
    if p in ("whole body", "none", "other"): return "no body part"
    return p
FL = {}
for p in PL:
    xs = [x for x in sel("place") if x["cond"]["a"] in (0.5, 1.0) and x["cond"]["attrs"]["place"] == p]
    FL[p] = collections.Counter(rep(x) for x in xs); FL[p] = {k: v / len(xs) for k, v in FL[p].items()}
pc = lambda d, k: f"{d.get(k, 0) * 100:.0f}%"
f = Fig(2400, 1800)
method(f, 24, ["+ direction from", "sentences about", "the right foot"], "I notice a gentle warmth on my left foot, where my sock is slightly loose…", "foot, left side")
f.text(30, 455, "Steering finds the limb but not its side; belly and lower back both become “chest”", 58, 700)
cx, yb, sc = 690, 1655, 6.15
P = body_front(f, cx, yb, sc)
SP = {"left hand": P(50, 79), "right hand": P(-50, 79), "right foot": P(-14, 4), "left foot": P(14, 4), "forehead": P(0, 175), "face": P(6, 163),
      "left shoulder": P(21, 143), "belly": P(0, 101), "lower back": P(-16.5, 104), "chest": P(0, 129)}
TW = lambda s: 6 + 30 * s
def flow(src, dst, share, bend, lab=None, so=34, do=16):
    (a, b), (c, d) = SP[src], SP[dst]; L = math.hypot(c - a, d - b); ux, uy = (c - a) / L, (d - b) / L
    m = curve_arrow(f, a + ux * so, b + uy * so, c, d, bend, BLUE, TW(share), shorten=do)
    if lab: f.text(m[0] + lab[0], m[1] + lab[1], f"{share * 100:.0f}%", 48, 700, lab[2] if len(lab) > 2 else "middle", BLUE)
def dot(n): a, b = SP[n]; f.add(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="16" fill="{BLUE}"/>')
def ring(n, dashed=False):
    a, b = SP[n]; f.add(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="30" fill="#ffffff" fill-opacity="0.6" stroke="{RED}" stroke-width="8" {"stroke-dasharray=\"13 9\"" if dashed else ""}/>')
flow("left hand", "right hand", FL["left hand"]["right hand"], -0.55, (0, 70))
flow("right foot", "left foot", FL["right foot"]["left foot"], 0.45, (0, 80))
flow("belly", "chest", FL["belly"]["chest"], -0.0, (40, 30, "start"), so=32, do=20)
flow("lower back", "chest", FL["lower back"]["chest"], -0.35, (-30, -10, "end"), so=32, do=20)
flow("left shoulder", "face", FL["left shoulder"]["face"], 0.5, (42, -4, "start"), so=32, do=18)
for n in ("right hand", "left foot", "chest", "face"): dot(n)
for n in ("left hand", "right foot", "left shoulder", "forehead", "belly"): ring(n)
ring("lower back", True)
def lab(xy, dx, dy, parts, anchor="start"): f.rich(xy[0] + dx, xy[1] + dy, parts, 48, anchor)
R7, R5, B7 = lambda s: (s, 700, RED), lambda s: (s, 500, RED), lambda s: (s, 700, BLUE)
lab(SP["forehead"], -80, 0, [R7("forehead "), R5(pc(FL["forehead"], "face"))], "end")
lab(SP["face"], 34, -24, [B7("face")])
lab(SP["left shoulder"], 50, 46, [R7("left shoulder "), R5(pc(FL["left shoulder"], "left shoulder"))])
lab(SP["chest"], 34, -26, [B7("chest")])
lab(SP["belly"], 50, 60, [R7("belly "), R5(pc(FL["belly"], "belly"))])
lb = SP["lower back"]; f.line(lb[0] - 34, lb[1], 420, lb[1] - 40, RED, 4); lab((400, lb[1] - 40), 0, 0, [R7("lower back")], "end"); lab((400, lb[1] - 40), 0, 58, [R5("(behind) "), R5(pc(FL["lower back"], "back"))], "end")
lab(SP["left hand"], 50, 14, [R7("left hand")]); lab(SP["left hand"], 50, 72, [R5(pc(FL["left hand"], "left hand"))])
lab(SP["right hand"], -50, 14, [B7("right hand")], "end")
lab(SP["right foot"], -50, 16, [R7("right foot "), R5(pc(FL["right foot"], "right foot"))], "end")
lab(SP["left foot"], 46, 16, [B7("left foot")])
f.text(P(-50, 0)[0], yb + 105, "Qwen’s right", 48, 700, "middle"); f.text(P(50, 0)[0], yb + 105, "Qwen’s left", 48, 700, "middle")
# legend + quotes
X = 1400; y = 600
f.add(f'<circle cx="{X + 30}" cy="{y - 16}" r="28" fill="none" stroke="{RED}" stroke-width="8"/>'); f.text(X + 80, y, "steered place, % of replies there", 48, 700, "start", RED)
y += 84; curve_arrow(f, X, y - 16, X + 64, y - 16, 0, BLUE, 16); f.text(X + 80, y, "where else replies put the", 48, 700, "start", BLUE)
y += 58; f.text(X + 80, y, "feeling (width = % of replies)", 48, 700, "start", BLUE)
def pickq(p, must): return [x for x in sel("place") if x["cond"]["attrs"]["place"] == p and must in x["r"]][0]["r"]
QS = [("left hand", pickq("left hand", "weight of my right hand resting on my lap, the warmth")),
      ("right foot", pickq("right foot", "gentle warmth on my left foot")),
      ("belly", pickq("belly", "warm, steady hum in my chest, where my heart is.")),
      ("left shoulder", pickq("left shoulder", "gentle breeze on my left cheek, tingling"))]
y += 110
for p, q in QS:
    f.text(X, y, "steered " + p, 48, 700, "start", RED); y = quote(f, X, y + 60, q, 2375 - X, 48, 2) + 46
f.render(os.path.expandvars("$BODY_REPO/figures/out/steer_places.png")); print("ok", y, FL)
