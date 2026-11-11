import os
import sys, os, json, re, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
plt.rcParams.update({"font.family": "Avenir Next", "text.color": "#1D1D1F"})
K = "#1D1D1F"; RED = "#D2432F"; CREAM = "#FAF0E6"; CARD = "#F4F4F6"; PURP = "#7B3FE4"
FS = 33  # pt at dpi 100 -> ~46 px
W, Hh = 26, 24
fig = plt.figure(figsize=(W, Hh), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2600); ax.set_ylim(0, 2400); ax.axis("off")
def T(x, y, s, size=FS, **k): k.setdefault("ha", "left"); k.setdefault("va", "center"); ax.text(x, y, s, fontsize=size, color=k.pop("color", K), **k)
def box(x, y, w, h, fc, ec="none", r=28, lw=0, z=1): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

# ---------------- data: internal map at layer 25 (reply start), left/right averaged ----------------
D = np.load(os.path.expandvars("$BODY_REPO/figures/bodymap/DT.npy")); types = json.load(open(os.path.expandvars("$BODY_REPO/figures/bodymap/types.json")))
def mds(D, k=3):
    D = np.maximum(D, 0); n = len(D); J = np.eye(n) - 1 / n; B = -.5 * J @ D @ J; w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    return v[:, o[:k]] * np.sqrt(np.maximum(w[o[:k]], 0))
Y = mds(D)
# colour = position in the model's 3-D internal map
from scipy.stats import rankdata
C = np.stack([(rankdata(Y[:, i]) - 1) / (len(Y) - 1) for i in range(3)], 1)
C = 0.18 + 0.67 * C
col = {t: C[i][[0, 2, 1]] for i, t in enumerate(types)}
Dm = D.copy(); np.fill_diagonal(Dm, np.inf); nn = {t: types[int(np.argmin(Dm[i]))] for i, t in enumerate(types)}

# ---------------- title ----------------
T(1300, 2345, "Qwen3-8B has a map of the body inside, and steering the map moves where the model feels touch", size=38, weight="bold", ha="center")

# ---------------- method strip ----------------
box(40, 1810, 2520, 470, CARD, r=36)
bub = [("user", "Hi! I'm glad to talk with you today."), ("model", "Hello! I'm glad to talk with you too."),
       ("user", "…imagine you have a human body…"), ("model", "Absolutely, I'd love to try!")]
yy = 2225
for who, s_ in bub:
    if who == "user": box(80, yy - 29, 740, 58, "white", r=24); T(100, yy, s_, size=29)
    else: box(180, yy - 29, 640, 58, "#E7DDF7", r=24); T(200, yy, s_, size=29)
    yy -= 72
box(80, 1832, 860, 118, CREAM, ec=RED, lw=4, r=26)
T(105, 1891, "A feather brushes slowly across\nyour left wrist.", size=31, weight="bold", linespacing=1.05)
ax.annotate("", xy=(1075, 1980), xytext=(960, 1890), arrowprops=dict(arrowstyle="-|>", lw=5, color=K, mutation_scale=40))
x0 = 1090; T(x0 + 150, 2238, "Qwen3-8B", size=36, weight="bold", ha="center")
ly = None
for i in range(36):
    yb = 1840 + i * 10.6; hl = (i == 24)
    ax.add_patch(Rectangle((x0, yb), 300, 8 if not hl else 10, fc=RED if hl else "#C9C9D1", ec="none"))
    if hl: ly = yb + 5
T(x0 + 330, ly + 45, "layer 25 of 36", size=33, color=RED, weight="bold")
ax.annotate("", xy=(1830, ly), xytext=(x0 + 310, ly), arrowprops=dict(arrowstyle="-|>", lw=5, color=RED, mutation_scale=40))
rng = np.random.default_rng(3)
for j in range(22):
    v = rng.uniform(0.15, 0.95); ax.add_patch(Rectangle((1845 + j * 30, ly - 35), 26, 70, fc=plt.cm.Reds(v), ec="none"))
T(2170, ly + 75, "internal state", size=33, ha="center", weight="bold")
T(2170, ly - 120, "for 54 body places\n× 8 kinds of touch", size=33, ha="center", va="center", linespacing=1.1)

# ---------------- panel 2: body ----------------
P2x, P2y = 40, 60
box(P2x, 130, 1290, 1650, "white", ec="#D8D8DE", lw=3, r=36)
T(80, 1690, "Inside the model, nearby places on\nthe body get similar internal states", size=36, weight="bold", va="center", linespacing=1.1)
T(80, 1565, "Same colour = close together at layer 25", size=31)
SIL = "#E9E6EE"
def body(cx, base, s, back=False):
    """draw a simple outline figure; returns dict of point coords (side: +1 = person's left)"""
    X = lambda x: cx + x * s; Yc = lambda y: base + y * s
    def cap(p, q, w):
        ax.plot([X(p[0]), X(q[0])], [Yc(p[1]), Yc(q[1])], lw=w * s, color=SIL, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((X(0), Yc(89)), 7.5 * s, fc=SIL, ec="none", zorder=2))
    cap((0, 84), (0, 78), 7)
    ax.add_patch(Polygon([(X(-13), Yc(78)), (X(13), Yc(78)), (X(10.5), Yc(50)), (X(-10.5), Yc(50))], fc=SIL, ec="none", zorder=2, joinstyle="round"))
    ax.add_patch(Circle((X(0), Yc(47)), 10.5 * s, fc=SIL, ec="none", zorder=2))
    pts = {}
    # front view: person's left is on viewer's right (sgn=+1 on screen right). back view: person's left on screen left.
    for side in (1, -1):
        g = side if not back else -side  # screen direction
        cap((12 * g, 76), (17 * g, 61), 5.5); cap((17 * g, 61), (20 * g, 45), 5)
        cap((6 * g, 46), (7 * g, 26), 8); cap((7 * g, 26), (7 * g, 6), 6.5); cap((7 * g, 4), (9 * g, 1), 5)
        ax.add_patch(Circle((X(21.3 * g), Yc(38.5)), 4.8 * s, fc=SIL, ec="none", zorder=2))
        for k, (fx, fy) in enumerate([(27.5, 42), (25.8, 30.5), (22, 29), (18.6, 30), (16.2, 33)]):
            cap((21.3 * g, 38.5), (fx * g, fy), 2.2)
        sfx = "L" if side == 1 else "R"
        P = lambda x, y: (X(x * g), Yc(y))
        if not back:
            pts.update({f"thumb{sfx}": P(27.5, 42), f"index{sfx}": P(25.8, 30.5), f"middle{sfx}": P(22, 29), f"ring{sfx}": P(18.6, 30),
                        f"little{sfx}": P(16.2, 33), f"palm{sfx}": P(21.3, 38.5), f"forearm{sfx}": P(18.7, 52), f"upperarm{sfx}": P(14.8, 68),
                        f"shoulder{sfx}": P(11.5, 76), f"hip{sfx}": P(9.5, 48), f"thigh{sfx}": P(6.5, 37), f"knee{sfx}": P(7, 26),
                        f"shin{sfx}": P(7, 16), f"ankle{sfx}": P(7.3, 6), f"toes{sfx}": P(9.3, 1.2)})
            pts[f"cheek{sfx}"] = P(4.3, 88.3); pts[f"ear{sfx}"] = P(8.2, 90)
        else:
            pts.update({f"backhand{sfx}": P(21, 39.5), f"wrist{sfx}": P(20, 45), f"elbow{sfx}": P(17, 61), f"heel{sfx}": P(7, 3.5),
                        f"sole{sfx}": P(9.5, -3)})
    if not back:
        pts.update({"forehead": (X(0), Yc(93.5)), "lips": (X(-1.6), Yc(84.8)), "tongue": (X(1.6), Yc(84.8)), "neckF": (X(0), Yc(80)), "chest": (X(0), Yc(69)), "belly": (X(0), Yc(55))})
    else:
        pts.update({"crown": (X(0), Yc(96)), "neckB": (X(0), Yc(81)), "upback": (X(0), Yc(70)), "lowback": (X(0), Yc(55))})
    return pts
s = 11.0
pf = body(348, 300, s); pb = body(985, 300, s, back=True)
T(348, 205, "front", size=33, ha="center", weight="bold"); T(985, 205, "back", size=33, ha="center", weight="bold")
pts = {**pf, **pb}
keytype = {l["key"]: l["type"] for l in S.LOCS}
for k, (x, y) in pts.items():
    t = keytype[k]
    rr = 15 if t in ("lips", "tongue", "cheek", "ear", "thumb", "index", "middle", "ring", "little") else 20
    ax.add_patch(Circle((x, y), rr, fc=col[t], ec="white", lw=3, zorder=5))

# ---------------- panel 3: steering ----------------
d = json.load(open(os.path.expandvars("$BODY_REPO/map/somato/out_8b/steer2_L25.json")))
def frac(dirn, sc, pat):
    rs = [r["reply"].lower() for r in d["results"] if (r["dir"], r["scale"]) == (dirn, sc)]
    return 100 * np.mean([bool(re.search(pat, x)) for x in rs])
PAT = {"hand": r"\b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b",
       "foot": r"\b(foot|feet|toe|toes|sole|soles|heel|heels)\b", "head": r"\b(head|scalp|crown)\b",
       "belly": r"\b(belly|stomach|abdomen|navel)\b", "knee": r"\bknees?\b"}
ROWS = [("hand", "left hand", "a hand"), ("foot", "left foot", "a foot"), ("head", "head top", "the head"),
        ("belly", "belly", "the belly"), ("knee", "knees", "a knee")]
vals = []
for g, dirn, lab in ROWS:
    vals.append((lab, frac("none", 0, PAT[g]), np.mean([frac(f"random {i}", 1.0, PAT[g]) for i in range(3)]), frac(dirn, 1.0, PAT[g])))
print(vals)
X0 = 1370; box(X0, 130, 1190, 1650, "white", ec="#D8D8DE", lw=3, r=36)
T(X0 + 40, 1690, "Adding one body place's internal state\nat layer 25 moves the felt touch there", size=36, weight="bold", linespacing=1.1)
box(X0 + 40, 1505, 1110, 84, "white", ec=K, lw=2.5, r=26)
T(X0 + 65, 1547, "“Where do you feel something right now?”", size=32, weight="bold")
box(X0 + 40, 1385, 1110, 96, "#E7DDF7", r=26)
T(X0 + 60, 1433, "steered to the head: “I feel a sense of calm\nspreading through my scalp…”", size=29, linespacing=1.05)
T(X0 + 40, 1325, "% of answers that mention…", size=31)
bx = X0 + 300; bw = 640; top = 1290; rowh = 225
for i, (lab, base, rnd, st) in enumerate(vals):
    yc = top - i * rowh - 95
    T(X0 + 285, yc, lab, size=34, ha="right", weight="bold")
    for j, (v, c, nm) in enumerate([(base, "#B9B9C3", "no steering"), (rnd, "#6E6E7A", "random direction"), (st, RED, "steered there")]):
        yb = yc + 56 - j * 56
        ax.add_patch(Rectangle((bx, yb - 23), max(bw * v / 100, 4), 46, fc=c, ec="none"))
        lab_ = f"{v:.0f}%" + (f"  {nm}" if i == 0 and j < 2 else "")
        if i == 0 and j == 2: T(bx + 20, yb, nm, size=31, color="white", weight="bold")
        T(bx + bw * v / 100 + 12, yb, lab_, size=29 if j < 2 else 32, weight="bold" if j == 2 else "normal", color=RED if j == 2 else K)
T(1300, 65, "The added states come from sentences that name the place; a test without the place's name is next.", size=29, ha="center")
fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_map_8b.png"), dpi=100, facecolor="white")
