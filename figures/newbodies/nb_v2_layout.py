# layout for nb_v2.py (exec'd inside it)
L0, R0 = 60, W - 60
COL = {"HUM": HUM, "INK": INK, "ACC": ACC}
text(L0, 76, "Qwen3-32B maps robot parts onto human parts with the same job", size=70, weight=700, fam="Inter Display")

# ===== method strip =====
y0 = 160; MH = 560; cw = 860
rect(L0, y0, cw, MH, CARD)
robot(L0 + 150, y0 + 520, s=0.6)
text(L0 + 320, y0 + 70, "\u201cYou are a small robot.", size=48)
text(L0 + 320, y0 + 128, [("You roll on two ", INK, 400), ("wheels", ACC, 700), ("\u2026\u201d", INK, 400)], size=48)
rect(L0 + 295, y0 + 300, 540, 230, "#FBE3D9", r=22)
text(L0 + 322, y0 + 360, "\u201cA feather brushes", size=48)
text(L0 + 322, y0 + 418, [("across your ", INK, 400), ("left wheel", ACC, 700), (".\u201d", INK, 400)], size=48)
text(L0 + 322, y0 + 478, "+ 7 other touches", size=48)
hx0 = R0 - cw
rect(hx0, y0, cw, MH, CARD)
P = human(hx0 + 150, y0 + 26, s=0.44)
for k, (x, y) in PL.items(): circ(*P(x, y), 8, HUM, "#FFFFFF", 2.5)
circ(*P(*PL["heelL"]), 15, HUM, "#FFFFFF", 4)
text(hx0 + 320, y0 + 70, "\u201cYou\u2019re standing in", size=48)
text(hx0 + 320, y0 + 128, "a quiet, sunny room\u2026\u201d", size=48)
rect(hx0 + 295, y0 + 300, 540, 230, "#DCE7F5", r=22)
text(hx0 + 322, y0 + 360, "\u201cA feather brushes", size=48)
text(hx0 + 322, y0 + 418, [("across your ", INK, 400), ("left heel", HUM, 700), (".\u201d", INK, 400)], size=48)
text(hx0 + 322, y0 + 478, [("54 places", HUM, 700)], size=48)
mx = W / 2; mt = y0 + 10
text(mx, mt + 12, "Qwen3-32B", size=48, anchor="middle", weight=700)
for i in range(11):
    yy = mt + 58 + i * 25; hl = i == 8
    rect(mx - 165, yy - (2 if hl else 0), 330, 20 if hl else 15, INK if hl else "#D5DAE1", r=6)
text(mx, mt + 58 + 11 * 25 + 30, "layer 52", size=48, anchor="middle", weight=700)
arrow(L0 + cw + 14, y0 + 300, mx - 182, mt + 150, ACC, 7, bend=-30)
arrow(hx0 - 14, y0 + 300, mx + 182, mt + 150, HUM, 7, bend=30)
import random
def vec(x, y, col, n=9, cell=30, seed=0):
    r = random.Random(seed)
    for j in range(n): rect(x + j * cell, y, cell - 5, cell - 5, col, r=4, extra=f'opacity="{0.3 + 0.7 * r.random():.2f}"')
vy = mt + 420
vec(mx - 300, vy, ACC, seed=1)
for j in range(3): vec(mx + 30 + j * 10, vy + 20 - j * 10, HUM, seed=5 + j)
text(mx - 165, vy + 72, [("left wheel", ACC, 700)], size=48, anchor="middle")
text(mx + 175, vy + 72, [("54 places", HUM, 700)], size=48, anchor="middle")
text(mx, vy + 12, "\u2248", size=64, anchor="middle", weight=700)
text(mx, vy + 135, "closest place", size=48, anchor="middle", weight=600)
arrow(L0 + cw + 14, y0 + 470, mx - 320, vy + 14, ACC, 0.001) if False else None

# ===== bodies with arcs =====
by = y0 + MH + 80
PW = (R0 - L0) / 2
def panel(i, title, body, names):
    x0 = L0 + i * PW
    text(x0 + 10, by + 10, title, size=50, weight=700)
    base = by + 900; hs = 0.66
    Ph = human(x0 + PW - 300, base - 1180 * hs, s=hs)
    A = robot(x0 + 300, base - 10, s=0.95)
    for part, (txt, dx, dy, anc) in names.items():
        q = A[part]; text(q[0] + dx, q[1] + dy, [(txt, ACC, 700)], size=48, anchor=anc)
    labs = {}
    for part, q in A.items():
        sims = D["near"][body][part]["sims"]; k, v = max(sims.items(), key=lambda kv: kv[1])
        h = Ph(*PL[k]); arc(q, h, ACC, 4 + 30 * max(v, 0), bend=0.10 if q[1] > h[1] else -0.10)
        labs.setdefault(nm(k), []).append(h)
    for k, pts in labs.items():
        x = max(p[0] for p in pts); y = sum(p[1] for p in pts) / len(pts)
        text(x + 30, y, k, size=48, weight=600)
RN = {"wheelL": ("wheels", 70, 105, "middle"), "antenna": ("antenna", -24, -50, "middle"), "camera": ("camera", -95, 0, "end"),
      "gripper": ("gripper", 0, 80, "middle"), "chassis": ("chassis", 0, 70, "middle")}
GN = {"wheelL": ("glorps", 70, 105, "middle"), "antenna": ("tesk", -24, -50, "middle"), "camera": ("mirl", -95, 0, "end"),
      "gripper": ("vask", 0, 80, "middle"), "chassis": ("brell", 0, 70, "middle")}
panel(0, "real names", "robot", RN)
panel(1, "made-up names: \u201cyou roll on two glorps\u201d", "glorp", GN)
add(f'<line x1="{L0 + PW}" y1="{by + 70}" x2="{L0 + PW}" y2="{by + 900}" stroke="#D5DAE1" stroke-width="4"/>')

# ===== steering =====
sy = by + 1060
text(L0, sy, "Adding a human body-part direction makes the robot say the matching part", size=58, weight=700, fam="Inter Display")
cy = sy + 70; bw = 980
rect(L0, cy, bw, 140, CARD)
text(L0 + 30, cy + 46, "\u201cWhich part of your body is", size=48); text(L0 + 30, cy + 102, "drawing your attention?\u201d", size=48)
arrow(L0 + 200, cy + 148, L0 + 200, cy + 200, INK, 6)
rect(L0, cy + 206, bw, 96, "#FFFFFF", stroke=HUM, sw=6)
text(L0 + 30, cy + 254, [("+ human foot direction, layer 52", HUM, 700)], size=48)
arrow(L0 + 200, cy + 310, L0 + 200, cy + 362, INK, 6)
rect(L0, cy + 368, bw, 140, "#FBE3D9")
text(L0 + 30, cy + 414, [("\u201cMy ", INK, 400), ("left glorp", ACC, 700), (" is drawing my attention", INK, 400)], size=48)
text(L0 + 30, cy + 470, "\u2026 slightly flattened at the bottom \u2026\u201d", size=48)
# bars
st = D["steer"]; n = st["n"]
gx = L0 + bw + 110; gw = R0 - gx; barmax = gw - 150
yy = cy + 10
first = True
for g in st["groups"]:
    text(gx, yy + 10, [(t, COL[c], 700) for t, c in g["label"]], size=48)
    yy += 52
    for v, c in zip([g["vals"][0], g["vals"][2], g["vals"][1]], [HUM, "#8C96A3", "#CBD2DB"]):
        w = max(barmax * v / n, 6); h = 52 if c == HUM else 40
        rect(gx, yy, w, h, c, r=6)
        tag = "" if (c == HUM or not first) else ("   random direction" if c == "#8C96A3" else "   no steering")
        text(gx + w + 16, yy + h / 2, [(f"{round(100 * v / n)}%", INK, 700 if c == HUM else 400), (tag, INK, 400)], size=48)
        yy += h + (12 if c == HUM else 16)
    yy += 26; first = False
rect(L0, cy + 545, bw, 140, "#FBE3D9")
text(L0 + 30, cy + 591, [("+ human hand", HUM, 700), (":  \u201cI notice a gentle hum", INK, 400)], size=48)
text(L0 + 30, cy + 647, [("in my ", INK, 400), ("metal fingers", ACC, 700), (" \u2026 ", INK, 400), ("gripper arm", ACC, 700), (" \u2026\u201d", INK, 400)], size=48)
