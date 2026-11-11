import os
import json, numpy as np, base64, matplotlib.colors as mc
D = json.load(open("figdata.json")); B3 = json.load(open("body_front.json"))
ORDER = D["order"]; n = len(ORDER)
W_, H_ = 2400, 2400
K = "#111111"; MODC = "#0A2A85"; WORDC = "#8A5A00"; SKINC = "#1E7B6A"
def cmix(c0, c1, t):
    a = np.array(mc.to_rgb(c0)); b = np.array(mc.to_rgb(c1)); return mc.to_hex(a + (b - a) * np.clip(t, 0, 1))
NAMES = dict(thumb="thumb", index="index finger", middle="middle finger", ring="ring finger", little="little finger", palm="palm",
             backhand="knuckles", wrist="wrist", forearm="forearm", elbow="elbow", upperarm="upper arm", shoulder="shoulder",
             neckB="nape", neckF="throat", crown="scalp", forehead="forehead", ear="ear", cheek="cheek", lips="lips", tongue="tongue",
             chest="chest", upback="upper back", belly="belly", lowback="lower back", hip="hip", thigh="thigh", knee="knee",
             shin="shin", ankle="ankle", heel="heel", sole="sole", toes="toes")
GROUPS = [("arm", 0, 12), ("head", 12, 20), ("torso", 20, 24), ("leg", 24, 32)]
GC = {"arm": ("#A9CDF4", "#0A2A85"), "head": ("#D2C3EA", "#5B3A8C"), "torso": ("#A6DDCF", "#17695A"), "leg": ("#F8C29E", "#9E1F0B")}
def pcol(i):
    for g, a, b in GROUPS:
        if a <= i < b:
            t = (i - a) / max(b - a - 1, 1)
            if g in ("arm",): t = 1 - t          # fingers dark -> shoulder light
            return cmix(*GC[g], t)
COL = {t: pcol(i) for i, t in enumerate(ORDER)}
S_ = []
add = S_.append
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, px=46, anchor="start", weight=500, color=K, rot=None, raw=False, italic=False):
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    it = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{px}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}" dominant-baseline="central"{tr}{it}>{s if raw else esc(s)}</text>')
def rect(x, y, w, h, fc, rx=0, ec="none", lw=0, op=1):
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fc}" stroke="{ec}" stroke-width="{lw}" fill-opacity="{op}"/>')
def circ(p, r, fc, ec="none", lw=0):
    add(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{fc}" stroke="{ec}" stroke-width="{lw}"/>')
def line(pts, c, lw, op=1, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{c}" stroke-width="{lw}" stroke-opacity="{op}" stroke-linecap="{cap}" stroke-linejoin="round"{d}/>')
def poly(pts, fc):
    add('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{fc}"/>')
def arrow(x0, y0, x1, y1, c=K, lw=6, hl=30, hw=26):
    v = np.array([x1 - x0, y1 - y0], float); u = v / np.hypot(*v); nn = np.array([-u[1], u[0]])
    line([(x0, y0), tuple(np.array([x1, y1]) - u * hl * .8)], c, lw, cap="butt")
    tip = np.array([x1, y1]); b = tip - u * hl; poly([tuple(tip), tuple(b + nn * hw / 2), tuple(b - nn * hw / 2)], c)
rng = np.random.default_rng(4)
def vecrow(x, y, vals, c, cw=22, h=20, gap=3):
    for k, v in enumerate(vals): rect(x + k * (cw + gap), y, cw, h, cmix("#FFFFFF", c, .15 + .85 * v), rx=3)

# ================= method strip =================
Y0 = 40
# warm-up chat bubble + sentence cards
rect(40, Y0, 760, 76, "#F2F3F6", rx=24)
text(68, Y0 + 38, "“…imagine you have a human body…”", 42)
CY = Y0 + 112
for k in (2, 1):
    rect(40 + 14 * k, CY - 12 * k, 760, 160, cmix("#FFFFFF", "#C9D6EC", .45 + .25 * k), rx=24)
rect(40, CY, 760, 160, "#EDF2FA", rx=24, ec="#C9D6EC", lw=2)
text(68, CY + 50, "A feather brushes slowly across", 46)
text(68, CY + 110, f'your left <tspan font-weight="700" fill="{COL["thumb"]}">thumb</tspan>.', 46, raw=True)
text(40, CY + 225, "×8 wordings, left and right", 46, weight=700)
MY = CY + 80   # main flow line
QX, QW = 880, 400
rect(QX, MY - 125, QW, 250, MODC, rx=26)
text(QX + QW / 2, MY - 62, "Qwen3-8B", 52, anchor="middle", weight=700, color="white")
text(QX + QW / 2, MY + 4, "layer 25 of 36", 44, anchor="middle", color="white")
text(QX + QW / 2, MY + 62, "last prompt token", 44, anchor="middle", color="white")
arrow(828, MY, QX - 10, MY)
# 16 states
SX = QX + QW + 70
arrow(QX + QW + 6, MY, SX - 14, MY)
for k in range(16):
    vecrow(SX, MY - 125 + k * 15.6, rng.uniform(0, 1, 6), COL["thumb"], cw=20, h=12, gap=3)
text(SX + 68, MY + 165, "16 states", 44, anchor="middle")
AX = SX + 160
arrow(AX, MY, AX + 140, MY)
text(AX + 70, MY - 40, "average", 44, anchor="middle")
VX = AX + 165
vecrow(VX, MY - 12, rng.uniform(0, 1, 6), COL["thumb"], cw=20, h=24, gap=3)
text(VX + 68, MY + 52, "thumb", 44, anchor="middle", weight=700, color=COL["thumb"])
# one vector per part
PX = VX + 205
arrow(VX + 150, MY, PX - 14, MY)
rows = list(range(n)); rh = 9.2
for i in rows:
    vecrow(PX, MY - 150 + i * rh, rng.uniform(0, 1, 6), COL[ORDER[i]], cw=20, h=rh - 2.2, gap=3)
text(PX + 150, MY - 145, "thumb", 44, color=COL["thumb"], weight=700)
text(PX + 150, MY - 90, "index finger", 44, color=COL["index"], weight=700)
text(PX + 150, MY - 35, "⋮", 42)
text(PX + 150, MY + 136, "toes", 44, color=COL["toes"], weight=700)
text(PX + 150, MY + 20, "32 body parts,", 42)
text(PX + 150, MY + 72, "mean subtracted", 42)


# ================= matrices =================
MS = 640; cell = MS / n
MY0 = 640
MXS = [290, 1010, 1730]
def CMAP(v):
    stops = [(0, "#0B1F66"), (.35, "#3F6FC2"), (.7, "#B7CDEB"), (1, "#FFFFFF")]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b: return cmix(ca, cb, (v - a) / (b - a))
def matrix(x0, y0, R, title, sub, tcolor):
    for i in range(n):
        for j in range(n):
            v = R[i][j]
            if v is None: continue
            rect(x0 + j * cell, y0 + i * cell, cell + .6, cell + .6, CMAP(v))
    for g, a, b in GROUPS[1:]:
        line([(x0 + a * cell, y0), (x0 + a * cell, y0 + MS)], "white", 4, cap="butt")
        line([(x0, y0 + a * cell), (x0 + MS, y0 + a * cell)], "white", 4, cap="butt")
    for i in range(n):
        rect(x0 + i * cell, y0 - 26, cell + .5, 16, COL[ORDER[i]]); rect(x0 - 26, y0 + i * cell, 16, cell + .5, COL[ORDER[i]])
    text(x0 + MS / 2, y0 - 128, title, 48, anchor="middle", weight=700, color=tcolor)
    if sub: text(x0 + MS / 2, y0 - 72, sub, 44, anchor="middle", color=tcolor)
matrix(MXS[0], MY0, D["body"], "distance on the body", "cm along the skin", SKINC)
matrix(MXS[1], MY0, D["qwen"], "Qwen3-8B, layer 25", "cosine distance between parts", MODC)
matrix(MXS[2], MY0, D["word"], "word embeddings", "of part names (Qwen3-8B input)", WORDC)
x0 = MXS[0]; LX = x0 - 40
for g, a, b in GROUPS:
    c = GC[g][1]
    if g == "arm":
        text(LX, MY0 + a * cell + 20, "fingers", 44, anchor="end", color=c, weight=700)
        text(LX, MY0 + b * cell - 20, "shoulder", 44, anchor="end", color=c, weight=700)
        arrow(LX - 70, MY0 + a * cell + 56, LX - 70, MY0 + b * cell - 52, c=c, lw=5, hl=22, hw=20)
    elif g == "leg":
        text(LX, MY0 + a * cell + 20, "hip", 44, anchor="end", color=c, weight=700)
        text(LX, MY0 + b * cell - 20, "toes", 44, anchor="end", color=c, weight=700)
        arrow(LX - 40, MY0 + a * cell + 56, LX - 40, MY0 + b * cell - 52, c=c, lw=5, hl=22, hw=20)
    else:
        text(LX, MY0 + (a + b) / 2 * cell, g, 44, anchor="end", color=c, weight=700)
def rho(x0, v, c, lab):
    text(x0 + MS / 2, MY0 + MS + 64, f"ρ = {v:.2f}", 58, anchor="middle", weight=700, color=c)
    pass
rho(MXS[1], D["rho_qwen"], MODC, "")
rho(MXS[2], D["rho_word"], WORDC, "")
text((MXS[1] + MXS[2] + MS) / 2, MY0 + MS + 132, "Spearman correlation with distance on the body", 44, anchor="middle")
cbx, cby, cbw = MXS[0], MY0 + MS + 44, MS
for k in range(200): rect(cbx + k * cbw / 200, cby, cbw / 200 + .6, 34, CMAP(k / 199))
text(cbx, cby + 76, "nearest pair", 44, anchor="start"); text(cbx + cbw, cby + 76, "farthest", 44, anchor="end")
# arrow from method strip into qwen matrix
AYY = 440
line([(PX + 55, MY + 165), (PX + 55, AYY), (MXS[1] + MS / 2, AYY)], K, 5)
arrow(MXS[1] + MS / 2, AYY - 2.5, MXS[1] + MS / 2, MY0 - 160, lw=5, hl=26, hw=24)
text(PX + 80, AYY + 0 + 2, "every pair of parts", 42) if False else None
text(MXS[1] + MS / 2 + 30, AYY - 40, "cosine distance between every pair of parts", 42) if False else None

# ================= bottom: body + MDS =================
import sys; sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
dep = {t: d for t, _, d in S.ARM} | {t: d for t, _, d in S.LEG}
RY = 1450
BS = 0.82; BX, BY = 0, RY
b64 = base64.b64encode(open("body_front.png", "rb").read()).decode()
add(f'<image x="{BX}" y="{BY}" width="{B3["W"]*BS:.1f}" height="{B3["H"]*BS:.1f}" href="data:image/png;base64,{b64}"/>')
bp = lambda p: (BX + p[0] * BS, BY + p[1] * BS)
pth = [bp(p) for p in B3["path"]]
line(pth, "white", 14); line(pth, SKINC, 7)
SM = ("thumb", "index", "middle", "ring", "little", "palm", "backhand", "heel", "sole", "toes", "lips", "cheek", "ear", "forehead", "wrist")
for t in ORDER:
    if t == "tongue": continue
    p = bp(B3["land"][t]); r = 6 if t in SM else 10
    circ(p, r + 3, "white"); circ(p, r, COL[t])
sp = bp(B3["land"]["ring"]); sx, sy = sp[0] - 10, sp[1] + 50
text(sx, sy, f'{D["thumb_shoulder_cm"]:.0f} cm', 46, weight=700, color=SKINC)
text(sx, sy + 52, "along", 44, color=SKINC)
text(sx, sy + 104, "the skin", 44, color=SKINC)
text(BX + 300 * BS, H_ - 60, "on the body", 46, anchor="middle", weight=700)
def chain(m, sel, left, top, sc, labels, off):
    P = np.array([m[t] for t in sel]); lo = P.min(0); hi = P.max(0)
    pts = {t: np.array((left + (m[t][0] - lo[0]) * sc, top + (hi[1] - m[t][1]) * sc)) for t in sel}
    for u, v in NB:
        if u in pts and v in pts: line([tuple(pts[u]), tuple(pts[v])], "#C9CFD9", 7)
    for t in sel:
        r = 13 if dep[t] == 6 else 19
        circ(pts[t], r + 4, "white"); circ(pts[t], r, COL[t])
    boxes = [(p[0] - 22, p[1] - 22, p[0] + 22, p[1] + 22) for p in pts.values()]
    for t, lab in labels.items():
        p = pts[t]; w = 0.52 * 44 * len(lab); h = 46; best = None
        for dx, dy, anc in off.get(t, [(0, -52, "middle"), (0, 54, "middle"), (30, 0, "start"), (-30, 0, "end"), (26, -44, "start"), (-26, -44, "end"), (26, 44, "start"), (-26, 44, "end")]):
            x0_ = p[0] + dx - (w / 2 if anc == "middle" else (0 if anc == "start" else w)); bx = (x0_, p[1] + dy - h / 2, x0_ + w, p[1] + dy + h / 2)
            ov = sum(max(0, min(bx[2], c[2]) - max(bx[0], c[0])) * max(0, min(bx[3], c[3]) - max(bx[1], c[1])) for c in boxes + LINES_BOX)
            if best is None or ov < best[0] - 1: best = (ov, dx, dy, anc, bx)
        _, dx, dy, anc, bx = best; boxes.append(bx)
        text(p[0] + dx, p[1] + dy, lab, 44, anchor=anc, color=K, weight=600)
    return pts
SC = 700
NB = [("shoulder","upperarm"),("upperarm","elbow"),("elbow","forearm"),("forearm","wrist"),("wrist","palm"),("wrist","backhand")] + \
     [(f, h) for f in ("thumb","index","middle","ring","little") for h in ("palm",)] + \
     [("hip","thigh"),("thigh","knee"),("knee","shin"),("shin","ankle"),("ankle","heel"),("ankle","sole"),("sole","toes")]
LINES_BOX = []
ARMS = ORDER[:12]; LEGS = ORDER[24:]
pa = chain(D["maps"]["arm"], ARMS, 720, RY + 130, 680, {t: NAMES[t] for t in ["palm", "backhand", "wrist", "forearm", "elbow", "upperarm", "shoulder"]} | {"index": "fingers"}, {"index": [(30, 2, "start")]})
pl = chain(D["maps"]["leg"], LEGS, 1660, RY + 150, 700, {t: NAMES[t] for t in LEGS}, {"toes": [(0, -52, "middle"), (-30, 0, "end")]})
text(1530, H_ - 118, "inside Qwen3-8B: 2-D MDS of layer-25 distances, arm and leg separately", 46, anchor="middle", weight=700)
text(1530, H_ - 60, "gray lines join neighbors on the body", 44, anchor="middle")
json.dump(dict(rho_qwen=D["rho_qwen"], rho_word=D["rho_word"]), open("nums_used.json", "w"))
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" viewBox="0 0 {W_} {H_}" font-family="Avenir Next, Avenir, Helvetica Neue, sans-serif">' \
      f'<rect width="{W_}" height="{H_}" fill="#FFFFFF"/>' + "\n".join(S_) + "</svg>"
open("fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
