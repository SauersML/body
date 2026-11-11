import os, numpy as np, json
exec(open(os.path.expandvars("$BODY_REPO/figures/bodychain/data_part.py")).read())
from scipy.stats import rankdata, spearmanr
import matplotlib.colors as mc
W_, H_ = 2400, 1940
K = "#111111"; SKIN = "#E6EAF0"; MODC = "#0A2A85"; WORDC = "#A06A00"
def hexc(t): return mc.to_hex(t)
def cmix(c0, c1, t):
    c0 = np.array(mc.to_rgb(c0)); c1 = np.array(mc.to_rgb(c1)); return hexc(c0 + (c1 - c0) * t)
ARMC = lambda d: cmix("#93C4F5", "#0A2A85", d / 6)
LEGC = lambda d: cmix("#F9B98F", "#9E1F0B", d / 6)
S_ = []
def add(s): S_.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, px=48, anchor="start", weight=500, color=K, rot=None, raw=False, extra=""):
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{px}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}" dominant-baseline="central"{tr} {extra}>{s if raw else esc(s)}</text>')
def poly(pts, fc, op=1):
    add('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{fc}" fill-opacity="{op}"/>')
def circ(p, r, fc, ec="none", lw=0, op=1):
    add(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{fc}" stroke="{ec}" stroke-width="{lw}" fill-opacity="{op}"/>')
def ell(c, w, h, ang=0, fc=SKIN):
    add(f'<ellipse cx="{c[0]:.1f}" cy="{c[1]:.1f}" rx="{w/2}" ry="{h/2}" transform="rotate({ang} {c[0]:.1f} {c[1]:.1f})" fill="{fc}"/>')
def line(pts, c, lw, op=1, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{c}" stroke-width="{lw}" stroke-opacity="{op}" stroke-linecap="{cap}" stroke-linejoin="round"{d}/>')
def arrow(x0, y, x1, c=K, lw=7, hl=34, hw=26):
    line([(x0, y), (x1 - hl * .6, y)], c, lw, cap="butt"); poly([(x1, y), (x1 - hl, y - hw / 2 * 1.4), (x1 - hl, y + hw / 2 * 1.4)], c)

# ================= header =================
text(40, 66, "Inside Qwen3-8B, body parts lie in body order:", 64, weight=700)
text(40, 142, "fingers → shoulder, toes → hip. Word meaning alone shows no such order.", 64, weight=700)
# ================= method =================
MY = 262
add(f'<rect x="40" y="{MY-48}" width="1150" height="96" rx="48" fill="#EDF2FA"/>')
text(76, MY, 'A feather brushes slowly across your left <tspan font-weight="700" fill="#0A2A85">thumb.</tspan>', 48, raw=True)
arrow(1172, MY, 1262)
# internal-state icon: column of cells
for k in range(6):
    v = [0.9, 0.3, 0.6, 0.15, 0.75, 0.45][k]
    add(f'<rect x="1282" y="{MY-66+k*22}" width="44" height="20" rx="3" fill="{cmix("#DCE6F5", MODC, v)}"/>')
text(1352, MY - 30, "Qwen3-8B internal state (layer 25 of 36)", 48)
text(1352, MY + 32, "8 such sentences for each of 32 body parts", 48)

# ================= body =================
dy = 110
def capsule(p0, p1, r0, r1, n=28, fc=SKIN):
    p0, p1 = np.array(p0, float), np.array(p1, float); d = p1 - p0; u = d / np.hypot(*d); a0 = np.arctan2(u[0], -u[1])
    pts = [p1 + r1 * np.array([np.cos(a0 - t), np.sin(a0 - t)]) for t in np.linspace(0, np.pi, n)]
    pts += [p0 + r0 * np.array([np.cos(a0 + np.pi - t), np.sin(a0 + np.pi - t)]) for t in np.linspace(0, np.pi, n)]
    poly(pts, fc)
cx = 312
ell((cx, 440 + dy), 128, 156)
capsule((cx, 500 + dy), (cx, 580 + dy), 36, 38)
torso = [(cx - 118, 588), (cx + 118, 588), (cx + 130, 640), (cx + 108, 780), (cx + 86, 880), (cx + 112, 990), (cx + 118, 1060),
         (cx - 118, 1060), (cx - 112, 990), (cx - 86, 880), (cx - 108, 780), (cx - 130, 640)]
poly([(a, b + dy) for a, b in torso], SKIN)
J = {}
for s in (1, -1):
    sh, el, wr = (cx + s * 134, 628 + dy), (cx + s * 196, 855 + dy), (cx + s * 240, 1055 + dy)
    capsule(sh, el, 42, 31); capsule(el, wr, 33, 24)
    hp, kn, an = (cx + s * 70, 1040 + dy), (cx + s * 76, 1390 + dy), (cx + s * 70, 1690 + dy)
    capsule(hp, kn, 58, 42); capsule(kn, an, 42, 26)
    ell((cx + s * 88, 1724 + dy), 112, 46, -s * 6)
    hc = np.array((cx + s * 248, 1122. + dy)); ell(hc, 78, 100, -s * 5)
    capsule(wr, hc + (0, -30), 24, 34)
    fx = [-26, -9, 8, 24]; fl = [62, 76, 70, 58]
    tips = [hc + (s * 52, 52)]; capsule(hc + (s * 26, -10), tips[0], 13, 10)
    for o, l in zip(fx[::-1] if s == 1 else fx, (fl[::-1] if s == 1 else fl)):
        b = hc + (s * o, 40); t = b + (s * o * 0.25, l); capsule(b, t, 11, 9); tips.append(t)
    J[s] = dict(sh=np.array(sh, float), el=np.array(el, float), wr=np.array(wr, float), hc=hc, tips=tips,
                hp=np.array(hp, float), kn=np.array(kn, float), an=np.array(an, float))
j = J[1]
bodypos = {"shoulder": j["sh"], "upperarm": (j["sh"] + j["el"]) / 2, "elbow": j["el"], "forearm": (j["el"] + j["wr"]) / 2, "wrist": j["wr"],
           "palm": j["hc"] + (-8, 4), "backhand": j["hc"] + (0, 52)}
for t, p in zip(["thumb", "index", "middle", "ring", "little"], j["tips"]): bodypos[t] = p
legpos = {"hip": j["hp"], "thigh": (j["hp"] + j["kn"]) / 2, "knee": j["kn"], "shin": (j["kn"] + j["an"]) / 2, "ankle": j["an"],
          "heel": j["an"] + (-10, 44), "sole": j["an"] + (26, 50), "toes": j["an"] + (70, 38)}
def node(p, c, r=19):
    circ(p, r + 4, "white"); circ(p, r, c)
for t, p in bodypos.items(): node(p, ARMC(dep[t]), r=11 if t in ARM[:5] else 18)
for t, p in legpos.items(): node(p, LEGC(dep[t]), r=14 if t in ("heel", "sole", "toes") else 18)
text(cx, 1890, "on the body", 48, anchor="middle", weight=700)

# ================= model maps =================
NAMES = dict(palm="palm", backhand="knuckles", wrist="wrist", forearm="forearm", elbow="elbow", upperarm="upper arm", shoulder="shoulder",
             toes="toes", sole="sole", heel="heel", ankle="ankle", shin="shin", knee="knee", thigh="thigh", hip="hip")
def draw_map(m, sel, col, cxm, top, scale):
    ymax = max(m[t][0] for t in sel)
    P = {t: np.array([cxm + scale * m[t][1], top + scale * (ymax - m[t][0])]) for t in sel}
    groups = sorted(set(dep[t] for t in sel)); cent = {d: np.mean([P[t] for t in sel if dep[t] == d], 0) for d in groups}
    line([cent[d] for d in groups], col(3), 10, op=.42)
    for t in sel:
        c = cent[dep[t]]
        if np.hypot(*(P[t] - c)) > 1: line([c, P[t]], col(dep[t]), 4, op=.42)
    for t in sel: node(P[t], col(dep[t]), r=21)
    return P
MX = 885
text(MX, 1890, "inside Qwen3-8B", 48, anchor="middle", weight=700)
PA = draw_map(maps["arm"], ARM, ARMC, MX, 430, 52)
PL = draw_map(maps["leg"], LEG, LEGC, MX, 1080, 50)
OFF = dict(wrist=(1, 46), elbow=(-1, -6), knee=(1, -26), ankle=(1, 30), heel=(1, 0))
for P in (PA, PL):
    for t, p in P.items():
        if t in ARM[:5]: continue
        side = OFF.get(t, (1 if p[0] >= MX else -1, 0))
        text(p[0] + side[0] * 34, p[1] + side[1], NAMES[t], 48, anchor="start" if side[0] > 0 else "end")
fc_ = np.array([PA[t] for t in ARM[:5]]); text(fc_[:, 0].max() + 34, fc_[:, 1].mean(), "fingers", 48)
json.dump({t: list(map(float, p)) for P in (PA, PL) for t, p in P.items()}, open("mappos.json", "w"))

# ================= right: two matching panels =================
raw = rankdata(D[it]); raw = (raw - 1) / (len(raw) - 1)
wr_ = rankdata(Dw[it]); wr_ = (wr_ - 1) / (len(wr_) - 1)
rho_raw = spearmanr(D[it], Gt[it])[0]
rng = np.random.default_rng(0); jit = rng.uniform(-.18, .18, len(x))
PX0, PW, PH = 1420, 930, 350
def panel(y0, ys, col, xlabels=False):
    X = lambda v: PX0 + PW * v / 15.5; Y = lambda v: y0 + PH * (1 - v)
    add(f'<rect x="{X(9.75):.1f}" y="{y0}" width="{X(15.5)-X(9.75):.1f}" height="{PH}" fill="#F0F0F0"/>')
    for xv, yv in zip(x + jit, ys): circ((X(xv), Y(yv)), 5.5, col, op=.33)
    g = np.arange(0.5, 15.01, 0.5); m = np.array([np.median(ys[np.abs(x - q) <= 1.0]) for q in g])
    line([(X(a), Y(b)) for a, b in zip(g, m)], "white", 17)
    line([(X(a), Y(b)) for a, b in zip(g, m)], col, 10)
    line([(PX0, y0), (PX0, y0 + PH), (PX0 + PW, y0 + PH)], K, 3, cap="butt")
    text(PX0 - 44, y0 + PH / 2, "differ more →", 46, anchor="middle", rot=-90)
    return X, Y
# panel A
PX0, PW = 1450, 900
A0 = 470
text(1345, 372, "Inside Qwen3-8B", 56, weight=700, color=MODC)
text(1345, 428, "each dot: two body parts compared", 44)
X, Y = panel(A0, raw, MODC)
text(X(12.62), A0 + 34, "mostly", 44, anchor="middle")
text(X(12.62), A0 + 84, "arm ↔ leg", 44, anchor="middle")
# panel B
B0 = 1085
text(1345, 900, "Could similar wording explain the order?", 50, weight=700, color=WORDC)
text(1345, 958, "‘left thumb’ and ‘left index finger’ share words.", 44)
text(1345, 1014, "Same pairs, word embeddings of the names:", 44)
X, Y = panel(B0, wr_, WORDC)
for v in (0, 5, 10, 15): text(X(v), B0 + PH + 36, str(v), 46, anchor="middle")
text(PX0 + PW / 2, B0 + PH + 96, "steps apart on the body →", 48, anchor="middle")
# gauge
G0 = 1712; GX0, GX1 = 1720, 2250; gx = lambda v: GX0 + (GX1 - GX0) * v
text(1345, G0 - 118, "How well closeness inside matches", 46, weight=700)
text(1345, G0 - 62, "closeness on the body", 46, weight=700)
for v, lab in ((0, "0 = none"), (1, "1 = perfect")):
    line([(gx(v), G0 - 10), (gx(v), G0 + 160)], K, 3, cap="butt")
    text(gx(v), G0 + 196, lab, 44, anchor="middle")
def bar(y, v, c, lab):
    x0, x1 = sorted((gx(0), gx(v)))
    add(f'<rect x="{x0:.1f}" y="{y-26}" width="{x1-x0:.1f}" height="52" fill="{c}"/>')
    text(1345, y, lab, 46, color=c, weight=700)
    text(gx(max(v, 0)) + 16, y, f"{v:.2f}".replace("-", "−"), 46, color=c, weight=700)
bar(G0 + 40, rho_raw, MODC, "Qwen")
bar(G0 + 118, rho_word, WORDC, "words only")
open("nums_used.json", "w").write(json.dumps(dict(rho_raw=rho_raw, rho_word=float(rho_word), rho_resid=float(rho_main))))

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" viewBox="0 0 {W_} {H_}" font-family="Avenir Next, Avenir, Helvetica Neue, sans-serif">' \
      f'<rect width="{W_}" height="{H_}" fill="#FFFFFF"/>' + "\n".join(S_) + "</svg>"
open("fig.svg", "w").write(svg)
open("fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
print(W_, H_)
