import os, sys, numpy as np, json
exec(open(os.path.expandvars("$BODY_REPO/figures/bodychain/data_part.py")).read())
from scipy.stats import rankdata, spearmanr
import matplotlib.colors as mc
W_, H_ = 2400, 2330
K = "#111111"; SKIN = "#E6EAF0"; MODC = "#0A2A85"; WORDC = "#A06A00"; NEU = "#7E8796"
def hexc(t): return mc.to_hex(t)
def cmix(c0, c1, t):
    c0 = np.array(mc.to_rgb(c0)); c1 = np.array(mc.to_rgb(c1)); return hexc(c0 + (c1 - c0) * t)
ARMC = lambda d: cmix("#93C4F5", "#0A2A85", d / 6)
LEGC = lambda d: cmix("#F9B98F", "#9E1F0B", d / 6)
S_ = []
def add(s): S_.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, px=46, anchor="start", weight=500, color=K, rot=None, raw=False):
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{px}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}" dominant-baseline="central"{tr}>{s if raw else esc(s)}</text>')
def poly(pts, fc, op=1):
    add('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{fc}" fill-opacity="{op}"/>')
def circ(p, r, fc, ec="none", lw=0, op=1):
    add(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{fc}" stroke="{ec}" stroke-width="{lw}" fill-opacity="{op}"/>')
def ell(c, w, h, ang=0, fc=SKIN):
    add(f'<ellipse cx="{c[0]:.1f}" cy="{c[1]:.1f}" rx="{w/2}" ry="{h/2}" transform="rotate({ang} {c[0]:.1f} {c[1]:.1f})" fill="{fc}"/>')
def rect(x, y, w, h, fc, rx=0, ec="none", lw=0):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fc}" stroke="{ec}" stroke-width="{lw}"/>')
def line(pts, c, lw, op=1, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{c}" stroke-width="{lw}" stroke-opacity="{op}" stroke-linecap="{cap}" stroke-linejoin="round"{d}/>')
def arrow(x0, y0, x1, y1, c=K, lw=6, hl=30, hw=24):
    v = np.array([x1 - x0, y1 - y0], float); L = np.hypot(*v); u = v / L; n = np.array([-u[1], u[0]])
    e = np.array([x1, y1]) - u * hl * .7; line([(x0, y0), tuple(e)], c, lw, cap="butt")
    tip = np.array([x1, y1]); b = tip - u * hl; poly([tuple(tip), tuple(b + n * hw / 2), tuple(b - n * hw / 2)], c)
def vec(x, y, vals, c, w=34, h=22, gap=4):
    for k, v in enumerate(vals): rect(x, y + k * (h + gap), w, h, cmix("#FFFFFF", c, .25 + .75 * v), rx=3)

# ================= title =================
text(40, 64, "Inside Qwen3-8B, body parts lie in body order: fingers → shoulder,", 60, weight=700)
text(40, 138, "toes → hip. Word embeddings of the place names show no such order.", 60, weight=700)

# ================= method strip =================
def cards(x, y, w, h, s):
    for k in (2, 1):
        rect(x + 14 * k, y - 12 * k, w, h, cmix("#FFFFFF", "#C9D6EC", .5 + .25 * k), rx=22)
    rect(x, y, w, h, "#EDF2FA", rx=22, ec="#C9D6EC", lw=2)
    text(x + 26, y + h / 2, s, 42, raw=True)
CX, CW, CH = 40, 1070, 92
R1, R2 = 268, 448
text(CX, 212 + 0, "", 10)
cards(CX, R1 - CH / 2, CW, CH, 'A feather brushes slowly across your left <tspan font-weight="700" fill="#0A2A85">thumb.</tspan>')
cards(CX, R2 - CH / 2, CW, CH, 'An ice cube touches your right <tspan font-weight="700" fill="#5E8FD0">elbow</tspan> …')
text(CX + CW - 30, R1, "×8", 44, weight=700, anchor="end")
text(CX + CW - 30, R2, "×8", 44, weight=700, anchor="end")
text(CX, 548, "8 wordings per body part, left and right sides pooled; each sent after a", 40)
text(CX, 598, "short chat: “imagine you have a human body … tell me what you notice”", 40)
# Qwen box
QX, QW = 1215, 320
rect(QX, R1 - 74, QW, R2 - R1 + 148, "#0A2A85", rx=26)
text(QX + QW / 2, R1 - 20, "Qwen3-8B", 50, anchor="middle", weight=700, color="white")
text(QX + QW / 2, R1 + 40, "layer 25 of 36", 42, anchor="middle", color="white")
text(QX + QW / 2, R1 + 95, "last token", 42, anchor="middle", color="white")
text(QX + QW / 2, R1 + 145, "before the reply", 42, anchor="middle", color="white")
arrow(CX + CW + 40, R1, QX - 8, R1); arrow(CX + CW + 40, R2, QX - 8, R2)
# vectors: two halves per part
VX = QX + QW + 60
rng0 = np.random.default_rng(3)
for (yy, nm, c) in ((R1, "thumb", "#0A2A85"), (R2, "elbow", "#5E8FD0")):
    arrow(QX + QW + 4, yy, VX - 6, yy, lw=5, hl=24, hw=20)
    vals = rng0.uniform(0, 1, 4)
    vec(VX, yy - 50, vals, c); vec(VX + 52, yy - 50, np.clip(vals + rng0.normal(0, .15, 4), 0, 1), c)
    text(VX + 120, yy - 28, f'{nm}<tspan baseline-shift="sub" font-size="30">1</tspan>  {nm}<tspan baseline-shift="sub" font-size="30">2</tspan>', 42, raw=True, weight=600)
    text(VX + 120, yy + 26, "means over odd / even wordings", 36)
# formula
FX = 1590
fy = 640
sub = lambda n: f'<tspan baseline-shift="sub" font-size="30">{n}</tspan>'
text(QX - 520 + 0, 0, "", 1)
text(1150, 668, f'distance = (thumb{sub(1)} − elbow{sub(1)}) · (thumb{sub(2)} − elbow{sub(2)})', 46, raw=True, weight=700)
text(1150, 728, "(cross-validated squared Euclidean distance)", 40)
line([(VX + 90, R2 + 66), (VX + 90, 630)], K, 4, dash="2 10")
# compare row
text(40, 800, "Repeat for all 496 pairs of 32 body parts, then compare with steps apart on the body:", 44, weight=700)
text(40, 858, "Spearman rank correlation (ρ). Control: the same distance between word embeddings", 44)
text(40, 914, "of the place names (Qwen’s input layer: similarity already in the input, before any processing).", 44)
line([(40, 960), (2360, 960)], "#D5DAE2", 3)

# ================= body =================
YS = 690
SC = 0.86; add(f'<g transform="translate({300-300*SC:.1f} {1000-362*SC:.1f}) scale({SC})">')
def capsule(p0, p1, r0, r1, n=28, fc=SKIN):
    p0, p1 = np.array(p0, float), np.array(p1, float); d = p1 - p0; u = d / np.hypot(*d); a0 = np.arctan2(u[0], -u[1])
    pts = [p1 + r1 * np.array([np.cos(a0 - t), np.sin(a0 - t)]) for t in np.linspace(0, np.pi, n)]
    pts += [p0 + r0 * np.array([np.cos(a0 + np.pi - t), np.sin(a0 + np.pi - t)]) for t in np.linspace(0, np.pi, n)]
    poly(pts, fc)
cx = 300; oy = 0
ell((cx, 440 + oy), 128, 156)
capsule((cx, 500 + oy), (cx, 580 + oy), 36, 38)
torso = [(cx - 118, 588), (cx + 118, 588), (cx + 130, 640), (cx + 108, 780), (cx + 86, 880), (cx + 112, 990), (cx + 118, 1060),
         (cx - 118, 1060), (cx - 112, 990), (cx - 86, 880), (cx - 108, 780), (cx - 130, 640)]
poly([(a, b + oy) for a, b in torso], SKIN)
J = {}
for s in (1, -1):
    sh, el, wr = (cx + s * 134, 628 + oy), (cx + s * 196, 855 + oy), (cx + s * 240, 1055 + oy)
    capsule(sh, el, 42, 31); capsule(el, wr, 33, 24)
    hp, kn, an = (cx + s * 70, 1040 + oy), (cx + s * 76, 1390 + oy), (cx + s * 70, 1690 + oy)
    capsule(hp, kn, 58, 42); capsule(kn, an, 42, 26)
    ell((cx + s * 88, 1724 + oy), 112, 46, -s * 6)
    hc = np.array((cx + s * 248, 1122. + oy)); ell(hc, 78, 100, -s * 5)
    capsule(wr, hc + (0, -30), 24, 34)
    fx = [-26, -9, 8, 24]; fl = [62, 76, 70, 58]
    tips = [hc + (s * 52, 52)]; capsule(hc + (s * 26, -10), tips[0], 13, 10)
    for o, l in zip(fx[::-1] if s == 1 else fx, (fl[::-1] if s == 1 else fl)):
        b = hc + (s * o, 40); t = b + (s * o * 0.25, l); capsule(b, t, 11, 9); tips.append(t)
    J[s] = dict(sh=np.array(sh, float), el=np.array(el, float), wr=np.array(wr, float), hc=hc, tips=tips,
                hp=np.array(hp, float), kn=np.array(kn, float), an=np.array(an, float))
j = J[1]
P0 = {"shoulder": j["sh"], "upperarm": (j["sh"] + j["el"]) / 2, "elbow": j["el"], "forearm": (j["el"] + j["wr"]) / 2, "wrist": j["wr"],
      "palm": j["hc"] + (-8, 4), "backhand": j["hc"] + (0, 52)}
for t, p in zip(["thumb", "index", "middle", "ring", "little"], j["tips"]): P0[t] = p
P0.update({"hip": j["hp"], "thigh": (j["hp"] + j["kn"]) / 2, "knee": j["kn"], "shin": (j["kn"] + j["an"]) / 2, "ankle": j["an"],
           "heel": j["an"] + (-10, 44), "sole": j["an"] + (26, 50), "toes": j["an"] + (70, 38)})
hy = 440 + oy
P0.update({"crown": np.array((cx, hy - 62.)), "forehead": np.array((cx, hy - 26.)), "cheek": np.array((cx + 34, hy + 18.)),
           "ear": np.array((cx + 66, hy + 2.)), "lips": np.array((cx - 8, hy + 50.)), "tongue": np.array((cx + 14, hy + 50.)),
           "neckF": np.array((cx - 16, 545. + oy)), "neckB": np.array((cx + 16, 545. + oy)),
           "chest": np.array((cx - 30, 690. + oy)), "upback": np.array((cx + 30, 690. + oy)),
           "belly": np.array((cx - 30, 900. + oy)), "lowback": np.array((cx + 30, 900. + oy))})
# skeleton graph edges (left side + midline) from the analysis's own graph
import networkx as nx
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato"))
Sg = __import__("stimuli")
ed = []
src = open(os.path.expandvars("$BODY_REPO/map/somato/stimuli.py")).read()
class G_:
    def __init__(s): s.E = []
    def add_edge(s, a, b, weight=1): s.E.append((a, b))
import types
gg = G_()
nxm = types.SimpleNamespace(Graph=lambda: gg, all_pairs_dijkstra_path_length=lambda G: {})
code = Sg.geodesic.__code__
try:
    sys.modules["networkx"] = nxm; Sg.geodesic()
except Exception: pass
finally:
    sys.modules["networkx"] = nx
def keyt(k):
    if k.endswith("L") and k[:-1] in P0: return k[:-1]
    if k.endswith("R"): return None
    return k if k in P0 else None
for a, b in gg.E:
    ta, tb = keyt(a), keyt(b)
    if ta and tb: line([P0[ta], P0[tb]], "#9AA3B2", 3.5)
def node(p, c, r=19):
    circ(p, r + 4, "white"); circ(p, r, c)
for t, p in P0.items():
    if t in dep: c = ARMC(dep[t]) if t in ARM else LEGC(dep[t])
    else: c = NEU
    small = t in ARM[:5] or t in ("heel", "sole", "toes") or t not in dep
    node(p, c, r=11 if t in ARM[:5] else (13 if small else 17))
text(cx, 1850 + oy - 200 + 250, "", 1)
BY = 1800 + oy - 1860 + 1860
add("</g>")
text(300, 2245, "on the body", 44, anchor="middle", weight=700)
text(300, 2297, "lines: skeleton graph", 40, anchor="middle")
text(cx, 2275 + 0, "", 1)

# ================= model maps (MDS) =================
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
MX = 960
PA = draw_map(maps["arm"], ARM, ARMC, MX, 995, 50)
PL = draw_map(maps["leg"], LEG, LEGC, MX, 1625, 46)
OFF = dict(wrist=(1, 46), elbow=(-1, -6), knee=(1, -26), ankle=(1, 30), heel=(1, 0))
for P in (PA, PL):
    for t, p in P.items():
        if t in ARM[:5]: continue
        side = OFF.get(t, (1 if p[0] >= MX else -1, 0))
        text(p[0] + side[0] * 34, p[1] + side[1], NAMES[t], 46, anchor="start" if side[0] > 0 else "end")
fc_ = np.array([PA[t] for t in ARM[:5]]); text(fc_[:, 0].max() + 34, fc_[:, 1].mean(), "fingers", 46)
# MDS axes for each map
def mdsaxes(P, pad=50):
    pts = np.array(list(P.values())); x0 = 640; y1 = pts[:, 1].max() + 30; y0 = pts[:, 1].min() - 20
    arrow(x0, y1, x0, y0, c=K, lw=3, hl=20, hw=16)
    text(x0 - 18, (y0 + y1) / 2, "MDS dimension 1", 38, anchor="middle", rot=-90)
ax0, ay0 = 610, 1440
arrow(ax0, ay0, ax0, ay0 - 90, lw=3, hl=20, hw=16); arrow(ax0, ay0, ax0 + 90, ay0, lw=3, hl=20, hw=16)
text(ax0 + 14, ay0 - 84, "MDS 1", 36); text(ax0 + 102, ay0, "MDS 2", 36)
text(MX, 2245, "inside Qwen3-8B: 2-D MDS of the", 44, anchor="middle", weight=700)
text(MX, 2297, "layer-25 distances (arm, leg separately)", 40, anchor="middle")

# ================= right: scatters =================
y_m = D[it]; y_w = Dw[it]
rho_raw = spearmanr(D[it], Gt[it])[0]
rng = np.random.default_rng(0); jit = rng.uniform(-.18, .18, len(x))
PX0, PW, PH = 1520, 830, 300
def panel(y0, ys, col, ymax, ticks, ylab, ylab2):
    X = lambda v: PX0 + PW * v / 15.5; Y = lambda v: y0 + PH * (1 - v / ymax)
    rect(X(9.75), y0, X(15.5) - X(9.75), PH, "#F0F0F0")
    for xv, yv in zip(x + jit, ys): circ((X(xv), Y(min(yv, ymax))), 5.5, col, op=.33)
    g = np.arange(0.5, 15.01, 0.5); m = np.array([np.median(ys[np.abs(x - q) <= 1.0]) for q in g])
    line([(X(a), Y(b)) for a, b in zip(g, m)], "white", 17)
    line([(X(a), Y(b)) for a, b in zip(g, m)], col, 10)
    line([(PX0, y0 - 10), (PX0, y0 + PH), (PX0 + PW, y0 + PH)], K, 3, cap="butt")
    for t in ticks:
        line([(PX0 - 12, Y(t)), (PX0, Y(t))], K, 3, cap="butt"); text(PX0 - 20, Y(t), f"{t:g}", 40, anchor="end")
    for v in (0, 5, 10, 15):
        line([(X(v), y0 + PH), (X(v), y0 + PH + 12)], K, 3, cap="butt"); text(X(v), y0 + PH + 40, str(v), 40, anchor="middle")
    text(PX0 + PW / 2, y0 + PH + 92, "steps apart on the body (graph at left) →", 42, anchor="middle")
    text(1400, y0 - 104, ylab, 44, weight=700, color=col)
    text(1400, y0 - 52, ylab2, 42, color=col)
    return X, Y
A0 = 1110
X, Y = panel(A0, y_m, MODC, 560, (0, 200, 400), "↑ Distance between Qwen3-8B’s layer-25", "vectors for the two body parts (one dot = one pair)")
text(X(12.62), A0 + 36, "mostly", 40, anchor="middle"); text(X(12.62), A0 + 82, "arm ↔ leg", 40, anchor="middle")
B0 = 1670
X, Y = panel(B0, y_w, WORDC, 5, (0, 2, 4), "↑ Distance between word embeddings", "of the two place names (Qwen’s input layer)")
# gauge
G0 = 2150; GX0, GX1 = 1990, 2290; gx = lambda v: GX0 + (GX1 - GX0) * v / 0.6
def bar(y, v, c, lab):
    x0, x1 = sorted((gx(0), gx(v)))
    rect(x0, y - 24, x1 - x0, 48, c)
    text(1400, y, lab, 42, color=c, weight=700)
    text(gx(max(v, 0)) + 14, y, f"{v:.2f}".replace("-", "−"), 42, color=c, weight=700)
text(1400, G0 - 26, "Spearman ρ with steps apart on the body", 44, weight=700)
line([(gx(0), G0 + 6), (gx(0), G0 + 172)], K, 3, cap="butt")
bar(G0 + 46, rho_raw, MODC, "Qwen3-8B layer 25")
bar(G0 + 126, rho_word, WORDC, "word embeddings")
json.dump(dict(rho_qwen_L25=float(rho_raw), rho_wordemb=float(rho_word), n_pairs=int(len(x))), open("nums_used.json", "w"))
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" viewBox="0 0 {W_} {H_}" font-family="Avenir Next, Avenir, Helvetica Neue, sans-serif">' \
      f'<rect width="{W_}" height="{H_}" fill="#FFFFFF"/>' + "\n".join(S_) + "</svg>"
open("fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
