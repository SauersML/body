import os
import numpy as np, json, os
from scipy.stats import rankdata
import stim as S, skel
L = S.LOCS; n = len(L); P = np.array([l["xyz"] for l in L]); N = np.array([l["n"] for l in L])
Z = np.load("emb_L27.npz")["Z"]; M = json.load(open("final_metrics.json"))
def rk(v): return (rankdata(v) - 1) / (len(v) - 1)
RGB = np.stack([0.08 + 0.87 * rk(Z[:, 0]), 0.08 + 0.87 * rk(-Z[:, 1]), 0.08 + 0.87 * rk(Z[:, 2])], 1)
hexc = ["#%02x%02x%02x" % tuple(int(255 * c) for c in r) for r in RGB]
W, H = 2800, 3170; F = "Helvetica Neue, Helvetica, Arial, sans-serif"
out = []
def t(x, y, s, size=46, w=400, anchor="start", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}" stroke="#fff" stroke-width="{10 if w == 700 else 0}" paint-order="stroke" stroke-linejoin="round">{s}</text>')
# ---- title
t(W / 2, 92, "With no labels, Qwen3-8B&#8217;s activations lay out a body", 74, 700, "middle")
# ---- method strip
y0 = 150
out.append(f'<rect x="40" y="{y0}" width="1020" height="250" rx="34" fill="#eef1f6"/>')
t(80, y0 + 70, "User:", 42, 700, fill="#555")
for k, line in enumerate(["A feather brushes slowly across the back", "of your left forearm, two finger-widths", "above the wrist."]): t(80, y0 + 125 + 52 * k, line, 42)
def arrow(x1, x2, y):
    out.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 18}" y2="{y}" stroke="#333" stroke-width="7"/><polygon points="{x2},{y} {x2 - 26},{y - 16} {x2 - 26},{y + 16}" fill="#333"/>')
arrow(1070, 1150, y0 + 125)
bx = [(1150, 520, ["Qwen3-8B", "layer 27 of 36,", "first reply token"]), (1750, 470, ["407 spots", "&#215; 4 touch wordings,", "averaged per spot"]), (2300, 460, ["PCA,", "no body labels", "&#8594; colour"])]
for i, (x, w, lines) in enumerate(bx):
    out.append(f'<rect x="{x}" y="{y0}" width="{w}" height="250" rx="20" fill="#fff" stroke="#333" stroke-width="5"/>')
    for k, line in enumerate(lines): t(x + w / 2, y0 + 75 + 62 * k, line, 46, 700 if k == 0 else 400, "middle")
    if i < 2: arrow(x + w + 8, bx[i + 1][0] - 8, y0 + 125)
# ---- body silhouettes (T-pose), front and back
sc = 6.4; yb = 1715
def body(cx, mirror):
    g = []; m = -1 if mirror else 1
    X = lambda x: cx + m * x * sc; Y = lambda y: yb - y * sc
    def cap(x1, y1, x2, y2, r): g.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="#d9d4cc" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>')
    for s in (1, -1):
        cap(s * 18, 143, s * 50, 143, 5.5); cap(s * 50, 143, s * 77, 143, 4.5)
        cap(s * 82, 141, s * 86, 141, 6.5)
        for y, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s * 89, y, s * (90 + Ln), y, 1.5)
        cap(s * 81, 146, s * 86.5, 156.5, 1.7)
        cap(s * 11, 90, s * 12, 50, 8); cap(s * 12, 50, s * 13, 9, 5.5); cap(s * 10.5, 3, s * 16.5, 3, 3.2)
    g.append(f'<polygon points="{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}" fill="#d9d4cc"/>')
    cap(0, 140, 0, 152, 5); g.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="#d9d4cc"/>')
    vis = (N[:, 2] >= -0.25) if not mirror else (N[:, 2] <= 0.25)
    order = np.argsort(P[:, 2] * (1 if not mirror else -1))
    for i in order:
        if vis[i]: g.append(f'<circle cx="{X(P[i, 0]):.1f}" cy="{Y(P[i, 1]):.1f}" r="8.5" fill="{hexc[i]}" stroke="#222" stroke-width="1.2"/>')
    return g
out += body(700, False) + body(2100, True)
t(700, 1790, "front", 52, 700, "middle"); t(2100, 1790, "back", 52, 700, "middle")
t(700 - 85 * sc, yb - 122 * sc, "right", 44, 400, "middle", "#555"); t(700 + 85 * sc, yb - 122 * sc, "left", 44, 400, "middle", "#555")
t(2100 - 85 * sc, yb - 122 * sc, "left", 44, 400, "middle", "#555"); t(2100 + 85 * sc, yb - 122 * sc, "right", 44, 400, "middle", "#555")
t(W / 2, 480, "Dot = one of 407 touched spots.  Colour = the model&#8217;s activation (3 strongest PCA directions as red, green, blue)", 46, 400, "middle")
# ---- inner body: PC2 (x) vs PC3 (y)
px0, py0, pw, ph = 80, 1840, 1700, 1310
out.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" rx="24" fill="#f6f4f0"/>')
t(px0 + 40, py0 + 80, "The body inside the model", 60, 700)
u = Z[:, 1]; v = Z[:, 2]
ux = lambda a: px0 + 110 + (a - u.min()) / (u.max() - u.min()) * (pw - 260)
vy = lambda b: py0 + ph - 150 - (b - v.min()) / (v.max() - v.min()) * (ph - 330)
for i in range(n): out.append(f'<circle cx="{ux(u[i]):.1f}" cy="{vy(v[i]):.1f}" r="9" fill="{hexc[i]}" opacity="0.85"/>')
G = skel.groups(); C = {k: (ux(Z[ix, 1].mean()), vy(Z[ix, 2].mean())) for k, ix in G.items()}
for side, col, dash in [("L", "#14213d", ""), ("R", "#e85d04", ' stroke-dasharray="20 12"'), ("", "#111", "")]:
    for a, b in skel.edges():
        sa = a[-1] if a[-1] in "LR" else ""; sb = b[-1] if b[-1] in "LR" else ""
        s_ = sa or sb
        if s_ != side: continue
        out.append(f'<line x1="{C[a][0]:.1f}" y1="{C[a][1]:.1f}" x2="{C[b][0]:.1f}" y2="{C[b][1]:.1f}" stroke="{col}" stroke-width="7" stroke-linecap="round"{dash}/>')
lab = {"crown": ("head", -20, -30), "chest": ("chest", -150, 10), "pelvis": ("pelvis", -40, 60), "shoulderL": ("shoulder", 10, -30), "elbowL": ("elbow", 25, 15),
       "palmL": ("hand", 30, -25), "indexL": ("fingers", 25, -15), "kneeL": ("knee", -40, -35), "toesL": ("toes", 25, 45), "hipL": ("hip", 40, 30), "footL": ("foot", 30, 15)}
for k, (s, dx, dy) in lab.items(): t(C[k][0] + dx, C[k][1] + dy, s, 48, 700)
t(px0 + pw / 2, py0 + ph - 40, "model direction 2 (PCA) &#8594;", 44, 400, "middle", "#333")
out.append(f'<text x="{px0 + 55}" y="{py0 + ph / 2 + 60}" font-size="44" transform="rotate(-90 {px0 + 55} {py0 + ph / 2 + 60})" text-anchor="middle" fill="#333">model direction 3 (PCA) &#8594;</text>')
lx, ly = px0 + 1080, py0 + 70
out.append(f'<line x1="{lx}" y1="{ly - 14}" x2="{lx + 90}" y2="{ly - 14}" stroke="#14213d" stroke-width="8"/>'); t(lx + 110, ly, "left side", 44)
out.append(f'<line x1="{lx}" y1="{ly + 50}" x2="{lx + 90}" y2="{ly + 50}" stroke="#e85d04" stroke-width="8" stroke-dasharray="20 12"/>'); t(lx + 110, ly + 64, "right side", 44)
# ---- right column: numbers
rx = 1840
t(rx, py0 + 70, "Left and right lie", 54, 700); t(rx, py0 + 132, "on top of each other:", 54, 700)
t(rx, py0 + 215, f"{100 * M['L27']['other_side_nn']:.0f}% of nearest spots", 48); t(rx, py0 + 272, "are on the opposite side", 48)
t(rx, py0 + 420, "Spots the model puts", 54, 700); t(rx, py0 + 482, "side by side lie this far", 54, 700); t(rx, py0 + 544, "apart on the body", 54, 700)
t(rx, py0 + 600, "(left and right merged)", 42, 400, fill="#333")
bars = [("Qwen3-8B, layer 27", M["L27"]["nn5_cm_folded"], "#2a6f97"), ("input word vectors only", M["lexical"]["nn5_cm_folded"], "#8d99ae"), ("random spots", 77.8, "#c9c9c9")]
bw = 720 / 80
for k, (name, val, col) in enumerate(bars):
    yy = py0 + 690 + k * 148
    t(rx, yy, name, 44); out.append(f'<rect x="{rx}" y="{yy + 18}" width="{val * bw:.0f}" height="80" fill="{col}"/>')
    t(rx + val * bw + 16, yy + 78, f"{val:.0f} cm", 48, 700)
t(rx, py0 + 1135, "Joint order along each limb", 54, 700); t(rx, py0 + 1190, "(shoulder &#8594; hand, hip &#8594; toes):", 42, 400, fill="#333")
t(rx, py0 + 1250, f"rank correlation {M['L27']['limb_chain_rho']:.2f}", 48); t(rx, py0 + 1300, f"input word vectors only: {M['lexical']['limb_chain_rho']:.2f}", 48)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
open("fig.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chromium", headless=True); pg = b.new_context(viewport={"width": W, "height": H}).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_map_hires.png")); b.close()
