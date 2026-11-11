import os
import numpy as np, json, os, sys
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from ana2 import load_stim, Body
S = load_stim("stim.py"); b = Body(S); L = S.LOCS; n = len(L); P = b.P; N = np.array([l["n"] for l in L])
E = np.load(sys.argv[1]); Z = PCA(10).fit_transform(E["ZRu"]); AX = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [1, 2]
M = json.load(open("fig_numbers.json"))
def rk(v): return (rankdata(v) - 1) / (len(v) - 1)
RGB = np.stack([0.08 + 0.87 * rk(Z[:, 0]), 0.08 + 0.87 * rk(-Z[:, 1]), 0.08 + 0.87 * rk(Z[:, 2])], 1)
hexc = ["#%02x%02x%02x" % tuple(int(255 * c) for c in r) for r in RGB]
W, H = 2800, 3130; F = "Helvetica Neue, Helvetica, Arial, sans-serif"; out = []
def t(x, y, s, size=46, w=400, anchor="start", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}" stroke="#fff" stroke-width="{10 if w == 700 else 0}" paint-order="stroke" stroke-linejoin="round">{s}</text>')
t(W / 2, 92, "With no labels, Qwen3 activations lay out a body; the 32B model draws a sharper one", 66, 700, "middle")
y0 = 150
out.append(f'<rect x="40" y="{y0}" width="1000" height="270" rx="34" fill="#ffffff" stroke="#9aa4b1" stroke-width="4"/>')
t(80, y0 + 62, "User:", 40, 700, fill="#555")
for k, line in enumerate(["A feather brushes slowly across the back", "of your left forearm, two finger-widths", "above the wrist."]): t(80, y0 + 115 + 50 * k, line, 40)
def arrow(x1, x2, y): out.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 18}" y2="{y}" stroke="#333" stroke-width="7"/><polygon points="{x2},{y} {x2 - 26},{y - 16} {x2 - 26},{y + 16}" fill="#333"/>')
arrow(1045, 1110, y0 + 135)
bx = [(1110, 480, ["Qwen3-32B", "layer 53 of 64,", "first reply token"]), (1650, 500, ["550 spots", "&#215; 26 wordings", "(touch, pain, doctor&#8230;)"]), (2210, 550, ["keep directions where", "two halves of the", "wordings agree"])]
for i, (x, w, lines) in enumerate(bx):
    out.append(f'<rect x="{x}" y="{y0}" width="{w}" height="270" rx="20" fill="#fff" stroke="#333" stroke-width="5"/>')
    for k, line in enumerate(lines): t(x + w / 2, y0 + 80 + 64 * k, line, 44, 700 if k == 0 else 400, "middle")
    if i < 2: arrow(x + w + 6, bx[i + 1][0] - 6, y0 + 135)
t(W / 2, 500, "Dot = one touched spot.  Colour = the 3 strongest activation directions as red, green, blue (no body labels used)", 44, 400, "middle")
sc = 6.0; yb = 1640
def body(cx, mirror):
    g = []; m = -1 if mirror else 1; X = lambda x: cx + m * x * sc; Y = lambda y: yb - y * sc
    def cap(x1, y1, x2, y2, r): g.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="#dedede" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>')
    for s in (1, -1):
        cap(s * 18, 143, s * 50, 143, 5.5); cap(s * 50, 143, s * 77, 143, 4.5); cap(s * 82, 141, s * 86, 141, 6.5)
        for y, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s * 89, y, s * (90 + Ln), y, 1.5)
        cap(s * 81, 146, s * 86.5, 156.5, 1.7); cap(s * 11, 90, s * 12, 50, 8); cap(s * 12, 50, s * 13, 9, 5.5); cap(s * 10.5, 3, s * 16.5, 3, 3.2)
    g.append(f'<polygon points="{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}" fill="#dedede"/>')
    cap(0, 140, 0, 152, 5); g.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="#dedede"/>')
    vis = (N[:, 2] >= -0.25) if not mirror else (N[:, 2] <= 0.25)
    for i in np.argsort(P[:, 2] * (1 if not mirror else -1)):
        if vis[i]: g.append(f'<circle cx="{X(P[i, 0]):.1f}" cy="{Y(P[i, 1]):.1f}" r="7.5" fill="{hexc[i]}" stroke="#222" stroke-width="1.1"/>')
    return g
out += body(700, False) + body(2100, True)
t(700, 1715, "front", 50, 700, "middle"); t(2100, 1715, "back", 50, 700, "middle")
for cx, a, c in [(700, "right", "left"), (2100, "left", "right")]: t(cx - 85 * sc, yb - 125 * sc, a, 42, 400, "middle", "#555"); t(cx + 85 * sc, yb - 125 * sc, c, 42, 400, "middle", "#555")
# inner body
px0, py0, pw, ph = 60, 1770, 1520, 1330
out.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" rx="24" fill="#ffffff" stroke="#9aa4b1" stroke-width="4"/>'); t(px0 + 40, py0 + 76, "The body inside Qwen3-32B", 58, 700)
u = Z[:, AX[0]]; v = Z[:, AX[1]]
ux = lambda a: px0 + 110 + (a - u.min()) / (u.max() - u.min()) * (pw - 220)
vy = lambda c: py0 + ph - 120 - (c - v.min()) / (v.max() - v.min()) * (ph - 290)
for i in range(n): out.append(f'<circle cx="{ux(u[i]):.1f}" cy="{vy(v[i]):.1f}" r="8" fill="{hexc[i]}" opacity="0.8"/>')
C = lambda ix: (ux(Z[ix, AX[0]].mean()), vy(Z[ix, AX[1]].mean()))
reg = lambda f: [i for i, l in enumerate(L) if f(l)]
head = reg(lambda l: l["region"] == "head"); neck = reg(lambda l: l["region"] == "neck")
chest = reg(lambda l: l["region"] == "torso" and l["xyz"][1] > 125); belly = reg(lambda l: l["region"] == "torso" and 98 < l["xyz"][1] <= 125); pelvis = reg(lambda l: l["region"] == "torso" and l["xyz"][1] <= 98)
def line(pts, col, dash=""):
    out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
line([C(head), C(neck), C(chest), C(belly), C(pelvis)], "#111")
for ci, (sd, col, dash) in enumerate([("L", "#14213d", ""), ("R", "#e85d04", ' stroke-dasharray="20 12"')]):
    arm, leg = b.chains[2 * ci], b.chains[2 * ci + 1]
    line([C(chest)] + [C(g) for g in arm], col, dash); line([C(pelvis)] + [C(g) for g in leg], col, dash)
    S_ = "left" if sd == "L" else "right"
    for f in ["thumb", "index finger", "middle finger", "ring finger", "little finger"]:
        ix = reg(lambda l: f"{S_} {f}" in l["phrase"] and "between" not in l["phrase"])
        if ix: line([C(arm[-1]), C(ix)], col, dash)
lab = [("head", head, 0, -30), ("pelvis", pelvis, -60, 60), ("elbow", b.chains[0][6], 20, -25), ("hand", b.chains[0][-1], 25, -25), ("knee", b.chains[1][6], 20, -25), ("toes", b.chains[1][-1], 25, 40)]
for s, ix, dx, dy in lab: x, y = C(ix); t(x + dx, y + dy, s, 46, 700)
lx, ly = px0 + 980, py0 + 70
out.append(f'<line x1="{lx}" y1="{ly - 14}" x2="{lx + 90}" y2="{ly - 14}" stroke="#14213d" stroke-width="8"/>'); t(lx + 110, ly, "left side", 42)
out.append(f'<line x1="{lx}" y1="{ly + 48}" x2="{lx + 90}" y2="{ly + 48}" stroke="#e85d04" stroke-width="8" stroke-dasharray="20 12"/>'); t(lx + 110, ly + 62, "right side", 42)
# fidelity panel
rx = 1640; cols = ["#a9b4c2", "#5b8db8", "#1d3557"]; names = ["v1: Qwen3-8B, 4 wordings, PCA", "v2: Qwen3-8B", "v2: Qwen3-32B"]
for k, nm in enumerate(names): out.append(f'<rect x="{rx}" y="{py0 + 10 + 56 * k}" width="40" height="40" fill="{cols[k]}"/>'); t(rx + 56, py0 + 44 + 56 * k, nm, 40)
panels = [("Neighbours lie this far apart", "on the body (cm, lower is better)", "nn5_cm", 35, "{:.0f} cm"), ("Joint order along each limb", "(rank correlation)", "chain_rho", 1, "{:.2f}"),
          ("Nearest spot is the mirror twin", "(after the left/right shift)", "partner_nn", 1, "{:.0%}"), ("Description with no body word:", "rank of the true spot of 407 (lower is better)", "probe_wf_rank", 150, "{:.0f}")]
yy = py0 + 225
for title, sub, key, vmax, fmt in panels:
    t(rx, yy, title, 44, 700); t(rx, yy + 46, sub, 38, 400, fill="#333")
    for k in range(3):
        val = M[k][key]; w = 760 * val / vmax; y = yy + 70 + k * 58
        out.append(f'<rect x="{rx}" y="{y}" width="{w:.0f}" height="48" fill="{cols[k]}"/>'); t(rx + w + 14, y + 40, fmt.format(val), 40, 700)
    yy += 292
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
open("fig2.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + os.path.abspath("fig2.html")); pg.screenshot(path="/tmp/f2.png"); br.close()
from PIL import Image; Image.open("/tmp/f2.png").resize((W, H), Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_map_hires_v2.png"))
