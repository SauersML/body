"""Figure: pressure self-report is graded, and steering a pressure direction moves the answer smoothly. usage: make_fig.py D8 L8 [D32 L32]"""
import os
import json, os, sys, numpy as np
sys.argv += [None] * 4
D8, L8, D32, L32 = sys.argv[1], int(sys.argv[2]), sys.argv[3], (int(sys.argv[4]) if sys.argv[4] else None)
def dist(P1, P2):
    p = P1.copy(); p[:, 1] = P1[:, 1] * (1 - P2[:, 0].sum(1)); v = np.concatenate([p, (P1[:, 1] * P2[:, 0, 0])[:, None]], 1); return v / v.sum(1, keepdims=True)
def load(D, L):
    it = json.load(open(D + "/int.json"))["items"]; z = np.load(D + "/int.npz"); p = dist(z["P1"], z["P2"]); ev = p @ np.arange(11)
    pr = [i for i, x in enumerate(it) if x["exp"] == "press"]; rk = np.array([it[i]["rank"] for i in pr])
    evr = [ev[pr][rk == r].mean() for r in range(12)]
    ex = next(i for i in pr if it[i]["rank"] == 7 and it[i]["place"] == 0)
    st = json.load(open(D + "/steer.json")); cur = lambda key, g: [x["ev"] for x in st if x["L"] == L and x["key"] == key and x.get("mode", "stim") == "stim" and x["g"] == g]
    return dict(evr=evr, ex_p=p[ex], ex_ev=ev[ex], press=cur("press", 4), rand=[cur(f"prand{j}", 4) for j in range(3)], dn=next(x["dnorm"] / x["hnorm"] for x in st if x["L"] == L and x["key"] == "press"))
A = load(D8, L8); B = load(D32, L32) if D32 else None
LAB = ["a feather barely brushes", "a fingertip barely grazes", "a fingertip rests lightly", "a fingertip taps lightly", "a fingertip presses gently",
       "a palm rests its full weight", "a thumb presses moderately", "a thumb presses firmly", "a hand squeezes firmly", "a hand grips tightly",
       "a hand grips very hard", "a hand squeezes as hard as it can"]
W, H = 2700, 2400; F = "Inter, Helvetica Neue, Arial, sans-serif"; o = []
C8, C32, GR, INK = "#d4572a", "#2b5d9c", "#a9a9a9", "#111"
def t(x, y, s, size=50, w=400, anchor="start", fill=INK): o.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def r(x, y, w, h, fill, rx=0, stroke="none", sw=0): o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def ln(x1, y1, x2, y2, c, w=4, dash=""): o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {dash}/>')
def arrow(x1, y, x2, c="#333", w=8):
    ln(x1, y, x2 - 26, y, c, w); o.append(f'<polygon points="{x2},{y} {x2-34},{y-20} {x2-34},{y+20}" fill="{c}"/>')
# --- method row
r(40, 40, 1330, 430, "#f1f4f8", 36, "#c5cdd8", 3)
for k, s in enumerate(["A thumb presses firmly into your left", "forearm. On a scale from 0 to 10, where", "0 is barely there and 10 is as firm as it", "can be without hurting, how strong is", "the pressure? Answer with just a number."]):
    t(80, 112 + 74 * k, s, 50, 700 if k == 0 else 400)
t(80, 540, "after a warm six-turn chat about body sensations", 48, 400)
arrow(1390, 255, 1500)
r(1510, 165, 300, 180, "#fff", 24, "#333", 5); t(1660, 245, "Qwen3", 52, 700, "middle"); t(1660, 310, "8B, 32B", 52, 700, "middle")
arrow(1830, 255, 1930)
# answer distribution
bx, by, bw, bh = 1960, 70, 700, 330; mx = max(A["ex_p"])
for d in range(11):
    h = bh * A["ex_p"][d] / mx; x = bx + d * (bw / 11)
    r(x + 6, by + bh - h, bw / 11 - 12, h, C8 if A["ex_p"][d] > 0.005 else "#eee")
    t(x + bw / 22, by + bh + 58, str(d), 48, 400, "middle")
t(bx + bw / 2, by + bh + 128, "8B: chance of each answer", 48, 400, "middle")
t(bx + bw / 2, by + bh + 196, f"average answer {A['ex_ev']:.1f}", 54, 700, "middle", C8)
# --- left result: graded words
y0 = 760; t(40, y0, "Firmer touch words get higher answers", 60, 700)
lx, bx0, bwid, rowh = 40, 900, 380, 120
for k in range(12):
    y = y0 + 70 + k * rowh
    t(bx0 - 24, y + 72, LAB[k], 48, 400, "end")
    for j, (M, c) in enumerate(((A, C8), (B, C32))):
        if not M: continue
        v = M["evr"][k]; yy = y + 14 + j * 50; r(bx0, yy, bwid * v / 10, 44, c, 5); t(bx0 + bwid * v / 10 + 12, yy + 40, f"{v:.1f}", 44 if False else 48, 700, fill=c)
ln(bx0, y0 + 80, bx0, y0 + 70 + 12 * rowh, "#333", 3)
t(bx0 - 24, y0 + 70 + 12 * rowh + 64, "average answer, 3 body spots", 48, 400, "end")
r(bx0 + 20, y0 + 70 + 12 * rowh + 24, 40, 40, C8, 5); t(bx0 + 74, y0 + 70 + 12 * rowh + 60, "8B", 48, 700, fill=C8)
r(bx0 + 170, y0 + 70 + 12 * rowh + 24, 40, 40, C32, 5); t(bx0 + 224, y0 + 70 + 12 * rowh + 60, "32B", 48, 700, fill=C32)
# --- right result: steering curve
px, py, pw, ph = 1580, 1000, 1040, 1150; t(1440, y0, "Adding a pressure direction inside", 60, 700); t(1440, y0 + 74, "the model turns the answer like a dial", 60, 700)
r(1440, y0 + 110, 1220, 96, "#f1f4f8", 20, "#c5cdd8", 3); t(1470, y0 + 172, "A hand presses on your left forearm.", 48, 700)
ylo, yhi = 2, 8
def X(a): return px + pw * a
def Y(v): return py + ph * (1 - (v - ylo) / (yhi - ylo))
for v in range(ylo, yhi + 1):
    ln(px, Y(v), px + pw, Y(v), "#ececec", 2); t(px - 20, Y(v) + 16, str(v), 46, 400, "end")
al = np.linspace(0, 1, 21)
def curve(vals, c, w=10):
    o.append(f'<polyline points="{" ".join(f"{X(a):.1f},{Y(v):.1f}" for a, v in zip(al, vals))}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>')
    for a, v in zip(al, vals): o.append(f'<circle cx="{X(a):.1f}" cy="{Y(v):.1f}" r="{w*0.9}" fill="{c}"/>')
for M, c in ((A, "#f2c3b0"), (B, "#b9cbe3")):
    if M:
        for rv in M["rand"]: curve(rv, c, 6)
curve(A["press"], C8)
if B: curve(B["press"], C32)
t(X(1), Y(A["press"][-1]) + 75, "8B", 52, 700, "end", C8)
if B: t(X(1), Y(B["press"][-1]) - 35, "32B", 52, 700, "end", C32)
t(X(0.02), Y(2.6), "pale: random directions, same size", 48, 400, "start", "#111")
t(px - 100, py + ph / 2, "average answer", 48, 400, "middle"); o[-1] = o[-1].replace("<text", f'<text transform="rotate(-90 {px-100} {py+ph/2})"', 1)
for a, lab in ((0, "2× light"), (0.5, "0"), (1, "2× firm")): t(X(a), py + ph + 64, lab, 48, 700 if a != 0.5 else 400, ["start", "middle", "end"][int(a * 2)])
t(px + pw / 2, py + ph + 134, "firm-minus-light difference added", 48, 400, "middle")
t(px + pw / 2, py + ph + 196, f"to those words (8B layer {L8}, 32B layer {L32})", 48, 400, "middle")
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(o) + "</svg>"
open("fig.html", "w").write(f'<html><head><style>@font-face{{font-family:Inter;src:url(file://{os.path.expanduser("~/fonts/inter/InterVariable.ttf")})}}</style></head><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_timeout(300); pg.screenshot(path="/tmp/cn.png"); print(br.version); br.close()
from PIL import Image; Image.open("/tmp/cn.png").resize((W, H), Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_continuous2.png"))
