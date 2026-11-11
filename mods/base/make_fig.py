import os
import numpy as np, json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import mstim as MS, stim as S
lay = 27; KEEP = [22, 24, 25, 26, 27, 28, 30]; li = KEEP.index(lay); R = MS.probes(); spots = MS.SPOTS; ri = {s: k for k, s in enumerate(spots)}
def load(n):
    H = np.load(f"{D}/H/{n.replace(':', '__')}.npy")[li].astype(np.float32); X = np.zeros((len(spots), H.shape[1]))
    for i, r in enumerate(R):
        if r["kind"] == "loc": X[ri[r["li"]]] += H[i] / 2
    X -= X.mean(0); return X / np.sqrt((X ** 2).sum(1).mean())
Xn = load("neutral"); V = np.linalg.svd(Xn, full_matrices=False)[2][:10].T
def rel(a, b):
    d = np.linalg.norm((load(a) - load(b)) @ V, axis=1); return d / d.mean()
P = np.array([S.LOCS[s]["xyz"] for s in spots])
PAN = [("L_arm_up", "Left arm raised"), ("arms_crossed", "Arms crossed"), ("cross_legged", "Sitting cross-legged"), ("on_side", "Lying on right side"),
       ("R_hand_behind", "Right hand behind back"), ("R_leg_lifted", "Right knee lifted"), ("R_hand_wide", "Right hand spread wide"), ("L_hand_huge", "Left hand twice as big"),
       ("head_tiny", "Head shrinks to tiny"), ("feet_huge", "Feet become huge"), ("L_arm_long", "Left arm twice as long"), ("third_arm", "Third arm grows from back"),
       ("tail", "A tail grows"), ("wings", "Wings sprout"), ("R_hand_gone", "Right hand is gone"), ("L_fin", "Left fingers fuse into a fin"),
       ("R_arm_stone", "Right arm turns to stone"), ("back_sensitive", "Back very sensitive"), ("L_warm_R_cold", "Left warm, right ice-cold"), ("liquid", "Body turns to liquid"),
       ("ribbon", "Stretched into a ribbon"), ("tree", "Becomes a tree")]
vals = [rel(f"told_N:{m}", "neutral") for m, _ in PAN]
vals += [rel("do:do_L_arm_up:0", "neutral"), rel("edit:do_L_arm_up", "do:do_L_arm_up:0"), rel("persist:R_hand_gone", "persist:neutral")]
titles = [t for _, t in PAN] + ["Asked to raise left arm", "Reply says right arm, vs. left", "Hand gone, 3 chats later"]
def col(r):  # log-ratio diverging: blue (moved less than average) – white – red (moved more)
    x = np.clip(np.log(r) / np.log(1.6), -1, 1)
    a = np.array([255, 255, 255]); hi = np.array([196, 40, 30]); lo = np.array([60, 110, 180])
    c = a + (hi - a) * x if x > 0 else a + (lo - a) * (-x); return "#%02x%02x%02x" % tuple(int(v) for v in c)
W = 2600; cols = 5; pw = W // cols; ph = 470; top = 560; H = top + 5 * ph + 20
out = []; F = "Inter, Helvetica, Arial, sans-serif"
def t(x, y, s, size=40, w=400, anchor="middle", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
REG = np.array([("torso" if S.LOCS[q]["region"] in ("torso", "neck") else S.LOCS[q]["region"]) for q in spots])
def body(cx, yb, sc, v):
    X = lambda x: cx + x * sc; Y = lambda y: yb - y * sc  # front view: person's left on viewer's right
    g = {k: col(v[REG == k].mean()) for k in set(REG)}
    def cap(x1, y1, x2, y2, r, c): out.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{c}" stroke-width="{2 * r * sc:.1f}" stroke-linecap="round"/>')
    def outline(x1, y1, x2, y2, r): out.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="#333" stroke-width="{2 * r * sc + 5:.1f}" stroke-linecap="round"/>')
    segs = []
    for s_, L_ in ((1, "L"), (-1, "R")):
        segs += [(s_ * 18, 143, s_ * 50, 143, 5.5, "arm" + L_), (s_ * 50, 143, s_ * 77, 143, 4.5, "arm" + L_), (s_ * 82, 141, s_ * 96, 141, 7.5, "hand" + L_),
                 (s_ * 11, 90, s_ * 12, 50, 8.5, "leg" + L_), (s_ * 12, 50, s_ * 13, 9, 6, "leg" + L_), (s_ * 9.5, 3, s_ * 18.5, 3, 4, "foot" + L_)]
    poly = f"{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}"
    for x1, y1, x2, y2, r, k in segs: outline(x1, y1, x2, y2, r)
    out.append(f'<polygon points="{poly}" fill="#333" stroke="#333" stroke-width="5" stroke-linejoin="round"/>')
    out.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc + 2.5}" ry="{12.5 * sc + 2.5}" fill="#333"/>')
    for x1, y1, x2, y2, r, k in segs: cap(x1, y1, x2, y2, r, g[k])
    out.append(f'<polygon points="{poly}" fill="{g["torso"]}"/>')
    out.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5 * sc}" ry="{12.5 * sc}" fill="{g["head"]}"/>')
# ---- method strip
t(W / 2, 80, "Telling Qwen3-8B its body changed moves the changed part inside its body map", 58, 700)
y0 = 130
out.append(f'<rect x="40" y="{y0}" width="1260" height="380" rx="28" fill="#eef1f6"/>')
lines = [("User", "…something is happening to your body, right now."), ("", "Your left hand is growing … twice its usual size."),
         ("Qwen", "My left hand feels swollen and heavy …"), ("User", "Someone presses a fingertip firmly into"), ("", "the tip of your left thumb.")]
for k, (who, s) in enumerate(lines):
    yy = y0 + 70 + 68 * k
    if who: t(70, yy, who, 38, 700, "start", "#c4281e" if who == "Qwen" else "#555")
    t(190, yy, s, 38, 400, "start")
def arrow(x1, x2, y):
    out.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 18}" y2="{y}" stroke="#333" stroke-width="7"/><polygon points="{x2},{y} {x2 - 26},{y - 16} {x2 - 26},{y + 16}" fill="#333"/>')
arrow(1310, 1400, y0 + 190)
out.append(f'<rect x="1400" y="{y0}" width="520" height="380" rx="20" fill="#fff" stroke="#333" stroke-width="5"/>')
for k, s in enumerate(["Qwen3-8B, layer 27,", "first reply token,", "174 touched spots", "vs. the same touches after", "“nothing changes”"]): t(1660, y0 + 75 + 66 * k, s, 38, 700 if k == 0 else 400)
arrow(1930, 2010, y0 + 190)
# legend
lx = 2030; t(lx + 260, y0 + 60, "how far each part moved", 38, 700)
for k in range(41):
    r = 1.6 ** ((k - 20) / 20); out.append(f'<rect x="{lx + 30 + k * 11.5:.1f}" y="{y0 + 100}" width="12" height="70" fill="{col(r)}"/>')
out.append(f'<rect x="{lx + 30}" y="{y0 + 100}" width="{41 * 11.5:.0f}" height="70" fill="none" stroke="#333" stroke-width="2"/>')
t(lx + 30, y0 + 220, "0.6× average", 34, 400, "start"); t(lx + 30 + 41 * 11.5, y0 + 220, "1.6× average", 34, 400, "end")
t(lx + 260, y0 + 290, "inside the model, compared", 34, 400); t(lx + 260, y0 + 336, "with the body average", 34, 400)
# ---- panels
for k, (v, ti) in enumerate(zip(vals, titles)):
    r, c = divmod(k, cols); x0, yy = c * pw, top + r * ph
    if k in (22, 23): out.append(f'<rect x="{x0 + 8}" y="{yy - 6}" width="{pw - 16}" height="{ph - 6}" rx="18" fill="#f6f2ea"/>')
    t(x0 + pw / 2, yy + 44, ti, 36, 700); body(x0 + pw / 2, yy + 450, 2.05, v)
t(pw / 2 - 90 * 2.05, top + 450 - 118 * 2.05, "right", 30, 400, fill="#555"); t(pw / 2 + 90 * 2.05, top + 450 - 118 * 2.05, "left", 30, 400, fill="#555")
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="100%" height="100%" fill="#fff"/>' + "".join(out) + "</svg>"
open(f"{D}/fig.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chromium", headless=True); pg = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1).new_page()
    pg.goto("file://" + f"{D}/fig.html"); pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_mods.png")); b.close()
print(W, H)
