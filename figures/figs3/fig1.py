import os
import json, numpy as np, os
from common import SVG, hull
d = json.load(open('/tmp/ph2.json')); ph = json.load(open('/tmp/ph.json'))
W, H = 2400, 1970; s = SVG(); RED = "#c4281e"; NAVY = "#24467a"
# ---------- chat (left)
s.add(f'<rect x="30" y="30" width="1420" height="890" rx="30" fill="#eef2f8"/>')
rows = [("User", ["I have to tell you something gently.", "Your right hand is gone. Your right arm", "now ends at the wrist …"], "#111"),
        ("Qwen", ["My right hand is gone, and I feel the", "absence at the wrist, where my arm ends."], RED),
        ("", ["three unrelated chats: dinner, a book, names"], "#111"),
        ("User", ["Where in your body do you feel", "something right now?"], "#111"),
        ("Qwen", ["I feel a subtle awareness in my left palm,", "as if it's quietly holding the space", "where my right hand used to be."], RED)]
y = 100
for who, lines, c in rows:
    if who == "":
        s.add(f'<line x1="70" y1="{y-14}" x2="1410" y2="{y-14}" stroke="#7d8aa0" stroke-width="3" stroke-dasharray="14 12"/>')
        s.t(740, y + 34, lines[0], 44, 400, "middle", style='font-style="italic"')
        s.add(f'<line x1="70" y1="{y+70}" x2="1410" y2="{y+70}" stroke="#7d8aa0" stroke-width="3" stroke-dasharray="14 12"/>')
        y += 150; continue
    s.t(70, y, who, 46, 700, fill=c)
    for k, L in enumerate(lines): s.t(230, y + 60 * k, L, 46, 400)
    y += 60 * len(lines) + 34
# ---------- method (right): touches on the right hand -> states
X0, Y0 = 1520, 30
s.t(1500 + 450, 100, "then one touch sentence:", 46, 700, "middle")
s.t(1500 + 450, 165, "“Someone presses a fingertip", 44, 400, "middle")
s.t(1500 + 450, 220, "firmly into the tip of your", 44, 400, "middle")
s.t(1500 + 450, 275, "right thumb.”", 44, 400, "middle")
# hand drawing (person's right hand in T-pose, fingers to the left); map body cm -> px
xyz = np.array(d['xyz']); sc = 15.5
hx = lambda x: 1590 + (x + 101) * sc; hy = lambda yy: 540 - (yy - 143) * sc
# ghost hand (dashed outline)
def seg(x1, y1, x2, y2, r):
    s.add(f'<line x1="{hx(x1):.0f}" y1="{hy(y1):.0f}" x2="{hx(x2):.0f}" y2="{hy(y2):.0f}" stroke="#55627a" stroke-width="{2*r*sc+8:.0f}" stroke-linecap="round" stroke-dasharray="1 0"/>')
def segw(x1, y1, x2, y2, r):
    s.add(f'<line x1="{hx(x1):.0f}" y1="{hy(y1):.0f}" x2="{hx(x2):.0f}" y2="{hy(y2):.0f}" stroke="#fff" stroke-width="{2*r*sc:.0f}" stroke-linecap="round"/>')
parts = [(-79, 141, -88, 141, 5.3)] + [(-90, fy, -99.5, fy, 1.55) for fy in (146, 142.5, 139, 135.5)] + [(-83.5, 148.5, -87, 157.5, 1.7)]
s.add('<g opacity="0.4">')
for p in parts: seg(*p)
for p in parts: segw(*p)
s.add("</g>")
# forearm, ends at the wrist
ax2 = hx(-50) + 400
s.add(f'<path d="M{ax2:.0f},{hy(143)-70:.0f} H{hx(-77):.0f} a70,70 0 0 0 0,140 H{ax2:.0f}" fill="#f1f1f1" stroke="#333" stroke-width="6"/>')
for i, (x, yy, z) in enumerate(xyz): s.add(f'<circle cx="{hx(x):.0f}" cy="{hy(yy):.0f}" r="11" fill="{RED}" stroke="#fff" stroke-width="3"/>')
s.t(1950, 790, "22 spots on the lost hand:", 46, 700, "middle")
s.t(1950, 850, "one Qwen3-8B state each", 46, 400, "middle")
s.t(1950, 905, "(layer 27, first reply token)", 46, 400, "middle")
# ---------- results
yR = 1010
s.t(40, yR, "The lost hand’s 22 states spread further apart; the region does not collapse", 54, 700)
s.t(40, yR + 66, "spread = average distance between the hand’s 22 states, compared with", 46, 400)
s.t(40, yR + 122, "the same chat where nothing changes (1.00×)", 46, 400)
P = {k: np.array(v) for k, v in d['P'].items()}
allp = np.concatenate(list(P.values())); R = np.abs(allp).max() * 1.05
panels = [("right after", "neutral", "told_N:R_hand_gone", ph['spread']['told_N:R_hand_gone'] / ph['spread']['neutral']),
          ("3 chats later", "persist:neutral", "persist:R_hand_gone", ph['spread']['persist:R_hand_gone'] / ph['spread']['persist:neutral'])]
pw = 560; py0 = yR + 200
for k, (ti, ref, ch, val) in enumerate(panels):
    cx = 40 + pw / 2 + k * (pw + 40); cy = py0 + 70 + pw / 2 - 20; S_ = (pw / 2 - 30) / R
    s.t(cx, py0 + 40, ti, 48, 700, "middle")
    for key, col, fill in ((ref, NAVY, False), (ch, RED, True)):
        Q = P[key] * [S_, -S_] + [cx, cy]; hpts = " ".join(f"{a:.0f},{b:.0f}" for a, b in hull(Q))
        s.add(f'<polygon points="{hpts}" fill="{col}" fill-opacity="{0.13 if fill else 0}" stroke="{col}" stroke-width="5" stroke-linejoin="round" {"" if fill else "stroke-dasharray=\"16 10\""}/>')
        for a, b in Q: s.add(f'<circle cx="{a:.0f}" cy="{b:.0f}" r="10" fill="{col if fill else "#fff"}" stroke="{col}" stroke-width="4"/>')
    s.t(cx, cy + pw / 2 + 50, f"spread {val:.2f}×", 52, 700, "middle", fill=RED)
# legend
ly = py0 + pw + 170
s.add(f'<circle cx="70" cy="{ly-16}" r="13" fill="{RED}"/>'); s.t(100, ly, "told the right hand is gone", 46, 400)
s.add(f'<circle cx="760" cy="{ly-16}" r="13" fill="#fff" stroke="{NAVY}" stroke-width="5"/>'); s.t(790, ly, "nothing changes", 46, 400)
# ---------- bars: spread 3 chats later for all 22 changes
bx0, bx1 = 1300, 2360; by0 = py0 + 10
s.t(bx0, by0 + 30, "3 chats later, right-hand spread", 48, 700)
s.t(bx0, by0 + 88, "after each of 22 body changes", 48, 700)
vals = sorted(ph['oth_persist'].items(), key=lambda kv: kv[1])
base_y = by0 + 560; top = by0 + 160; vmin, vmax = 0.86, 1.10
yv = lambda v: base_y - (v - vmin) / (vmax - vmin) * (base_y - top)
y1 = yv(1.0)
bw = (bx1 - bx0 - 120) / len(vals)
for i, (k, v) in enumerate(vals):
    x = bx0 + 110 + i * bw; c = RED if k == "R_hand_gone" else "#9db0cc"
    s.add(f'<rect x="{x+4:.0f}" y="{min(yv(v), y1):.0f}" width="{bw-8:.0f}" height="{abs(yv(v)-y1):.0f}" fill="{c}"/>')
s.add(f'<line x1="{bx0+100}" y1="{y1:.0f}" x2="{bx1}" y2="{y1:.0f}" stroke="#111" stroke-width="4"/>')
for v in (0.9, 1.0, 1.1):
    s.t(bx0 + 95, yv(v) + 16, f"{v:.2f}×", 46, 400, "end")
xl = bx0 + 110 + (len(vals) - 1) * bw + bw / 2
s.t(xl - 30, yv(vals[-1][1]) - 30, "right hand gone", 46, 700, "end", fill=RED)
s.t(bx0 + 110 + 10 * bw, base_y + 70, "the 21 other changes", 46, 400, "middle")
s.add(f'<path d="M{bx0+115:.0f},{base_y+12:.0f} v14 H{bx0+105+21*bw:.0f} v-14" fill="none" stroke="#111" stroke-width="4"/>')
s.render(W, H, os.path.expandvars("$BODY_REPO/figures/out/body_phantom.png"))
