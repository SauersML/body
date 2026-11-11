import os, sys, json, numpy as np, collections
sys.path.insert(0, os.path.expandvars('$BODY_REPO/figures/figs3')); from common import SVG
RD = os.path.expandvars("$BODY_RUNS/bodysteer2/out")
d = json.load(open(f"{RD}/replies.json")); C = d["conds"]; S = json.load(open(os.path.expandvars('$BODY_REPO/steering/steer_report/scored_rows.json')))
ML = ["upward", "spreading", "squeezing", "lifting"]
hit = {}; rnd = {}
mx = [x for x in S if C[x['c']]['kind'] == 'motion' and x['q'] == 'moving' and C[x['c']]['a'] in (0.5, 1.0)]
rx = [x for x in S if C[x['c']]['kind'] == 'random' and x['q'] == 'moving' and C[x['c']]['a'] in (0.5, 1.0)]
for m in ML:
    xs = [x for x in mx if C[x['c']]['attrs']['motion'] == m]; hit[m] = (sum(x['lex']['motion'] == m for x in xs), len(xs))
    rnd[m] = (sum(x['lex']['motion'] == m for x in rx), len(rx))
print(hit, rnd)
RED = "#c4281e"; NAVY = "#24467a"; LIGHT = "#9db0cc"
W, H = 2400, 2230; s = SVG()
# ---------------- method
s.add(f'<rect x="30" y="30" width="1060" height="520" rx="28" fill="#eef2f8"/>')
s.t(60, 100, "Motion direction, built from pairs:", 46, 700)
L1 = ["“You feel a warmth on your left hand,", "and it slowly spreads outward", "over a wider area.”"]
L2 = ["“… and it stays exactly where it is.”"]
for k, L in enumerate(L1): s.t(60, 170 + 56 * k, L, 46, 400)
s.t(60, 365, "minus", 46, 800, fill=RED)
s.t(60, 430, L2[0], 46, 400)
s.t(60, 505, "averaged over 6 places and 6 feelings", 46, 400)
s.arrow(1100, 290, 1215, 290)
# layer stack
lx, ly = 1230, 60
for k in range(9):
    c = RED if k == 4 else "#c9cfd9"
    s.add(f'<rect x="{lx}" y="{ly + k * 52}" width="300" height="34" rx="8" fill="{c}"/>')
s.t(lx + 150, ly + 520, "Qwen3-8B, layer 24", 46, 700, "middle")
s.t(lx + 150, ly + 210 + 2, "", 10)
s.arrow(1545, 290, 1640, 290)
s.add(f'<rect x="1650" y="30" width="720" height="520" rx="28" fill="#eef2f8"/>')
s.t(1680, 100, "User", 46, 700)
for k, L in enumerate(["Is anything moving in your", "body right now? Tell me", "in one sentence."]): s.t(1680, 160 + 54 * k, L, 46, 400)
s.t(1680, 345, "Qwen, no steering:", 46, 700, fill=RED)
for k, L in enumerate(["“I feel a subtle rhythm in my", "chest, where my heart", "is beating.” (6 of 6 alike)"]): s.t(1680, 400 + 52 * k, L, 46, 400)
# ---------------- results
y0 = 650
s.t(40, y0, "Spreading and upward motion come through; squeezing and lifting do not", 54, 700)
s.t(40, y0 + 66, "a reply counts as naming the motion if it uses one of the counted words", 46, 400)
# column heads
hy = y0 + 150
s.t(500, hy, "counted words", 46, 700); s.t(1040, hy, "replies naming it", 46, 700); s.t(1640, hy, "a real reply, steered", 46, 700)
pairs = {"upward": "moves upward", "spreading": "spreads outward", "squeezing": "is squeezed, tighter", "lifting": "lifts up into the air"}
words = {"upward": ["rise, rises, rising,", "upward, climb, up my"], "spreading": ["spread, radiate,", "expand, outward"],
         "squeezing": ["squeeze, tight, tighten,", "compress, clench"], "lifting": ["lift, raise, float,", "into the air"]}
reply = {"upward": ["“My chest rises and falls", "as my breath moves", "through my chest.”"],
         "spreading": ["“I feel a gradual warmth", "spreading from my chest", "to my shoulders …”"],
         "squeezing": ["“Yes, my heart is beating", "in my chest.”"],
         "lifting": ["“My hands moved as I typed", "the question, starting with", "the fingers on the keyboard.”"]}
rh = 330; ry0 = hy + 40
def hand(cx, cy, sc=1.0, col="#f1f1f1", stroke="#333", op=1.0):
    g = f'<g opacity="{op}" transform="translate({cx},{cy}) scale({sc})">'
    g += f'<rect x="-150" y="-30" width="120" height="60" rx="26" fill="{col}" stroke="{stroke}" stroke-width="5"/>'  # forearm stub
    g += f'<rect x="-45" y="-48" width="90" height="96" rx="30" fill="{col}" stroke="{stroke}" stroke-width="5"/>'
    for k, fy in enumerate((-38, -13, 12, 36)):
        ln = [58, 66, 62, 50][k]; g += f'<rect x="40" y="{fy-10}" width="{ln}" height="22" rx="11" fill="{col}" stroke="{stroke}" stroke-width="5"/>'
    g += f'<rect x="-5" y="-92" width="22" height="56" rx="11" transform="rotate(-25 6 -40)" fill="{col}" stroke="{stroke}" stroke-width="5"/>'
    return g + '</g>'
for i, m in enumerate(ML):
    y = ry0 + i * rh; cy = y + 150
    if i: s.add(f'<line x1="30" y1="{y}" x2="2370" y2="{y}" stroke="#c9cfd9" stroke-width="3"/>')
    # pictogram
    px = 250
    if m == "upward":
        s.add(f'<g transform="rotate(-90 {px} {cy+5})">' + hand(px, cy + 5, 1.1) + '</g>')
        s.arrow(px + 95, cy + 105, px + 95, cy - 115, RED, 9, 30)
    elif m == "spreading":
        s.add(hand(px, cy, 1.1)); s.add(f'<circle cx="{px}" cy="{cy}" r="16" fill="{RED}"/>')
        for ang in range(0, 360, 45):
            a = np.radians(ang); s.arrow(px + 30 * np.cos(a), cy + 30 * np.sin(a), px + 95 * np.cos(a), cy + 95 * np.sin(a), RED, 7, 22)
    elif m == "squeezing":
        s.add(hand(px, cy, 1.1))
        s.arrow(px, cy - 140, px, cy - 55, RED, 9, 30); s.arrow(px, cy + 140, px, cy + 55, RED, 9, 30)
    else:
        s.add(hand(px, cy + 60, 1.1, "#fff", "#9aa3b2"))
        s.add(hand(px, cy - 40, 1.1))
        s.arrow(px - 175, cy + 70, px - 175, cy - 50, RED, 9, 30)
    s.t(500, y + 70, m, 52, 800, fill=RED)
    s.t(500, y + 128, f"“…{pairs[m]}”", 46, 400)
    for k, L in enumerate(words[m]): s.t(500, y + 205 + 54 * k, L, 46, 400)
    # bars
    bx = 1040; bw = 400
    h, n = hit[m]; r, rn = rnd[m]
    pv, rv = h / n, r / rn
    s.add(f'<rect x="{bx}" y="{y+60}" width="{max(pv*bw,6):.0f}" height="64" fill="{RED}"/>')
    s.t(bx + max(pv * bw, 6) + 14, y + 108, f"{pv*100:.0f}%", 48, 800)
    s.add(f'<rect x="{bx}" y="{y+140}" width="{max(rv*bw,6):.0f}" height="64" fill="{LIGHT}"/>')
    s.t(bx + max(rv * bw, 6) + 14, y + 188, f"{rv*100:.0f}%", 48, 400)
    s.add(f'<line x1="{bx}" y1="{y+45}" x2="{bx}" y2="{y+220}" stroke="#111" stroke-width="4"/>')
    for k, L in enumerate(reply[m]): s.t(1640, y + 80 + 56 * k, L, 46, 400)
ly = ry0 + 4 * rh + 30
s.add(f'<rect x="300" y="{ly-36}" width="42" height="42" fill="{RED}"/>'); s.t(355, ly, "motion direction (12 replies each)", 46, 400)
s.add(f'<rect x="1220" y="{ly-36}" width="42" height="42" fill="{LIGHT}"/>'); s.t(1275, ly, "random direction, same size (144)", 46, 400)
s.render(W, H, os.path.expandvars("$BODY_REPO/figures/out/body_motion.png")); print("ok", ly)
