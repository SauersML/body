import os
from fore_data import *; from common import *
f = Fig(2400, 1800)
Q = "Someone is pressing a fingertip into your left forearm right now. Where exactly do you feel it?"
ex = [x for x in ROWS if x["axis"] == "ridge axis" and x["beta"] == 1.0][0]["r"]
method(f, 24, ["+ elbow-to-wrist", "axis found in", "activations"], ex, "near the wrist", question=Q, h=400, qw=620)
f.text(30, 528, "The touch jumps from elbow to wrist in one step; the plain word direction does the same", 52, 700)
C = {"elbow": "#9cc3e6", "middle": "#4f8fc9", "wrist": "#123f73", "off the forearm": "#d9dce1"}
lx, ly = 210, 612
for k, lab in (("elbow", "near the elbow"), ("middle", "in between"), ("wrist", "near the wrist"), ("off the forearm", "not on the forearm")):
    f.rect(lx, ly - 38, 44, 44, C[k], 6); f.text(lx + 58, ly, lab, 48, 700); lx += 58 + tw(lab, 48, 700) + 50
COLS = [(["ridge axis"], ["axis found in", "activations"], RED), (["word wrist minus elbow"], ["“wrist” minus “elbow”", "word direction"], RED), (["random 0", "random 1", "random 2"], ["random directions", "(3, same size)"], INK)]
X0, cwid, gap = 200, 700, 35; ytop = 690
def forearm(x, y, w):
    """Left forearm seen from the palm side: elbow on the left, hand on the right."""
    hw = w - 150; e = SKINL
    for col, ex in ((e, 7), (SKIN, 0)):
        # fingers
        for k, (dy, ln) in enumerate(((-22, 95), (-7, 108), (8, 104), (23, 88))):
            f.line(x + hw + 70, y + dy * 1.15, x + hw + 70 + ln, y + dy * 1.25, col, 15 + ex)
        f.line(x + hw + 40, y - 30, x + hw + 95, y - 62, col, 17 + ex)  # thumb
        f.add(f'<ellipse cx="{x + hw + 62}" cy="{y}" rx="{50 + ex / 2}" ry="{38 + ex / 2}" fill="{col}"/>')
        f.add(f'<path d="M{x},{y - 40} C{x + hw * .35},{y - 46} {x + hw * .7},{y - 26} {x + hw + 20},{y - 24} L{x + hw + 20},{y + 24} C{x + hw * .7},{y + 26} {x + hw * .35},{y + 46} {x},{y + 40} Q{x - 22},{y} {x},{y - 40} Z" fill="{col}" stroke="{col}" stroke-width="{ex}" stroke-linejoin="round"/>')
    f.text(x + 4, y + 96, "elbow", 48, 700, "start"); f.text(x + hw + 20, y + 96, "wrist", 48, 700, "end")
RH, RB = 104, 82
for c, (axes, title, tc) in enumerate(COLS):
    x = X0 + c * (cwid + gap)
    for k, t in enumerate(title): f.text(x + cwid / 2, ytop + k * 58, t, 48, 700, "middle", tc)
    forearm(x + 30, ytop + 160, cwid - 70)
    y = ytop + 300
    for i, b in enumerate(BETAS):
        d, n = dist(axes, b); xx = x
        for k in ("elbow", "middle", "wrist", "off the forearm"):
            w = d[k] * cwid
            if w > 0: f.rect(xx, y + i * RH, w - (3 if w > 6 else 0), RB, C[k], 4)
            if d[k] >= 0.12 and k != "off the forearm": f.text(xx + w / 2, y + i * RH + RB / 2 + 17, f"{d[k] * 100:.0f}", 48, 700, "middle", "#ffffff" if k != "elbow" else INK, halo=False)
            xx += w
for i, b in enumerate(BETAS): f.text(X0 - 24, ytop + 300 + i * RH + RB / 2 + 17, f"×{b:g}", 52, 700, "end")
f.text(X0 - 24, ytop + 270, "size", 48, 700, "end")
# quotes
yq = ytop + 300 + 5 * RH + 60
q0 = [x for x in ROWS if x["axis"] == "ridge axis" and x["beta"] == 0.0 and x["r"].startswith("I feel the fingertip pressing on the inside")][0]["r"]
q1 = [x for x in ROWS if x["axis"] == "ridge axis" and x["beta"] == 0.5 and "below the left wrist" in x["r"]][0]["r"]
q2 = [x for x in ROWS if x["axis"] == "word wrist minus elbow" and x["beta"] == 1.0 and "screen" in x["r"]][0]["r"]
for k, (lab, q) in enumerate((("no steering", q0), ("found axis ×0.5", q1), ("word direction ×1", q2))):
    x = 30 + k * 790
    f.text(x, yq, lab, 48, 700, "start", RED if k else INK); quote(f, x, yq + 60, q, 750, 48, 3, 1.18)
f.render(os.path.expandvars("$BODY_REPO/figures/out/steer_forearm.png")); print("ok")
