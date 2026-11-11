import os
import sys; sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/src"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from common import *
Z = states()
SC = ["#2b8a3e", "#e67700", "#5f3dc4", "#c2255c"]; SN = ["inside (palm side)", "thumb side", "back", "little-finger side"]
def side(l):
    p = l["phrase"]; return 0 if "the inside" in p else 1 if "thumb side" in p else 2 if "the back" in p else 3
fa = np.array([i for i, l in enumerate(L) if "left forearm" in l["phrase"]]); sd = np.array([side(L[i]) for i in fa]); x = np.abs(P[fa, 0])
W, H = 2400, 1880; out = []
T(out, 40, 80, "Qwen3 groups forearm spots by side of the arm, not by position along the arm", 60, 700)
# method line
T(out, 40, 160, "40 touched spots on the left forearm: 4 sides &#215; 10 steps from elbow to wrist,", 48)
T(out, 40, 222, "e.g. &#8220;the back of your left forearm, two finger-widths above the wrist&#8221;.", 48)
T(out, 40, 284, "Each spot is joined to its most similar spot (activation at the first reply word).", 48)
x0, x1 = 560, 1640; rh = 78
colx = 1700
T(out, colx + 150, 380, "same side", 48, 700, "middle"); T(out, colx + 500, 380, "apart along", 48, 700, "middle"); T(out, colx + 500, 432, "the arm", 48, 700, "middle")
T(out, colx + 150, 432, "of the arm", 48, 700, "middle")
panels = [("input word vectors", "words"), ("Qwen3-8B", "8B"), ("Qwen3-32B", "32B")]
y = 470
rnd_side = np.mean([(np.sum(sd == sd[i]) - 1) / 39 for i in range(40)])
rnd_gap = np.mean([abs(x[i] - x[j]) for i in range(40) for j in range(40) if i != j])
for name, key in panels:
    T(out, x0, y + 52, name, 56, 700)
    top = y + 90
    X = lambda v: x0 + 30 + (v - x.min()) / (x.max() - x.min()) * (x1 - x0 - 60)
    Yr = lambda s: top + rh / 2 + s * rh
    out.append(f'<rect x="{x0}" y="{top}" width="{x1-x0}" height="{4*rh}" fill="#f1f3f5" rx="10"/>')
    for s in range(4):
        out.append(f'<rect x="{x0}" y="{top + s*rh + 4}" width="{x1-x0}" height="{rh-8}" fill="{SC[s]}" opacity="0.10" rx="8"/>')
        T(out, x0 - 24, Yr(s) + 17, SN[s], 48, 400, "end", SC[s])
    D = squareform(pdist(Z[key][fa])); np.fill_diagonal(D, np.inf); nn = np.argmin(D, 1)
    for i in range(40):
        j = nn[i]; xa, ya, xb, yb_ = X(x[i]), Yr(sd[i]), X(x[j]), Yr(sd[j])
        if sd[i] == sd[j]:
            bend = -min(70, 0.22 * abs(xb - xa)) * (1 if (i % 2 == 0) else -1)
            mx, my = (xa + xb) / 2, ya + bend
        else:
            mx, my = (xa + xb) / 2 + 20, (ya + yb_) / 2
        out.append(f'<path d="M{xa:.1f},{ya:.1f} Q{mx:.1f},{my:.1f} {xb:.1f},{yb_:.1f}" fill="none" stroke="#222" stroke-width="4.5" opacity="0.75"/>')
    for i in range(40):
        out.append(f'<circle cx="{X(x[i]):.1f}" cy="{Yr(sd[i]):.1f}" r="15" fill="{SC[sd[i]]}" stroke="#fff" stroke-width="3"/>')
    same = np.mean(sd[nn] == sd); gap = np.mean(np.abs(x[nn] - x))
    T(out, colx + 150, top + 2 * rh + 30, f"{same*100:.0f}%", 84, 700, "middle")
    T(out, colx + 500, top + 2 * rh + 30, f"{gap:.0f} cm", 84, 700, "middle")
    print(name, round(same, 3), round(gap, 2))
    y += 4 * rh + 130
T(out, x0, y + 20, "elbow", 48, 700, "start"); T(out, x1, y + 20, "wrist", 48, 700, "end")
out.append(f'<line x1="{x0+170}" y1="{y+4}" x2="{x1-150}" y2="{y+4}" stroke="#222" stroke-width="5"/><polygon points="{x1-140},{y+4} {x1-166},{y-11} {x1-166},{y+19}" fill="#222"/>')
T(out, colx + 150, y + 20, f"random {rnd_side*100:.0f}%", 48, 400, "middle"); T(out, colx + 500, y + 20, f"random {rnd_gap:.0f} cm", 48, 400, "middle")
render(out, W, y + 60, os.path.expandvars("$BODY_REPO/figures/out/body_forearm_strands.png"), "fa")
print("H", y + 60, rnd_side, rnd_gap)
