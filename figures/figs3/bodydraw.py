import numpy as np
def col(r, lim=1.45):
    x = float(np.clip(np.log(r) / np.log(lim), -1, 1)); a = np.array([255, 255, 255]); hi = np.array([196, 40, 30]); lo = np.array([60, 110, 180])
    c = a + (hi - a) * x if x > 0 else a + (lo - a) * (-x); return "#%02x%02x%02x" % tuple(int(v) for v in c)
def body(s, cx, yb, sc, g, lim=1.45):
    """front view, person's left on viewer's right; g: dict group->ratio"""
    X = lambda x: cx + x * sc; Y = lambda y: yb - y * sc
    gc = {k: col(v, lim) for k, v in g.items()}
    segs = []
    for s_, L_ in ((1, "L"), (-1, "R")):
        segs += [(s_ * 18, 143, s_ * 50, 143, 5.5, "arm" + L_), (s_ * 50, 143, s_ * 77, 143, 4.5, "arm" + L_), (s_ * 82, 141, s_ * 96, 141, 7.5, "hand" + L_),
                 (s_ * 11, 90, s_ * 12, 50, 8.5, "leg" + L_), (s_ * 12, 50, s_ * 13, 9, 6, "leg" + L_), (s_ * 9.5, 3, s_ * 18.5, 3, 4, "foot" + L_)]
    poly = f"{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}"
    for x1, y1, x2, y2, r, k in segs: s.add(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="#333" stroke-width="{2*r*sc+5:.1f}" stroke-linecap="round"/>')
    s.add(f'<polygon points="{poly}" fill="#333" stroke="#333" stroke-width="5" stroke-linejoin="round"/>')
    s.add(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5*sc+2.5}" ry="{12.5*sc+2.5}" fill="#333"/>')
    for x1, y1, x2, y2, r, k in segs: s.add(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{gc[k]}" stroke-width="{2*r*sc:.1f}" stroke-linecap="round"/>')
    s.add(f'<polygon points="{poly}" fill="{gc["torso"]}"/>')
    s.add(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5*sc}" ry="{12.5*sc}" fill="{gc["head"]}"/>')
