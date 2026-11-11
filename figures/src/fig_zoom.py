import os
import sys; sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/src"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from common import *
from scipy.stats import spearmanr, rankdata
Z = states()
REG["hand"] = [i for i in REG["hand"]]
def rho(key, ix):
    iu = np.triu_indices(len(ix), 1); return spearmanr(squareform(pdist(Z[key][ix]))[iu], squareform(pdist(P[ix]))[iu])[0]
def colours(ix):
    G = Z["32B"][ix]; G = G - G.mean(0); u, s, vt = np.linalg.svd(G, full_matrices=False); c = u[:, :2] * s[:2]
    a = (rankdata(c[:, 0]) - 1) / (len(ix) - 1); b = (rankdata(c[:, 1]) - 1) / (len(ix) - 1)
    C00, C10, C01, C11 = np.array([22, 120, 190]), np.array([235, 90, 30]), np.array([60, 170, 80]), np.array([150, 40, 170])
    cols = [(1-x)*(1-y)*C00 + x*(1-y)*C10 + (1-x)*y*C01 + x*y*C11 for x, y in zip(a, b)]
    return ["#%02x%02x%02x" % tuple(int(v) for v in c_) for c_ in cols]
def nn1(ix):
    D = squareform(pdist(Z["32B"][ix])); np.fill_diagonal(D, np.inf); return np.argmin(D, 1)
W, H = 2400, 1960; out = []
T(out, 40, 76, "Up close, Qwen3-32B maps face and foot better than its input words; hand and forearm worse", 52, 700)
T(out, 40, 150, "Dot = a touched spot. Colour = the 2 strongest activation directions inside that body part", 48)
T(out, 40, 212, "(no position labels used). Line = joins each spot to its most similar spot.", 48)
T(out, 40, 274, "Bars = how well distances between spots in the model match distances on the skin (rank correlation).", 48)
SK = "#e9e9e9"
def panel(px, py, title, ix, proj, draw, sc):
    T(out, px, py + 60, title, 60, 700)
    cols = colours(ix); nn = nn1(ix); Q = np.array([proj(i) for i in ix])
    draw(sc)
    for k, i in enumerate(ix):
        j = nn[k]; out.append(f'<line x1="{Q[k,0]:.1f}" y1="{Q[k,1]:.1f}" x2="{Q[j,0]:.1f}" y2="{Q[j,1]:.1f}" stroke="#222" stroke-width="4" opacity="0.7"/>')
    for k, i in enumerate(ix): out.append(f'<circle cx="{Q[k,0]:.1f}" cy="{Q[k,1]:.1f}" r="15" fill="{cols[k]}" stroke="#fff" stroke-width="3"/>')
def bars(px, py, ix):
    r32, rw = rho("32B", ix), rho("words", ix)
    for k, (nm, v, col) in enumerate([("Qwen3-32B", r32, "#1864ab"), ("input words", rw, "#9a9a9a")]):
        y = py + k * 72; w = 560 * max(v, 0) / 0.6
        T(out, px + 270, y + 46, nm, 48, 700 if k == 0 else 400, "end", col if k == 0 else "#111")
        out.append(f'<rect x="{px+290}" y="{y+6}" width="{w:.0f}" height="56" fill="{col}"/>'); T(out, px + 300 + w, y + 48, f"{v:.2f}", 48, 700)
    return r32, rw
res = {}
# FACE
ix = REG["face"]; fx, fy, s = 680, 935, 25
proj = lambda i: (fx + P[i, 0] * s, fy - (P[i, 1] - 150) * s)
def dface(sc):
    out.append(f'<ellipse cx="{fx}" cy="{fy - 11.5*s}" rx="{11*s}" ry="{13.2*s}" fill="{SK}"/>')
    for e in (-1, 1): out.append(f'<ellipse cx="{fx + e*3.6*s}" cy="{fy - 16*s}" rx="{1.8*s}" ry="{0.8*s}" fill="#fff"/>')
panel(60, 290, "face", ix, proj, dface, s); res["face"] = bars(60, 980, ix)
# FOOT (side view from the inner side, toes to the right)
ix = REG["foot"]; fx, fy, s = 1500, 900, 27
proj = lambda i: (fx + (P[i, 2] + 6) * s + (0.7*s if P[i, 0] > 14 else 0), fy - P[i, 1] * s * 1.25)
def dfoot(sc):
    pts = [(-5, 2), (-4, 0), (14, 0), (21.5, 0.5), (22, 2.5), (17, 4.5), (10, 7.5), (5, 10), (4, 14), (-3, 14), (-3.5, 9), (-5.5, 5)]
    out.append('<polygon points="' + " ".join(f"{fx + (z+6)*s:.0f},{fy - y*s*1.25:.0f}" for z, y in pts) + f'" fill="{SK}" stroke="{SK}" stroke-width="30" stroke-linejoin="round"/>')
panel(1300, 290, "foot", ix, proj, dfoot, s); res["foot"] = bars(1300, 980, ix)
# HAND: palm-side view and back-side view
ix = REG["hand"]; s = 19
hx = {True: 280, False: 860}; hy = 1500
def hp(i):
    back = L[i]["n"][2] < -0.3; return (hx[not back] - 84 * s + P[i, 0] * s if False else hx[not back] + (P[i, 0] - 84) * s, hy - (P[i, 1] - 141) * s)
def dhand(sc):
    for palm in (True, False):
        X = lambda x: hx[palm] + (x - 84) * s; Y = lambda y: hy - (y - 141) * s
        def cap(x1, y1, x2, y2, r): out.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{SK}" stroke-width="{2*r*s:.1f}" stroke-linecap="round"/>')
        out.append(f'<rect x="{X(75):.1f}" y="{Y(148.5):.1f}" width="{16*s:.1f}" height="{15*s:.1f}" rx="{4*s:.1f}" fill="{SK}"/>')
        for y, tip in [(146, 99.5), (142.5, 100.5), (139, 99.5), (135.5, 97.5)]: cap(89, y, tip, y, 1.6)
        cap(82, 146, 87, 157.5, 1.9)
        T(out, X(87), Y(129) , "palm side" if palm else "back side", 48, 400, "middle")
panel(60, 1120, "hand", ix, hp, dhand, s); res["hand"] = bars(60, 1790, ix)
# FOREARM unrolled
ix = REG["forearm"]
def sd(i):
    p = L[i]["phrase"]; return 0 if "the inside" in p else 1 if "thumb side" in p else 2 if "the back" in p else 3
xs = np.abs(P[ix, 0]); a0, a1 = xs.min(), xs.max(); rx0, rx1, ry0, rh = 1330, 2330, 1190, 110
fp = lambda i: (rx0 + 40 + (abs(P[i, 0]) - a0) / (a1 - a0) * (rx1 - rx0 - 80), ry0 + rh/2 + sd(i) * rh)
def dfa(sc):
    out.append(f'<rect x="{rx0}" y="{ry0}" width="{rx1-rx0}" height="{4*rh}" fill="{SK}" rx="14"/>')
    for k in range(1, 4): out.append(f'<line x1="{rx0}" y1="{ry0+k*rh}" x2="{rx1}" y2="{ry0+k*rh}" stroke="#fff" stroke-width="4"/>')
    T(out, rx0, ry0 + 4*rh + 60, "elbow", 48, 700); T(out, rx1, ry0 + 4*rh + 60, "wrist", 48, 700, "end")
    T(out, (rx0+rx1)/2, ry0 + 4*rh + 60, "4 sides, unrolled", 48, 400, "middle")
panel(1300, 1120, "forearm", ix, fp, dfa, 1); res["forearm"] = bars(1300, 1790, ix)
render(out, W, H, os.path.expandvars("$BODY_REPO/figures/out/body_zoomins.png"), "zoom")
print({k: tuple(round(float(x), 3) for x in v) for k, v in res.items()})
