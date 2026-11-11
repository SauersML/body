"""partial chain rho (lexical distance regressed out) and within-region nearest human parts."""
import json, sys, itertools, numpy as np
from scipy.stats import spearmanr, rankdata
sys.argv = [sys.argv[0], sys.argv[1]]; exec(open("ana.py").read().split("res = {}")[0])
def pairs(H, kd="wing"):
    x, y = [], []
    for s in ("left", "right"):
        Dm = cvd(H, kd, CH, s)
        for i, j in itertools.combinations(range(7), 2): x.append(Dm[i, j]); y.append(abs(TT[CH[i]] - TT[CH[j]]))
    return np.array(x), np.array(y)
def lexd(kd="wing"):
    X = LEX[0]; x = []
    for s in ("left", "right"):
        C = [cent(X, k, kd, s) for k in CH]
        for i, j in itertools.combinations(range(7), 2): x.append(np.sum((C[i] - C[j]) ** 2))
    return np.array(x)
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz); return np.corrcoef(ex, ey)[0, 1]
lz = lexd(); out = {}
_, y = pairs(LEX[0]); print("lex vs along", spearmanr(lz, y)[0])
rng = np.random.default_rng(0)
for li, L in enumerate(KEEP):
    o = {}
    for cn, H in (("W", HW[li]), ("N", HN[li])):
        x, y = pairs(H); o[f"partial_{cn}"] = partial(x, y, lz); o[f"modelVSlex_{cn}"] = spearmanr(x, lz)[0]
        # permutation null for the partial: shuffle spot labels
        t = np.array([TT[k] for k in CH]); P = list(itertools.combinations(range(7), 2)); nl = []
        for _ in range(1000):
            p = rng.permutation(7); yy = np.array([abs(t[p][i] - t[p][j]) for i, j in P] * 2); nl.append(partial(x, yy, lz))
        o[f"partial_p_{cn}"] = float(np.mean(np.array(nl) >= o[f"partial_{cn}"]))
    x, y = pairs(HW[li], "owing"); o["partial_her"] = partial(x, y, lexd("owing"))
    # within-region nearest: wing spot minus wing mean vs human part minus human mean
    H = HW[li]; wm = H[kind == "wing"].mean(0); hm = H[kind == "hum"].mean(0)
    names = [f"{k}:{s}" for k in HUM for s in sorted(set(side[(kind == "hum") & (key == k)]))]
    M = np.stack([cent(H, *n.split(":")[:1], "hum", n.split(":")[1]) - hm for n in names]); M /= np.linalg.norm(M, axis=1, keepdims=True)
    nw = {}
    for k in CH + ["lead", "under", "trail"]:
        for s in ("left", "right"):
            v = cent(H, k, "wing", s) - wm; c = M @ (v / np.linalg.norm(v)); oo = np.argsort(-c)[:3]
            nw[f"{k}:{s}"] = [(names[i], round(float(c[i]), 2)) for i in oo]
    gm = H[kind == "glorp"].mean(0)
    for k in ("groot", "gtip"):
        v = cent(H, k, "glorp", "left") - gm; c = M @ (v / np.linalg.norm(v)); oo = np.argsort(-c)[:3]; nw[f"{k}:left"] = [(names[i], round(float(c[i]), 2)) for i in oo]
    o["near_within"] = nw; out[L] = o
    print(L, {k: round(v, 2) for k, v in o.items() if not isinstance(v, dict)})
json.dump(out, open(f"{O}/ana2.json", "w"), indent=1)
for L in KEEP[3:7]:
    print(L, {k: v[0] for k, v in out[L]["near_within"].items() if "left" in k})
