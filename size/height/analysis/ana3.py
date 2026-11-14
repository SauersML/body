"""ana3.py DIR... : implied height = 1.0 m + integral of P(taller/above | mark) over marks 1.0-2.5 m (survival-function mean). Stated heights as a check."""
import sys, json, numpy as np
HS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]; G = np.arange(100, 231)
def E(rows, nb=300, rng=np.random.default_rng(0)):
    a = np.array(rows); h, p = a[:, 0], a[:, 1]; hs = np.unique(h); ix = [np.where(h == x)[0] for x in hs]
    est = lambda pm: 1.0 + np.trapezoid(pm, hs)
    pm = np.array([p[i].mean() for i in ix]); bs = []
    for _ in range(nb):   # resample wordings (items) within each mark
        bs.append(est(np.array([p[rng.choice(i, len(i))].mean() for i in ix])))
    return float(est(pm)), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
def dm(d): d = np.array(d); p = np.exp(d - d.max()); p /= p.sum(); return float((p * G).sum()) / 100
out = {}
for D in sys.argv[1:]:
    R = json.load(open(f"{D}/res.json")); n = R["model"].split("/")[-1]; B = R["beh"]; o = {}
    for k in ("ts", "wall"):
        o[k] = {c: E(B[c][k]) for c in B}
        st = [o[k][f"self{int(H*100)}"][0] for H in HS]; o[k + "_track"] = dict(r=float(np.corrcoef(st, HS)[0, 1]), slope=float(np.polyfit(HS, st, 1)[0]), vals=st)
        sp = [o[k][f"person{int(H*100)}"][0] for H in HS]; o[k + "_track_person"] = dict(r=float(np.corrcoef(sp, HS)[0, 1]), slope=float(np.polyfit(HS, sp, 1)[0]), vals=sp)
    o["direct"] = {c: dm(B[c]["direct"]) for c in B}
    o["direct_track"] = float(np.corrcoef([o["direct"][f"self{int(H*100)}"] for H in HS], HS)[0, 1])
    out[n] = o; t = o["ts"]
    print(f"{n:15s} TS self {t['self'][0]:.2f}[{t['self'][1]:.2f},{t['self'][2]:.2f}] person {t['person'][0]:.2f} woman {t['woman'][0]:.2f} man {t['man'][0]:.2f} | track r {o['ts_track']['r']:.2f} slope {o['ts_track']['slope']:.2f} | WALL self {o['wall']['self'][0]:.2f} track {o['wall_track']['r']:.2f}/{o['wall_track']['slope']:.2f} | DIRECT self {o['direct']['self']:.2f} person {o['direct']['person']:.2f} w {o['direct']['woman']:.2f} m {o['direct']['man']:.2f} track {o['direct_track']:.2f}")
json.dump(out, open("summary3.json", "w"), indent=1)
