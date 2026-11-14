"""ana4.py DIR... : known-height entities (child 1.2, player 1.9, dog 0.5, giraffe 2.5) as calibration; self read on the same scale."""
import sys, json, numpy as np
from sklearn.linear_model import RidgeCV
CALH = {"dog": 0.5, "child": 1.2, "player": 1.9, "giraffe": 2.5}; G = np.arange(30, 301, 2)
def E(rows):
    a = np.array(rows); hs = np.unique(a[:, 0]); pm = np.array([a[a[:, 0] == h, 1].mean() for h in hs]); return float(hs[0] + np.trapezoid(pm, hs))
def dm(d): d = np.array(d); p = np.exp(d - d.max()); p /= p.sum(); return float((p * G).sum()) / 100
def cal(vals):   # fit known = a*readout + b on the 4 entities; return r, mapped self/person
    x = np.array([vals[k] for k in CALH]); y = np.array(list(CALH.values())); r = np.corrcoef(x, y)[0, 1]; a, b = np.polyfit(x, y, 1)
    return dict(r=float(r), raw={k: float(v) for k, v in vals.items()}, self=float(a * vals["self"] + b), person=float(a * vals["person"] + b))
out = {}
for D in sys.argv[1:]:
    R = json.load(open(f"{D}/res.json")); n = R["model"].split("/")[-1]; B = R["beh"]; o = {}
    for k in ("ts", "wall"): o[k] = cal({c: E(B[c][k]) for c in B})
    o["direct"] = cal({c: dm(B[c]["direct"]) for c in B})
    X = np.load(f"{D}/states.npz")["X"].astype(np.float32); items = R["state_items"]; nT = len(R["tpl"]); ent = R["ent"]; nE = len(ent)
    y = np.log(np.tile(np.repeat([h for _, h in ent], nT), 2)); tr = np.arange(2 * nE * nT)
    subj = {"self": "I", "person": "The person", "child": "The child", "player": "The basketball player", "dog": "The dog", "giraffe": "The giraffe"}
    o["probe"] = []
    for li, L in enumerate(R["layers"]):
        A = X[tr, li]; mu, sd = A.mean(0), A.std(0) + 1e-3; m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit((A - mu) / sd, y)
        vals = {}
        for c, s in subj.items():
            ii = [i for i, (cc, ss) in enumerate(items) if i >= 2 * nE * nT and cc == c and ss.startswith(s + " ")]
            vals[c] = float(np.exp(m.predict((X[ii, li] - mu) / sd)).mean())
        rec = cal(vals); rec["layer"] = L; o["probe"].append(rec)
    o["probe_best"] = max(o["probe"], key=lambda r: r["r"]); out[n] = o
    f = lambda c: f"r {c['r']:.2f} raw " + " ".join(f"{k} {v:.2f}" for k, v in c["raw"].items()) + f" -> self {c['self']:.2f} person {c['person']:.2f}"
    print(n); [print(f"   {k:6s}", f(o[k])) for k in ("direct", "ts", "wall")]; pb = o["probe_best"]; print(f"   probe L{pb['layer']}", f(pb))
json.dump(out, open("summary4.json", "w"), indent=1)
