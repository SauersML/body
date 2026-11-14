"""ana6.py DIR : summary6.json for run4.py outputs (every Qwen3 chat size). Three methods + combined."""
import sys, json, os, numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
D = sys.argv[1]; SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]; CALH = {"dog": 50, "child": 120, "player": 190, "giraffe": 250}
THR = 0.8
def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(est=float(m), se=float(se), lo=float(m - stats.t.ppf(.975, n - 1) * se), hi=float(m + stats.t.ppf(.975, n - 1) * se), n=n)
def r_(x, y): return float(np.corrcoef(x, y)[0, 1])
def combine(ms):   # DerSimonian-Laird random effects over calibrated methods
    if not ms: return None
    y = np.array([m["est"] for m in ms]); v = np.array([m["se"] ** 2 for m in ms]); w = 1 / v
    if len(ms) == 1: return dict(est=float(y[0]), lo=ms[0]["lo"], hi=ms[0]["hi"], k=1)
    mu = (w * y).sum() / w.sum(); Q = (w * (y - mu) ** 2).sum(); t2 = max(0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + t2); mu = (ws * y).sum() / ws.sum(); se = 1 / np.sqrt(ws.sum())
    return dict(est=float(mu), lo=float(mu - 1.96 * se), hi=float(mu + 1.96 * se), k=len(ms), tau=float(np.sqrt(t2)))
out = {}
for s in SIZES:
    P = f"{D}/Qwen3-{s}"
    if not os.path.exists(f"{P}/res.json"): continue
    R = json.load(open(f"{P}/res.json")); B = R["beh"]; G = np.array(R["grid"]); M = np.array(R["marks"]); HS = R["HS"]; o = {}
    # (a) direct
    def dmean(v): v = np.array(v); p = np.exp(v - v.max()); p /= p.sum(); return float((p * G).sum())
    dv = {c: np.array([[dmean(v) for v in W] for W in B[c]["direct"]]) for c in B}      # warmups x wordings
    st = [dv[f"self{H}"].mean() for H in HS]; en = [dv[k].mean() for k in CALH]
    o["direct"] = dict(**ci(dv["self"].ravel()), r_stated=r_(HS, st), r_ent=r_(list(CALH.values()), en), stated=st,
                       ent=dict(zip(CALH, en)), person=float(dv["person"].mean()), she160=float(dv["she160"].mean()))
    # (b) comparisons: threshold, mapped per (warm-up, wording) on the 4 entities
    T = {c: np.array([[M[0] + np.trapezoid(v, M) for v in W] for W in B[c]["ts"]]) for c in B}   # warmups x wordings
    ys = np.array(list(CALH.values())); selfm, persm, shem = [], [], []
    for i in range(T["self"].shape[0]):
        for k in range(T["self"].shape[1]):
            a, b = np.polyfit([T[e][i, k] for e in CALH], ys, 1)
            selfm.append(a * T["self"][i, k] + b); persm.append(a * T["person"][i, k] + b); shem.append(a * T["she160"][i, k] + b)
    st = [T[f"self{H}"].mean() for H in HS]; en = [T[e].mean() for e in CALH]
    o["comp"] = dict(**ci(selfm), r_stated=r_(HS, st), r_ent=r_(ys, en), stated=st, ent=dict(zip(CALH, en)),
                     raw_self=float(T["self"].mean()), person=float(np.mean(persm)), she160=float(np.mean(shem)))
    # (c) internal: ridge on "You are X cm tall." contexts; pick (position, layer) by held-out wordings
    X = np.load(f"{P}/states.npz")["X"]; meta = R["meta"]; kind = np.array([m[0] for m in meta]); Hh = np.array([m[1] for m in meta], float)
    si = np.array([m[2] for m in meta]); best = None
    for pos in ("s", "q"):
        tr = (kind == "train_" + pos) & (si < 8); va = (kind == "train_" + pos) & (si >= 8)
        for li, L in enumerate(R["layers"]):
            A = X[tr, li].astype(np.float32); mu, sd = A.mean(0), A.std(0) + 1e-3
            m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit((A - mu) / sd, np.log(Hh[tr]))
            rd = lambda msk: np.exp(m.predict((X[msk, li].astype(np.float32) - mu) / sd))
            rv = r_(Hh[va], rd(va))
            if best is None or rv > best["r_heldout"]:
                vals = {c: rd(kind == f"{c}_{pos}") for c in ["self", "person", "she160", *CALH]}
                best = dict(pos=pos, layer=L, r_heldout=rv, vals=vals)
    v = best["vals"]; en = [float(v[e].mean()) for e in CALH]
    o["internal"] = dict(**ci(v["self"]), r_stated=best["r_heldout"], r_ent=r_(ys, en), ent=dict(zip(CALH, en)), pos=best["pos"], layer=best["layer"],
                         person=float(v["person"].mean()), she160=float(v["she160"].mean()))
    # object-fit direction (42 objects, log height), read on the same probe sentences, best layer by entity r
    bo = None
    for li, L in enumerate(R["layers"]):
        tr = np.char.startswith(kind, "ent_"); A = X[tr, li].astype(np.float32); mu, sd = A.mean(0), A.std(0) + 1e-3
        m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit((A - mu) / sd, np.log(Hh[tr]))
        rd = lambda c: float(np.exp(m.predict((X[kind == c, li].astype(np.float32) - mu) / sd)).mean())
        en = [rd(e + "_s") for e in CALH]; rr = r_(ys, en)
        if bo is None or rr > bo["r_ent"]: bo = dict(layer=L, r_ent=rr, ent=dict(zip(CALH, en)), self=rd("self_s"), person=rd("person_s"), she160=rd("she160_s"))
    o["objfit"] = bo
    for k in ("direct", "internal"): o[k]["slope_ent"] = float(np.polyfit(ys, [o[k]["ent"][e] for e in CALH], 1)[0])
    o["comp"]["slope_ent"] = 1.0   # mapped onto the entities by construction
    for k in ("direct", "comp", "internal"): o[k]["ok"] = bool(o[k]["r_stated"] >= THR and o[k]["r_ent"] >= THR and 0.5 <= o[k]["slope_ent"] <= 2)
    o["combined"] = combine([o[k] for k in ("direct", "comp", "internal") if o[k]["ok"]])
    out[s] = o
    f = lambda d: f"{d['est']:6.0f} [{d['lo']:4.0f},{d['hi']:4.0f}] rS {d['r_stated']:.2f} rE {d['r_ent']:.2f} sl {d['slope_ent']:.2f} {'OK' if d['ok'] else '--'} pers {d['person']:4.0f} she {d['she160']:4.0f}"
    print(s); [print(f"  {k:8s}", f(o[k])) for k in ("direct", "comp", "internal")]
    print(f"  internal ent", {k: round(x) for k, x in o['internal']['ent'].items()}, o["internal"]["pos"], o["internal"]["layer"])
    print(f"  objfit   self {bo['self']:.0f} person {bo['person']:.0f} she {bo['she160']:.0f} rE {bo['r_ent']:.2f}")
    print(f"  combined", o["combined"] and {k: round(x) for k, x in o["combined"].items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary6.json"), "w"), indent=1)
