"""ana2.py DIR... -> summary2.json: stated-height calibration of thresholds and of states."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana import thresh
import sys, json, numpy as np
from sklearn.linear_model import RidgeCV
HS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]; G = np.arange(100, 231)
def dmean(d): d = np.array(d); p = np.exp(d - d.max()); p /= p.sum(); return float((p * G).sum())
out = {}
for D in sys.argv[1:]:
    R = json.load(open(f"{D}/res.json")); name = R["model"].split("/")[-1]; B = R["beh"]; o = {}
    for kind in ("wall", "ts"):
        T = {c: thresh(B[c][kind]) for c in B}
        e = {"raw": {c: T[c]["H"] for c in T}, "ok": {c: T[c]["ok"] for c in T}, "curve": {c: T[c]["curve"] for c in T},
             "lohi": {c: (T[c]["lo"], T[c]["hi"]) for c in T}}
        for fam, tgt in (("self", ["self"]), ("person", ["person", "woman", "man"])):
            x = np.array([T[f"{fam}{int(H*100)}"]["H"] for H in HS]); r = np.corrcoef(x, HS)[0, 1]
            a, b = np.polyfit(x, HS, 1)   # raw threshold -> stated height
            e[f"cal_{fam}"] = dict(r=float(r), slope=float(a), icpt=float(b),
                                   est={t: float(a * T[t]["H"] + b) for t in tgt}, est_lohi={t: (float(a*T[t]["lo"]+b), float(a*T[t]["hi"]+b)) for t in tgt})
        o[kind] = e
    o["direct"] = {c: dmean(B[c]["direct"]) for c in B}
    X = np.load(f"{D}/states.npz")["X"].astype(np.float32); items = R["state_items"]; nT = len(R["tpl"])
    SX = [("I", "self"), ("The person", "person"), ("She", "woman"), ("He", "man")] + [("I", f"self{int(H*100)}") for H in HS] + [("The person", f"person{int(H*100)}") for H in HS]
    idx = lambda key: [i for i, (c, s) in enumerate(items) if c == key[1] and s.startswith(key[0] + " ")]
    o["probe"] = []
    for k, L in enumerate(R["layers"]):
        rec = {"layer": L}
        for fam, subj in (("self", "I"), ("person", "The person")):
            ii = sum([idx((subj, f"{fam}{int(H*100)}")) for H in HS], []); A = X[ii, k]; y = np.repeat(HS, nT); g = np.repeat(np.arange(len(HS)), nT)
            mu, sd = A.mean(0), A.std(0) + 1e-3; Z = (A - mu) / sd; pred = np.zeros(len(y))
            for h in range(len(HS)):
                tr = g != h; pred[~tr] = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Z[tr], y[tr]).predict(Z[~tr])
            m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Z, y)
            rec[fam] = dict(loo_r=float(np.corrcoef(pred, y)[0, 1]), loo_mae=float(np.abs(pred - y).mean()),
                            est={f"{s}|{c}": float(m.predict((X[idx((s, c)), k] - mu) / sd).mean()) for s, c in SX[:4]})
        o["probe"].append(rec)
    out[name] = o
    w, t = o["wall"], o["ts"]
    print(f"{name:16s} WALL cal_r {w['cal_self']['r']:.2f} self {w['cal_self']['est']['self']:.2f} | TS cal_r {t['cal_self']['r']:.2f} raw self {t['raw']['self']:.2f} ok {t['ok']['self']} cal self {t['cal_self']['est']['self']:.2f} person {t['cal_person']['est']['person']:.2f} w {t['cal_person']['est']['woman']:.2f} m {t['cal_person']['est']['man']:.2f} | direct self {o['direct']['self']:.0f} person {o['direct']['person']:.0f} w {o['direct']['woman']:.0f} m {o['direct']['man']:.0f} s150 {o['direct']['self150']:.0f} s200 {o['direct']['self200']:.0f}")
    for r in o["probe"]: print(f"   L{r['layer']} self loo_r {r['self']['loo_r']:.2f} mae {r['self']['loo_mae']:.2f} I {r['self']['est']['I|self']:.2f} person->{r['self']['est']['The person|person']:.2f} | personprobe r {r['person']['loo_r']:.2f} person {r['person']['est']['The person|person']:.2f} she {r['person']['est']['She|woman']:.2f} he {r['person']['est']['He|man']:.2f} I {r['person']['est']['I|self']:.2f}")
json.dump(out, open("summary2.json", "w"), indent=1)
