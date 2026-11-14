"""ana5.py MODEL... : entity size direction (run3 ENT sentences) applied to run2 stated-height states ("You are X m tall" -> "I ...", "a person who is X m" -> "The person ...")."""
import sys, json, numpy as np
from sklearn.linear_model import RidgeCV
D2, D3 = "$BODY_RUNS/out", "$BODY_RUNS/out"
HS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]; out = {}
for n in sys.argv[1:]:
    R3 = json.load(open(f"{D3}/{n}/res.json")); X3 = np.load(f"{D3}/{n}/states.npz")["X"].astype(np.float32)
    R2 = json.load(open(f"{D2}/{n}/res.json")); X2 = np.load(f"{D2}/{n}/states.npz")["X"].astype(np.float32); it2 = R2["state_items"]
    nT = len(R3["tpl"]); ent = R3["ent"]; nE = len(ent); y = np.log(np.tile(np.repeat([h for _, h in ent], nT), 2)); tr = np.arange(2 * nE * nT)
    o = []
    for li, L in enumerate(R3["layers"]):
        A = X3[tr, li]; mu, sd = A.mean(0), A.std(0) + 1e-3; m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit((A - mu) / sd, y)
        rd = lambda c, s: float(np.exp(m.predict((X2[[i for i, (cc, ss) in enumerate(it2) if cc == c and ss.startswith(s + " ")], li] - mu) / sd)).mean())
        sv = [rd(f"self{int(H*100)}", "I") for H in HS]; pv = [rd(f"person{int(H*100)}", "The person") for H in HS]
        o.append(dict(layer=L, self_r=float(np.corrcoef(sv, HS)[0, 1]), self_slope=float(np.polyfit(HS, sv, 1)[0]), self=sv,
                      person_r=float(np.corrcoef(pv, HS)[0, 1]), person_slope=float(np.polyfit(HS, pv, 1)[0]), person=pv, unstated=rd("self", "I")))
    out[n] = o
    for r in o: print(f"{n:15s} L{r['layer']:2d} stated-self r {r['self_r']:.2f} slope {r['self_slope']:.2f} [{r['self'][1]:.2f} @1.5, {r['self'][4]:.2f} @1.8, {r['self'][7]:.2f} @2.1] | person r {r['person_r']:.2f} slope {r['person_slope']:.2f} [{r['person'][2]:.2f} @1.6] | unstated I {r['unstated']:.2f}")
json.dump(out, open("summary5.json", "w"), indent=1)
