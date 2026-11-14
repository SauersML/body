"""ana7.py : object ruler (42 objects, run3 / height6) read on reference entities, unprompted "I", and told heights (run2 / height4: "You are X m" -> I, "a person who is X m" -> The person). -> summary7.json"""
import json, numpy as np
from sklearn.linear_model import RidgeCV, Ridge
D2, D3 = "$BODY_RUNS/out", "$BODY_RUNS/out"
HS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]; out = {}
for s in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]:
    n = f"Qwen3-{s}"
    R3 = json.load(open(f"{D3}/{n}/res.json")); X3 = np.load(f"{D3}/{n}/states.npz")["X"].astype(np.float32); it3 = R3["state_items"]
    R2 = json.load(open(f"{D2}/{n}/res.json")); X2 = np.load(f"{D2}/{n}/states.npz")["X"].astype(np.float32); it2 = R2["state_items"]
    ent = R3["ent"]; nT = len(R3["tpl"]); nE = len(ent); tr = np.arange(2 * nE * nT)
    eid = np.tile(np.repeat(np.arange(nE), nT), 2); y = np.log([ent[i][1] for i in eid])
    o = []
    for li, L in enumerate(R3["layers"]):
        A = X3[tr, li]; mu, sd = A.mean(0), A.std(0) + 1e-3; Z = (A - mu) / sd
        m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Z, y)
        rd = lambda X, idx: float(np.exp(m.predict((X[idx, li] - mu) / sd)).mean())
        sel = lambda it, c, sub: [i for i, (cc, ss) in enumerate(it) if cc == c and ss.startswith(sub + " ")]
        # leave-one-object-out readings of the 42 objects
        loo = []
        for e in range(nE):
            k = eid != e; mm = Ridge(alpha=m.alpha_).fit(Z[k], y[k]); loo.append(float(np.exp(mm.predict(Z[eid == e]).mean())))
        calsub = {"dog": "The dog", "child": "The child", "player": "The basketball player", "giraffe": "The giraffe"}
        cal = {k: rd(X3, sel(it3, k, v)) for k, v in calsub.items()}
        sv = [rd(X2, sel(it2, f"self{int(H*100)}", "I")) for H in HS]; pv = [rd(X2, sel(it2, f"person{int(H*100)}", "The person")) for H in HS]
        o.append(dict(layer=L, alpha=float(m.alpha_), loo=loo, true=[h for _, h in ent], names=[e for e, _ in ent], cal=cal,
                      I=rd(X3, sel(it3, "self", "I")), person=rd(X3, sel(it3, "person", "The person")),
                      told_I=sv, told_person=pv, sl_I=float(np.polyfit(HS, sv, 1)[0]), sl_person=float(np.polyfit(HS, pv, 1)[0]),
                      r_loo=float(np.corrcoef(np.log(loo), np.log([h for _, h in ent]))[0, 1])))
        r = o[-1]; print(s, L, f"rLOO {r['r_loo']:.2f} I {r['I']:.2f} pers {r['person']:.2f} cal", {k: round(v, 2) for k, v in cal.items()}, f"slI {r['sl_I']:.2f} slP {r['sl_person']:.2f}", flush=True)
    out[s] = o
json.dump(out, open("summary7.json", "w"))
