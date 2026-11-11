import json, sys, numpy as np
from scipy.stats import spearmanr, pearsonr
O = sys.argv[1]; P = json.load(open(f"{O}/probes.json")); RO = json.load(open(f"{O}/readout.json"))
out = {}
for k, d in RO.items():
    cn, nm, q = k.split("_"); R = [P[nm][i] for i in d["idx"]]; e = np.array(d["ev"])
    t = np.array([r["t"] for r in R]); kind = np.array([r["kind"] for r in R]); reg = np.array([r["reg"] for r in R]); pos = np.array([r["pos"] for r in R])
    if q == "pos":
        m = [e[pos == p].mean() for p in range(5)]
        out[k] = dict(by_pos=np.round(m, 2).tolist(), r=round(pearsonr(t, e)[0], 3), by_kind={kk: round(e[kind == kk].mean(), 2) for kk in np.unique(kind)},
                      by_reg=[round(e[reg == g].mean(), 2) for g in range(3)], mass=round(np.mean(d["mass"]), 3))
    else:
        out[k] = dict(by_kind={kk: round(e[kind == kk].mean(), 2) for kk in ["finger", "drag", "press", "warm", "palm"]},
                      by_pos=np.round([e[pos == p].mean() for p in range(5)], 2).tolist(), by_reg=[round(e[reg == g].mean(), 2) for g in range(3)],
                      mass=round(np.mean(d["mass"]), 3))
for k, v in out.items(): print(k, v)
TR = json.load(open(f"{O}/traces.json")); TRO = json.load(open(f"{O}/trace_readout.json"))
for cn in ("W", "N"):
    for nm in ("wing", "desc"):
        for d in (0, 1):
            sel = [i for i, t in enumerate(TR) if t["nm"] == nm and t["dir"] == d]
            print("trace", cn, nm, "out" if d == 0 else "in", [round(float(np.mean(np.array(TRO[f"{cn}_{k}"]["ev"])[sel])), 2) for k in range(6)])
json.dump(out, open(f"{O}/readout_summary.json", "w"), indent=1)
