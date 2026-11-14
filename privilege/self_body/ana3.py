"""python ana3.py out/t8_t3.json -> per body P(yes) curves + alpha where P(yes) first reaches .5 (bootstrap over 64 items/body/question)."""
import sys, json, numpy as np
O = json.load(open(sys.argv[1])); T = {k: np.array(v) for k, v in O["T3"].items()}; AL = O["T3_alphas"]; M = O["T3_meta"]
BOD = list(dict.fromkeys(m["b"] for m in M)); rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))
def thr(y):
    if y[0] >= .5: return 0.
    for k in range(1, len(y)):
        if y[k - 1] < .5 <= y[k]: return AL[k - 1] + (.5 - y[k - 1]) / (y[k] - y[k - 1]) * (AL[k] - AL[k - 1])
    return np.inf
R = {"alphas": AL, "bodies": BOD}
for q in ("own", "felt"):
    for dn in ("dhum", "neg", "rand0", "rand1", "rand2"):
        for b in BOD:
            ix = np.array([n for n, m in enumerate(M) if m["b"] == b and m["q"] == q])
            L = np.stack([T[f"{dn}_{a}" if a else "dhum_0"][ix] for a in AL])  # alphas x items
            P = sig(L); y = P.mean(1)
            bs = [thr(P[:, rng.integers(0, len(ix), len(ix))].mean(1)) for _ in range(1000)]
            R[f"{q}|{dn}|{b}"] = dict(curve=y.tolist(), thr=thr(y), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     y_end_ci=[float(np.percentile([P[-1, rng.integers(0, len(ix), len(ix))].mean() for _ in range(1000)], p)) for p in (2.5, 97.5)])
json.dump(R, open(sys.argv[1].replace(".json", "_ana.json"), "w"), indent=1, default=float)
for q in ("own", "felt"):
    for dn in ("dhum", "neg", "rand0", "rand1", "rand2"):
        print(q, dn, " ".join(f"{b[:5]}:{R[f'{q}|{dn}|{b}']['curve'][0]:.2f}>{R[f'{q}|{dn}|{b}']['curve'][-1]:.2f}" for b in BOD))
