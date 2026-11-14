"""python ana2.py DIR TAG -> DIR/sum_TAG.json: T2 ownership effect sizes (my vs its, at L*), paired by item; bootstrap 95% CI."""
import sys, json, numpy as np
from collections import defaultdict
D, TAG = sys.argv[1:3]
O = json.load(open(f"{D}/{TAG}.json")); Z = np.load(f"{D}/{TAG}.npz")
BOD = list(dict.fromkeys(m[0] for m in O["part_meta"])); LA = O["LS_ALL"]; li = LA.index(O["Lstar"])
rng = np.random.default_rng(0); W = np.stack([np.bincount(rng.integers(0, 128, 128), minlength=128) for _ in range(2000)]) / 128
def ci(x): return [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
R = {"bodies": BOD, "Lstar": O["Lstar"]}
for pos in ("sent", "end"):
    Y = Z[f"own_{pos}_pron"][:, li]; idx = defaultdict(list)
    for n, m in enumerate(O["own_meta"]): idx[(m["sb"], m["pb"], m["w"])].append((m["i"] * 8 + m["j"], n))
    get = lambda k: Y[[n for _, n in sorted(idx[k])]]
    sd = np.std(np.concatenate([get(("default", "human", w)) for w in ("my", "its")]))  # one scale per model: default-condition spread
    out = {}
    for (sb, pb, w) in list(idx):
        if w != "my" or (sb, pb, "its") not in idx: continue
        d = (get((sb, pb, "my")) - get((sb, pb, "its"))) / sd
        out[f"{sb}|{pb}"] = ci(W @ d)
    R[f"T2_{pos}"] = out
json.dump(R, open(f"{D}/sum_{TAG}.json", "w"), indent=1)
f = lambda x: "%.2f[%.2f,%.2f]" % tuple(x)
for pos in ("sent",):
    o = R[f"T2_{pos}"]
    print(TAG, pos, "own:", " ".join(f"{b[:5]}={f(o[f'{b}|{b}'])}" for b in BOD))
    print(TAG, pos, "default:", " ".join(f"{b[:5]}={f(o[f'default|{b}'])}" for b in BOD))
    print(TAG, pos, "foreign (sb|human):", " ".join(f"{b[:5]}={f(o[f'{b}|human'])}" for b in BOD if f"{b}|human" in o))
