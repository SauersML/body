"""Per-spot means (framing means removed) for every readout x layer; also a 4-framing (v1-like touch) subset and per-family subsets."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stim as S
from sm import spot_means
raw, out = sys.argv[1], sys.argv[2]
D = json.load(open(f"{raw}/rows.json")); rows = D["rows"]; npb = len(S.probes())
fam = [f for f, _ in S.FRAMINGS]
subsets = {"all": None, "v1like": [0, 2, 3, 11]}
for f in sorted(set(fam)): subsets["fam_" + f] = [t for t, g in enumerate(fam) if g == f]
res = {}
for rd in ["reply", "final", "body"]:
    X = np.load(f"{raw}/{rd}.npy", mmap_mode="r")
    for li, k in enumerate(D["keep"]):
        x = np.asarray(X[li], dtype=np.float32)
        for sn, sub in subsets.items():
            if sn != "all" and li not in (5, 8, 11): continue
            A, W = spot_means(x, rows, npb, sub); res[f"{rd}|{k}|{sn}|A"] = A.astype(np.float16); res[f"{rd}|{k}|{sn}|W"] = W.astype(np.float16)
        print(rd, k, flush=True)
np.savez(out, keep=np.array(D["keep"]), **res)
