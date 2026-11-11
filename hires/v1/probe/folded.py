import numpy as np, sys, json; sys.path.insert(0, 'probe'); sys.argv = ['x']
import probe as pr
P = pr.P; side = np.array([l["side"] for l in pr.L]); Pf = P.copy(); Pf[:, 0] = np.abs(P[:, 0])
H = np.load("H.npy", mmap_mode="r"); out = {}
for name, X in [("E", np.load("E.npy")), ("L22", np.asarray(H[pr.KEEP.index(22)]))]:
    A = pr.spotmean(X); q, _ = pr.cv(A, Pf); q[:, 0] = side * np.abs(q[:, 0])
    out[name] = {k: round(v, 3) for k, v in pr.metrics(q).items()}; np.save(f"probe/folded_{name}.npy", q); print(name, out[name], flush=True)
json.dump(out, open("probe/folded_metrics.json", "w"))
