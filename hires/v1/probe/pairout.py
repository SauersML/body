# hold out BOTH sides of a body part at once, so the mirror twin is never in training
import numpy as np, sys, json; sys.path.insert(0, 'probe'); sys.argv = ['x']
import probe as pr
P = pr.P; side = np.array([l["side"] for l in pr.L]); Pf = P.copy(); Pf[:, 0] = np.abs(P[:, 0])
pr.F = np.array([f.rstrip("LR") if f[-1] in "LR" else f for f in pr.F]); pr.FOLDS = sorted(set(pr.F)); print(pr.FOLDS)
H = np.load("H.npy", mmap_mode="r"); out = {}; preds = {}
base = np.zeros_like(P)
for f in pr.FOLDS: base[pr.F == f] = P[pr.F != f].mean(0)
out["mean guess"] = pr.metrics(base)
rng = np.random.default_rng(1)
for name, X in [("E", np.load("E.npy"))] + [(f"L{k}", None) for k in pr.KEEP if k >= 10]:
    A = pr.spotmean(X if X is not None else np.asarray(H[pr.KEEP.index(int(name[1:]))]))
    q, al = pr.cv(A, P); out[name] = pr.metrics(q); preds[name] = q
    qf, _ = pr.cv(A, Pf); qf[:, 0] = side * np.abs(qf[:, 0]); out[name + " folded"] = pr.metrics(qf); preds[name + "f"] = qf
    if name in ("L22", "E"):
        sh = [pr.metrics(pr.cv(A, P[rng.permutation(len(P))])[0]) for _ in range(2)]
        out[name + " shuffled"] = {k: float(np.mean([s[k] for s in sh])) for k in sh[0]}
    print(name, {k: round(v, 3) for k, v in out[name].items()}, "folded", {k: round(v, 3) for k, v in out[name + " folded"].items()}, flush=True)
for k, v in out.items(): print(f"{k:16s}", {a: round(b, 3) for a, b in v.items()})
json.dump(out, open("probe/pairout_metrics.json", "w"), indent=1); np.savez("probe/pairout_preds.npz", F=pr.F, **preds)
