import numpy as np, sys
import ana3 as a
from ana2 import embed
X, rows = a.load(sys.argv[1], sys.argv[2], int(sys.argv[3])); Y, Q = a.prep(X, rows); del X
T = Y.shape[1]; perm = a.rng.permutation(T); F1, F2 = perm[:T // 2], perm[T // 2:]
# fit reliable directions on framings F1 only, test reliability on F2 (held-out framings)
sp1 = [F1[a.rng.permutation(len(F1))] for _ in range(20)]
C = np.zeros((Y.shape[2],) * 2)
for p in sp1:
    u, v = Y[:, p[:len(p) // 2]].mean(1), Y[:, p[len(p) // 2:]].mean(1); C += (u.T @ v + v.T @ u) / 2
ev, V = np.linalg.eigh(C); V = V[:, ::-1]
r = []
for j in range(150):
    rr = [np.corrcoef(Y[:, F2[q[:len(q) // 2]]].mean(1) @ V[:, j], Y[:, F2[q[len(q) // 2:]]].mean(1) @ V[:, j])[0, 1] for q in [a.rng.permutation(len(F2)) for _ in range(5)]]
    r.append(np.mean(rr))
r = np.array(r); print("held-out split-half r by component:", np.round(r[[0, 5, 10, 20, 40, 60, 80, 100, 149]], 2), "n with r>0.5:", (r > .5).sum(), "first below 0.3:", np.argmax(r < .3))
spa = a.splits(T); _, _, _ = None, None, None
Vall = np.linalg.eigh(sum((Y[:, p[:T // 2]].mean(1).T @ Y[:, p[T // 2:]].mean(1) + Y[:, p[T // 2:]].mean(1).T @ Y[:, p[:T // 2]].mean(1)) / 2 for p in spa))[1][:, ::-1]
A = Y.mean(1); W = Q.mean(1)
for k in [6, 10, 20, 40, 80, 150]:
    m = a.score(A @ Vall[:, :k], W @ Vall[:, :k]); print(k, {c: m[c] for c in ["v1set_nn5_cm", "v1set_chain_rho", "v1set_partner_nn", "v1set_probe_wf_rank", "nn5_cm", "probe_alt_rank"]})
np.save(f"relr_{sys.argv[4]}.npy", r)
