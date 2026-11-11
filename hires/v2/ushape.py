import numpy as np, sys
from ana2 import *
from scipy.stats import spearmanr
S = load_stim("stim.py"); b = Body(S); R = np.load(sys.argv[1])
for key in sys.argv[2:]:
    A = R[key + "|A"].astype(np.float32); W = R[key + "|W"].astype(np.float32); Z, _ = embed(A, W)
    lin, u = [], []
    for nm, ix in b.shafts.items():
        P = b.P[ix]; t = np.linalg.norm(P - P[0], axis=1); t = t / t.max()  # 0 = proximal end
        DZ = squareform(pdist(Z[ix])); DT = np.abs(t[:, None] - t[None]); DJ = np.abs(np.minimum(t, 1 - t)[:, None] - np.minimum(t, 1 - t)[None])  # distance-to-nearest-joint difference
        iu = np.triu_indices(len(ix), 1); lin.append(spearmanr(DZ[iu], DT[iu])[0]); u.append(spearmanr(DZ[iu], DJ[iu])[0])
    print(key, "along-shaft distance rho %.2f   joint-proximity rho %.2f" % (np.mean(lin), np.mean(u)))
