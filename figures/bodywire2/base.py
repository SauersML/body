import os
import sys, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/bodywire"))
from mesh import *   # L, P, N, reg, edges, X, col, cat, LIM
E = edges(k=4)
dm = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
E = E[dm < np.percentile(dm, 90)]
side = np.array([l["side"] for l in L])
C6 = np.array([cat(l["region"]) for l in L])
adj = [set() for _ in L]
for a, b in E: adj[a].add(b); adj[b].add(a)
T = sorted({tuple(sorted((a, b, c))) for a, b in E for c in adj[a] & adj[b]})
T = np.array(T)
