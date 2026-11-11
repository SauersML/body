import os
import sys, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/bodypca3d"))
from make import *
P = np.array([l["xyz"] for l in L]); N = np.array([l["n"] for l in L]); N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
reg = np.array([l["region"] if l["region"] != "neck" else "head" for l in L])
def edges(k=5, nd=-0.1, dmax=14.0):
    E = set()
    for i in range(len(L)):
        m = np.where(reg == reg[i])[0]; m = m[m != i]
        d = np.linalg.norm(P[m] - P[i], axis=1)
        ok = (N[m] @ N[i] > nd) & (d < dmax)
        for j in m[ok][np.argsort(d[ok])][:k]: E.add((min(i, j), max(i, j)))
    return np.array(sorted(E))
