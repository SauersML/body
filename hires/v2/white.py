import numpy as np, sys
from ana2 import *
S = load_stim("stim.py"); b = Body(S); probes = S.probes(); R = np.load(sys.argv[1]); keep = list(R["keep"])
S1 = load_stim("../hires/stim.py"); b1 = Body(S1); p2 = [l["phrase"] for l in S.LOCS]; IDX = [p2.index(l["phrase"]) for l in S1.LOCS]; pr1 = [dict(p, kind="wf") for p in S1.wordfree()]
def cat(rd, ks):
    As, Ws = [], []
    for k in ks:
        A = R[f"{rd}|{k}|all|A"].astype(np.float32); W = R[f"{rd}|{k}|all|W"].astype(np.float32); mu = A.mean(0); sc = np.linalg.norm(A - mu, axis=1).mean(); As.append((A - mu) / sc); Ws.append((W - mu) / sc)
    return np.concatenate(As, 1), np.concatenate(Ws, 1)
C = ["nn5_cm", "chain_rho", "shaft_pair_rho", "partner_nn", "probe_wf_rank"]
for nm, ks in [("L27", [27]), ("L22-30", list(range(22, 31))), ("L26-30", [26, 27, 28, 29, 30]), ("L22-32", [22,23,24,25,26,27,28,29,30,31,32])]:
    A, W = cat("reply", [k for k in ks if k in keep])
    for k in [10, 20, 30, 50]:
        for wh in [False, True]:
            Z1, ZW1 = embed(A[IDX], W[:45], k, wh); m = b1.metrics(Z1, ZW1, pr1)
            print(nm, k, "white" if wh else "     ", " ".join(f"{c[:6]}={m[c]:.3g}" for c in C))
