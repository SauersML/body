"""Unsupervised framing selection: keep framings whose spot geometry agrees with the other framings (no body labels)."""
import numpy as np, json, sys
from ana2 import *
S = load_stim("stim.py"); b = Body(S); probes = S.probes()
pref, rd, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
D = json.load(open(f"out8b_{pref}/rows.json")); rows = D["rows"]; X = np.asarray(np.load(f"out8b_{pref}/{rd}.npy", mmap_mode="r")[D["keep"].index(L)], dtype=np.float32)
nT = len(S.FRAMINGS); Ds = []
for t in range(nT):
    A, _ = spot_means(X, rows, len(probes), [t]); An = A - A.mean(0); An /= np.linalg.norm(An, axis=1, keepdims=True); Ds.append(pdist(An, "cosine"))
Ds = np.array(Ds); M = Ds.mean(0)
ag = np.array([spearmanr(Ds[t], (M * nT - Ds[t]) / (nT - 1))[0] for t in range(nT)])
o = np.argsort(-ag)
for t in o: print(f"{ag[t]:.3f} {S.FRAMINGS[t][1]}")
res = {}
for k in [3, 5, 8, 13, 26]:
    A, W = spot_means(X, rows, len(probes), sorted(o[:k].tolist())); Z, ZW = embed(A, W); m = b.metrics(Z, ZW, probes); res[k] = m
    print(k, {c: m[c] for c in ["nn5_cm", "chain_rho", "shaft_pair_rho", "partner_nn", "probe_wf_rank", "probe_alt_rank"]})
json.dump(dict(agree=ag.tolist(), order=o.tolist(), res=res), open(f"agree_{pref}_{rd}_{L}.json", "w"))
