import numpy as np, json; from ana2 import *
S = load_stim("../hires/stim.py"); b = Body(S); D = json.load(open("../hires/rows.json")); rows = D["rows"]; H = np.load("../hires/H.npy", mmap_mode="r")
probes = [dict(p, kind="wf") for p in S.wordfree()]
print(len(b.chains[0]), [len(g) for g in b.chains[0]], len(b.shafts))
res = {}
for L in [22, 24, 26, 27, 28]:
    A, W = spot_means(H[D["keep"].index(L)].astype(np.float32), rows, len(probes)); Z, ZW = embed(A, W); res[L] = b.metrics(Z, ZW, probes); print(L, res[L])
json.dump(res, open("v1_metrics.json", "w"), indent=1)
