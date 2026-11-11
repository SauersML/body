import json, sys, itertools, numpy as np
from scipy.stats import spearmanr
sys.argv = [sys.argv[0], "out8b"]; exec(open("ana.py").read().split("res = {}")[0])
L = 27; li = KEEP.index(L); out = {}
SP = CH + ["lead", "under", "trail"]
for cn, H in (("W", HW[li]), ("N", HN[li]), ("words", LEX[0])):
    d = {}
    for s in ("left", "right"):
        Dm = cvd(H, "wing", CH, s); d[s] = (Dm[0] / Dm[0].max()).tolist()  # cv sq distance from root, scaled
    out[f"fromroot_{cn}"] = d
# model's own coordinate: PC1 of wing spot centroids (W, both sides, wing-centred)
H = HW[li]; C = np.stack([cent(H, k, "wing", s) for s in ("left", "right") for k in SP]); C -= C.mean(0)
U, S_, Vt = np.linalg.svd(C, full_matrices=False); pc = C @ Vt[:2].T
lab = [f"{k}:{s}" for s in ("left", "right") for k in SP]
t = [TT.get(k, None) for s in ("left", "right") for k in SP]
out["pc"] = dict(zip(lab, pc.tolist())); out["pc_var"] = (S_[:3] ** 2 / np.sum(S_ ** 2)).tolist()
ch = [i for i, k in enumerate(lab) if k.split(":")[0] in CH]
out["pc1_vs_t"] = spearmanr([pc[i, 0] for i in ch], [t[i] for i in ch])[0]; out["pc2_vs_side"] = float(np.corrcoef(pc[:, 1], [1]*10 + [-1]*10)[0, 1])
out["near"] = json.load(open("out8b/ana2.json"))[str(L)]["near_within"]
json.dump(out, open("fig_numbers.json", "w"), indent=1)
print(out["pc_var"], out["pc1_vs_t"], out["pc2_vs_side"]); print({k: [round(x, 2) for x in v] for k, v in out["pc"].items()})
for c in ("W", "N", "words"): print(c, [round(x, 2) for x in out[f"fromroot_{c}"]["left"]])
