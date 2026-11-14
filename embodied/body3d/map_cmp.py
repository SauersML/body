"""body-map metrics: base (named 8B) vs sp2 final checkpoints. python map_cmp.py"""
import json, os, numpy as np
B = "$BODY_RUNS/analysis_Qwen3-8B_named.json"
A = {"base": B}
for m in ("ctrl", "sim", "self"):
    f = f"$BODY_RUNS{m}/final/analysis_Qwen3-8B_named.json"
    if os.path.exists(f): A[m] = f
J = {k: json.load(open(v)) for k, v in A.items()}
def ft(d):  # final-token RSA by layer
    return {k.split("_L")[-1]: round(v["rho"], 3) for k, v in d["rsa"].items() if k.startswith("final")}
ut = np.triu_indices(len(J["base"]["part_dist_matrix"]), 1)
for k, d in J.items():
    pm = np.array(d["part_dist_matrix"])[ut]; r = np.corrcoef(pm, np.array(J["base"]["part_dist_matrix"])[ut])[0, 1]
    print(f"{k:5s} rsa_final {ft(d)} | lr36 {d['lr_decode']['36']} | prox36 {d['proxdist_r2']['36']} | xyz27 {d['xyz_r2']['27']} xyz36 {d['xyz_r2']['36']} | mirror {d['mirror_closer_than_same_side']:.2f} | partdist r vs base {r:.3f}")
