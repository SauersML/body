"""named <-> nameless map comparison from analysis outputs. python transfer.py DIR_NAMED DIR_NAMELESS TAG_MODEL"""
import sys, json, numpy as np
from scipy.stats import spearmanr
dn, dl, m = sys.argv[1:4]
A = json.load(open(f"{dn}/analysis_{m}_named.json")); Bj = json.load(open(f"{dl}/analysis_{m}_nameless.json"))
iu = np.triu_indices(17, 1); Pa, Pb = np.array(A["part_dist_matrix"]), np.array(Bj["part_dist_matrix"])
ra = np.load(f"{dn}/reps_{m}_named.npz"); rb = np.load(f"{dl}/reps_{m}_nameless.npz"); assert (ra["ti"] == rb["ti"]).all()
def cosd(Z): Z = Z - Z.mean(0); Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); return 1 - Z @ Z.T
ju = np.triu_indices(len(ra["ti"]), 1)
out = dict(part_map_r=round(float(np.corrcoef(Pa[iu], Pb[iu])[0, 1]), 3), part_map_spearman=round(float(spearmanr(Pa[iu], Pb[iu])[0]), 3),
           poke_rsm_spearman=round(float(spearmanr(cosd(ra["HF"])[ju], cosd(rb["HF"])[ju])[0]), 3))
print(json.dumps(out)); json.dump(out, open(f"transfer_{m}.json", "w"))
