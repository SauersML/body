"""Q1 numbers from existing runs only. Body control: v2 recipe 8B vs 32B (192 held-out diverse-start scenes, same scenes both sizes).
Text map: hires v2 fig_numbers (8B v2 reliable subspace, 32B). Poke place readout: v1 analyze3d 0.6B vs 8B (named, top layer)."""
import json, numpy as np
from scipy.stats import binomtest
V = "$BODY_WORK/body3d/scale/v2/"
R = {"8B": json.load(open(V + "x8abs/b3s_Qwen3-8B_named_dagger.json")), "32B": json.load(open(V + "x32abs/b3s_Qwen3-32B_named_dagger.json"))}
def wil(x):
    x = np.asarray(x, float); n = len(x); p = x.mean(); z = 1.96; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return [round(p, 3), round(c-h, 3), round(c+h, 3)]
def ev(r, st, t): return [s for s in r["stages"] if s["stage"] == st][0]["eval"][f"new_{t}"]
def mcn(a, b):
    a, b = np.asarray(a), np.asarray(b); k, n = int((a & ~b).sum()), int((a ^ b).sum()); return float(binomtest(k, n).pvalue) if n else 1.0
out = {"control": {}}
for st in ("r0", "r2"):
    for task, key, f in (("reach", "reach", lambda e: e["ok"]), ("touch where touched", "poke_reach", lambda e: e["ok"]),
                         ("exact spot", "poke_reach", lambda e: list(np.array(e["spot_cm"]) < 5)), ("grasp", "grasp", lambda e: e["ok"])):
        x = {m: np.array(f(ev(R[m], st, key)), bool) for m in R}
        out["control"][f"{st}|{task}"] = dict(**{m: wil(x[m]) for m in x}, p=mcn(x["8B"], x["32B"]), n=len(x["8B"]))
H = json.load(open("$BODY_WORK/body/hires2/fig_numbers.json"))
out["text_map_cm"] = {"8B": H[1]["nn5_cm"], "32B": H[2]["nn5_cm"]}
A = "$BODY_RUNS"
a06 = json.load(open(A + "q06c-named/analysis_Qwen3-0.6B_named.json")); a8 = json.load(open(A + "q8b-named/analysis_Qwen3-8B_named.json"))
out["poke_xyz_r2"] = {"0.6B": a06["xyz_r2"]["28"], "8B": a8["xyz_r2"]["36"]}
json.dump(out, open("q1_numbers.json", "w"), indent=1)
for k, v in out["control"].items(): print(k, v)
print(out["text_map_cm"], out["poke_xyz_r2"])
