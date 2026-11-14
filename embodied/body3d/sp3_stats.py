"""Stats for the self-judge follow-up (vcal / probe arms) vs sp2 sim/self/ctrl. python sp3_stats.py -> prints + sp3_stats.json"""
import json, os, numpy as np
from scipy.stats import binomtest
RP = "$BODY_RUNS{m}/selfplay_Qwen3-8B_named_{m}.json"
VA = "$BODY_RUNS{m}/selfplay_Qwen3-8B_named_{m}.json"
J = {m: json.load(open(RP.format(m=m))) for m in ("sim", "self", "ctrl")}
for m in ("vcal", "probe"):
    if os.path.exists(VA.format(m=m)): J[m] = json.load(open(VA.format(m=m)))
TASKS = ["reach", "poke_reach", "grasp"]; rng = np.random.default_rng(0)
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return [round(p, 3), round(c - h, 3), round(c + h, 3)]
def mcn(a, b):
    x = int((a & ~b).sum()); y = int((~a & b).sum()); return dict(plus=x, minus=y, p=float(binomtest(x, x + y, .5).pvalue) if x + y else 1.0)
def ok(m, r, t): return np.array(J[m]["rounds"][r]["eval"][t]["ok"], bool)
def pooled(m, r=-1): return np.concatenate([ok(m, r, t) for t in TASKS])
def bootdiff(a, b, n=10000):  # paired over scenes, stratified by task (192 each)
    d = (a.astype(float) - b.astype(float)).reshape(3, -1); k = d.shape[1]
    s = np.stack([d[i][rng.integers(0, k, (n, k))].mean(1) for i in range(3)]).mean(0)
    return [round(float(d.mean()), 3), round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]
R = dict(primary={}, per_task={}, rounds={}, judge={})
base = pooled("ctrl", 0)
for m in [x for x in ("vcal", "probe") if x in J]:
    A = pooled(m); R["primary"][m] = dict(final=wilson(A.sum(), len(A)), vs_self=mcn(A, pooled("self")), vs_sim=mcn(A, pooled("sim")),
                                          diff_vs_sim=bootdiff(A, pooled("sim")), diff_vs_self=bootdiff(A, pooled("self")), vs_ctrl=mcn(A, pooled("ctrl")), vs_before=mcn(A, base))
for t in TASKS:
    R["per_task"][t] = {"before": wilson(ok("ctrl", 0, t).sum(), 192), "expert": round(float(np.mean(J["ctrl"]["rounds"][0]["eval"][t]["expert"])), 3)}
    for m in J:
        a = ok(m, -1, t); R["per_task"][t][m] = dict(final=wilson(a.sum(), len(a)), vs_before=mcn(a, ok("ctrl", 0, t)),
                                                    **{f"vs_{o}": mcn(a, ok(o, -1, t)) for o in ("sim", "self", "ctrl") if o != m})
for m in J:
    R["rounds"][m] = {t: [round(float(ok(m, r, t).mean()), 3) for r in range(len(J[m]["rounds"]))] for t in TASKS}
    R["rounds"][m]["mean_r1_4_pooled"] = round(float(np.mean([pooled(m, r).mean() for r in range(1, len(J[m]["rounds"]))])), 3)
    R["judge"][m] = [{t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[t].items() if k in ("sim_success", "kept", "keep_precision", "keep_recall", "judge_precision")}
                      for t in TASKS if t in r} for r in J[m]["rounds"][1:]]
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp3_stats.json"), "w"), indent=1)
print(json.dumps(R["primary"], indent=0)); print(json.dumps(R["rounds"]))
