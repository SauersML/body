"""Stats for selfplay runs. python sp_stats.py PREFIX (e.g. hypha-prod-b3d-sp2)"""
import sys, json, os, numpy as np
from scipy.stats import binomtest
pre = sys.argv[1]; D = "$BODY_WORK/runpod"; tag = "Qwen3-8B_named"
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return p, c - h, c + h
def mcnemar(a, b):  # paired: a better than b?
    a, b = np.array(a), np.array(b); x = int((a & ~b).sum()); y = int((~a & b).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)
R = {}
for m in ("sim", "self", "ctrl"):
    f = f"{D}/{pre}-{m}/selfplay_{tag}_{m}.json"
    if os.path.exists(f): R[m] = json.load(open(f))
for task in ("reach", "poke_reach", "grasp"):
    print(f"## {task}")
    for m, J in R.items():
        rs = J["rounds"]; line = []
        for r in rs:
            ev = r["eval"][task]; ok = np.array(ev["ok"]); p, lo, hi = wilson(ok.sum(), len(ok))
            line.append(f"r{r['round']} {p:.2f} [{lo:.2f},{hi:.2f}] j{np.mean(np.array(ev['judge']) == ok):.2f}")
        ev0, evf = rs[0]["eval"][task], rs[-1]["eval"][task]; ex = np.array(evf["expert"], bool)
        a, b, pv = mcnemar(np.array(evf["ok"], bool), np.array(ev0["ok"], bool)); a2, b2, pv2 = mcnemar(np.array(evf["ok"], bool), ex)
        print(f"{m:5s} | " + " | ".join(line) + f" || expert {ex.mean():.2f} | final vs r0 +{a}/-{b} p={pv:.3g} | final vs expert +{a2}/-{b2} p={pv2:.3g} | tip {np.mean(ev0['tip_cm']):.1f}->{np.mean(evf['tip_cm']):.1f}cm")
    ms = list(R)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            A = np.array(R[ms[i]]["rounds"][-1]["eval"][task]["ok"], bool); Bb = np.array(R[ms[j]]["rounds"][-1]["eval"][task]["ok"], bool)
            if len(A) == len(Bb): a, b, pv = mcnemar(A, Bb); print(f"  final {ms[i]} vs {ms[j]}: +{a}/-{b} p={pv:.3g}")
for m, J in R.items():
    for r in J["rounds"][1:]:
        if "reach" in r and isinstance(r["reach"], dict):
            print(m, "r", r["round"], {k: (round(r[k]["sim_success"], 2), round(r[k]["judge_acc"], 2), r[k]["judge_precision"] and round(r[k]["judge_precision"], 2), r[k]["kept"]) for k in ("reach", "poke_reach", "grasp")})
