"""Stats for b3s runs: success + named-spot (<5 cm) with Wilson CIs per stage; exact McNemar vs expert and between runs (paired scenes).
python3 b3s_stats.py JSON [JSON...]  (prints a table; writes scale/stats.json)"""
import sys, json, numpy as np
from scipy.stats import binomtest


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def mcnemar(a, b):
    a, b = np.array(a, bool), np.array(b, bool); x, y = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a_only=x, b_only=y, p=float(binomtest(x, x + y).pvalue) if x + y else 1.0)


OUT = {}
for fn in sys.argv[1:]:
    R = json.load(open(fn)); name = fn.split("/")[-2].replace("hypha-prod-b3s-", ""); OUT[name] = {}
    for st in R["stages"]:
        if "eval" not in st: continue
        row = {}
        for k, v in st["eval"].items():
            ok = v["ok"]; sp = np.array(v["spot_cm"]) < 5; ex = R["expert"][k]["new_expert"]
            row[k] = dict(success=wilson(sum(ok), len(ok)), spot5=wilson(int(sp.sum()), len(sp)), expert=wilson(sum(ex), len(ex)), vs_expert=mcnemar(ok, ex), ok=ok)
            if R["expert"][k].get("old_expert"): row[k]["old_expert"] = wilson(sum(R["expert"][k]["old_expert"]), len(ok))
        OUT[name][st["stage"]] = row
        print(name, st["stage"], "  ".join(f"{k} {row[k]['success'][0]:.2f}/{row[k]['spot5'][0]:.2f}" for k in row))
json.dump(OUT, open("stats.json", "w"))
