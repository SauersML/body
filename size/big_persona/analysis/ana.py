import json, numpy as np, sys
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else "$BODY_RUNS"
R = json.load(open(f"{D}/persona.json")); C = json.load(open(f"{D}/conds.json")); ARMS = C["arms"]
tr = np.array(R["traits"]); my = np.array(R["match_yes"]); T = list(dict.fromkeys(tr))
def pm(c): p = np.array(R[c]["p_yes"]); return np.where(my, p, 1 - p)
B = pm("base"); RN = [0.2, 0.5, 0.9, 1.4, 1.9]
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
def per_trait(c):
    d = pm(c) - B; m = np.array([d[tr == t].mean() for t in T]); se = np.array([d[tr == t].std(ddof=1) / np.sqrt((tr == t).sum()) for t in T])
    p = np.array([stats.ttest_1samp(d[tr == t], 0).pvalue for t in T]); return m, se, p
out = {"traits": T, "base": [float(B[tr == t].mean()) for t in T], "arms": {}}
for k, v in ARMS.items():
    n = (k.lstrip("bigsmal") if k[:3] in ("big","sma") else min(RN, key=lambda x: abs(x - abs(v)))); m, se, p = per_trait(k); q = bh(p)
    rm = np.array([per_trait(f"rand{j}_{n}")[0] for j in range(3)]); band = np.abs(rm).max(0)
    rq = [bh(per_trait(f"rand{j}_{n}")[2]) for j in range(3)]
    surv = (q < .05) & (np.abs(m) > band)
    mass = np.mean(R[k]["mass"])
    out["arms"][k] = dict(nats=v, rand_nats=n, d=m.tolist(), se=se.tolist(), q=q.tolist(), band=band.tolist(), surv=surv.tolist(), mass=float(mass),
                          n_fdr=int((q < .05).sum()), n_surv=int(surv.sum()), rand_fdr=[int((x < .05).sum()) for x in rq], mean_abs=float(np.abs(m).mean()), rand_mean_abs=float(np.abs(rm).mean()))
    print(f"{k:8s} {v:+.2f}nat mass {mass:.3f} meanabs {np.abs(m).mean():.3f} rand {np.abs(rm).mean():.3f} | FDR {int((q<.05).sum()):3d} rand FDR {[int((x<.05).sum()) for x in rq]} | surv {int(surv.sum())}")
print("base mass", np.mean(R["base"]["mass"]))
json.dump(out, open("$BODY_WORK/body/bigp/summary_" + D.split("-")[-1] + ".json", "w"))
