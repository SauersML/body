"""Per-trait effect with any global Yes/No bias removed: mean over statements of s*(logit P(yes) steered - unsteered), s=+1 if Yes matches the trait, -1 if No.
Each trait has 25+25 statements, so a uniform shift toward Yes cancels exactly. Reported also as probability points at the trait's base rate."""
import json, numpy as np, sys
from scipy import stats
D = sys.argv[1]; R = json.load(open(f"{D}/persona.json"))
tr = np.array(R["traits"]); s = np.where(R["match_yes"], 1, -1); T = list(dict.fromkeys(tr))
lg = lambda c: (lambda p: np.log(p) - np.log1p(-p))(np.clip(np.array(R[c]["p_yes"]), 1e-6, 1 - 1e-6))
B = lg("base")
def bh(p):
    n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
def pt(c):
    d = s * (lg(c) - B); m = np.array([d[tr == t].mean() for t in T]); se = np.array([d[tr == t].std(ddof=1) / np.sqrt(50) for t in T])
    p = np.array([stats.ttest_1samp(d[tr == t], 0).pvalue for t in T]); return m, se, p
out = {"traits": T, "arms": {}}
for k in [k for k in R if k.startswith(("big", "small"))]:
    n = k.lstrip("bigsmal"); m, se, p = pt(k); q = bh(p)
    rr = [pt(f"rand{j}_{n}") for j in range(3)]; band = np.abs([r[0] for r in rr]).max(0)
    surv = (q < .05) & (np.abs(m) > band)
    out["arms"][k] = dict(d=m.tolist(), se=se.tolist(), q=q.tolist(), band=band.tolist(), surv=surv.tolist(), yes_shift=float((lg(k) - B).mean()),
                          rand_fdr=[int((bh(r[2]) < .05).sum()) for r in rr], rand_absmean=float(np.mean([np.abs(r[0]).mean() for r in rr])))
    print(f"{k:9s} yes-shift {(lg(k)-B).mean():+.2f} | mean|d| {np.abs(m).mean():.2f} rand {out['arms'][k]['rand_absmean']:.2f} | FDR {int((q<.05).sum()):3d} rand {out['arms'][k]['rand_fdr']} | beyond-random {int(surv.sum())}")
A = out["arms"]
for a, b in [("big50", "small50"), ("big85", "small85"), ("big25", "small25")]: print(a, b, "r", round(np.corrcoef(A[a]["d"], A[b]["d"])[0, 1], 2))
for k in ["big85", "small85"]:
    d = np.array(A[k]["d"]); sv = np.array(A[k]["surv"]); o = np.argsort(d)
    print("==", k, "down", [(T[i], round(d[i], 2)) for i in o[:10] if sv[i]]); print("   up", [(T[i], round(d[i], 2)) for i in o[::-1][:10] if sv[i]])
json.dump(out, open("summary_logit.json", "w"))
