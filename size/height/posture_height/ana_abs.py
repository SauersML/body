"""ana_abs.py OUTDIR: ridge size ruler on FIT items (log10 m), CV r leave-items-out; read held-out labels.
CI = t over paraphrases (each paraphrase averaged over 4 templates)."""
import sys, json, numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
O = sys.argv[1]; X = np.load(f"{O}/abs_states.npy").astype(np.float32); J = json.load(open(f"{O}/abs_items.json"))
its = J["items"]; FH = {x: h for x, h in J["fit"]}; out = {}
fi = [i for i, it in enumerate(its) if it["lab"] in FH]; y = np.log10([FH[its[i]["lab"]] for i in fi])
g = np.unique([its[i]["lab"] for i in fi], return_inverse=True)[1]
def ci(v):
    v = np.asarray(v); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), v.mean() - h, v.mean() + h
def fmt(m):  # log10 m -> readable
    v = 10 ** m; return f"{v*100:.0f}cm" if v < 1 else (f"{v:.1f}m" if v < 1000 else f"{v/1000:.1f}km")
for li, L in enumerate(J["layers"]):
    Z0 = X[:, li]; mu, sd = Z0[fi].mean(0), Z0[fi].std(0) + 1e-3; Z = (Z0 - mu) / sd
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(Z[fi], y, g):
        pred[te] = RidgeCV(alphas=np.logspace(1, 6, 11)).fit(Z[fi][tr], y[tr]).predict(Z[fi][te])
    r = float(np.corrcoef(pred, y)[0, 1]); m = RidgeCV(alphas=np.logspace(1, 6, 11)).fit(Z[fi], y)
    P = m.predict(Z); res = {}
    for lab, phs in J["read"].items():
        v = [np.mean([P[i] for i, it in enumerate(its) if it["lab"] == lab and it["ph"] == p]) for p in phs]
        res[lab] = ci(v)
    out[L] = dict(cv_r=r, read={k: list(map(float, v)) for k, v in res.items()})
    print(f"L{L} cv_r {r:.2f}  " + " | ".join(f"{k}: {fmt(v[0])} [{fmt(v[1])},{fmt(v[2])}]" for k, v in res.items()))
json.dump(out, open(f"{O}/abs_summary.json", "w"), indent=1)
