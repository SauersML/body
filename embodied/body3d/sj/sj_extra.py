import numpy as np, json
from sklearn.metrics import roc_auc_score
D = np.load("sj_scores.npz"); Z = np.load("selfjudge.npz"); v, p, y, task, says = D["verbal"], D["probe"], D["y"], D["task"], D["says"]; scene = Z["scene"]
rng = np.random.default_rng(1); out = {}
def bacc(s, th, i): return float(0.5 * ((s[i & (y == 1)] > th).mean() + (s[i & (y == 0)] <= th).mean()))
for t in ["reach", "poke_reach", "grasp"]:
    i = task == t; sc = np.unique(scene[i]); folds = np.array_split(rng.permutation(sc), 8); pv = np.zeros(len(y), bool)
    for f in folds:  # threshold for verbal fit on other scenes (max balanced acc), applied to held-out scenes
        te = i & np.isin(scene, f); tr = i & ~np.isin(scene, f); ths = np.quantile(v[tr], np.linspace(.02, .98, 97))
        th = ths[np.argmax([bacc(v, h, tr) for h in ths])]; pv[te] = v[te] > th
    d = []
    for _ in range(2000):
        bs = rng.choice(sc, len(sc)); idx = np.concatenate([np.where(i & (scene == s))[0] for s in bs])
        d.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], v[idx]))
    out[t] = dict(verbal_acc_at_yes=round(float((says == y)[i].mean()), 3), verbal_bacc_at_yes=round(bacc(v, 0, i), 3),
                  verbal_acc_recal=round(float((pv == y)[i].mean()), 3), verbal_bacc_recal=round(float(0.5 * (pv[i & (y == 1)].mean() + (~pv[i & (y == 0)]).mean())), 3),
                  probe_acc=round(float(((p > 0) == y)[i].mean()), 3), probe_bacc=round(bacc(p, 0, i), 3), always_no_acc=round(float(1 - y[i].mean()), 3),
                  auc_diff=round(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], v[i]), 3), auc_diff_ci=[round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)])
json.dump(out, open("sj_extra.json", "w"), indent=1); print(json.dumps(out))
