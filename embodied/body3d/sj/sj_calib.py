"""Frozen judges for practice: verbal threshold + L27 "A:" probe, per-task keep-rate-matched thresholds.
python sj_calib.py selfjudge.npz -> sj_judges.npz, sj_calib.json  (fit split scenes 0-95, check split 96-127)"""
import sys, json, numpy as np, warnings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = np.load(sys.argv[1]); task = D["task"]; scene = D["scene"]; y = D["ok"].astype(int); TASKS = ["reach", "poke_reach", "grasp"]
X = D["ha"][:, list(D["layers"]).index(27)].astype(np.float32); verbal = D["lp"][:, 0] - D["lp"][:, 1]
fit, chk = scene < 96, scene >= 96; grp = np.array([f"{t}_{s}" for t, s in zip(task, scene)])
mu = X[fit].mean(0); sd = np.maximum(X[fit].std(0), 1e-2); Z = (X - mu) / sd
Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]; fi = np.where(fit)[0]
cv = LogisticRegressionCV(Cs=Cs, cv=list(GroupKFold(4).split(Z[fi], y[fi], grp[fi])), scoring="roc_auc", max_iter=3000).fit(Z[fi], y[fi])
C = float(cv.C_[0]); oof = np.zeros(len(fi))
for tr, te in GroupKFold(4).split(Z[fi], y[fi], grp[fi]):
    oof[te] = LogisticRegression(C=C, max_iter=3000).fit(Z[fi][tr], y[fi][tr]).decision_function(Z[fi][te])
m = LogisticRegression(C=C, max_iter=3000).fit(Z[fi], y[fi]); probe = Z @ m.coef_[0] + m.intercept_[0]
def thr(s, t, idx):  # threshold so kept fraction == true success rate
    i = idx[task[idx] == t]; return float(np.quantile(s[i], 1 - y[i].mean()))
thv = {t: thr(verbal, t, fi) for t in TASKS}; pf = np.zeros(len(y)); pf[fi] = oof; thp = {t: thr(pf, t, fi) for t in TASKS}
R = dict(C=C, thr_verbal=thv, thr_probe=thp, check={})
ci = np.where(chk)[0]
for t in TASKS:
    i = ci[task[ci] == t]; r = dict(success=float(y[i].mean()))
    for nm, s, th in (("raw_verbal", verbal, {t: 0.0}), ("verbal_cal", verbal, thv), ("probe", probe, thp)):
        k = s[i] > th[t]; r[nm] = dict(keep=round(float(k.mean()), 3), precision=round(float(y[i][k].mean()), 3) if k.any() else None,
                                       acc=round(float((k == y[i]).mean()), 3), auc=round(roc_auc_score(y[i], s[i]), 3))
    R["check"][t] = r
np.savez("sj_judges.npz", mu=mu, sd=sd, w=m.coef_[0].astype(np.float32), b=np.float32(m.intercept_[0]),
         thr_verbal=np.array([thv[t] for t in TASKS]), thr_probe=np.array([thp[t] for t in TASKS]), tasks=np.array(TASKS))
json.dump(R, open("sj_calib.json", "w"), indent=1); print(json.dumps(R, indent=1))
