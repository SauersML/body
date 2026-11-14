"""Analysis for self-judge. python sj_analyze.py selfjudge.npz -> sj_results.json"""
import sys, json, numpy as np, warnings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = np.load(sys.argv[1]); task = D["task"]; scene = D["scene"]; y = D["ok"].astype(int); lp = D["lp"]; LAY = list(D["layers"])
verbal = lp[:, 0] - lp[:, 1]; says = (verbal > 0).astype(int); TASKS = ["reach", "poke_reach", "grasp"]
grp = np.array([f"{t}_{s}" for t, s in zip(task, scene)]); rng = np.random.default_rng(0)
import os, pickle
CACHE = pickle.load(open("oof_cache.pkl", "rb")) if os.path.exists("oof_cache.pkl") else {}
def oof(X, target=y, folds=8, key=None):
    if key in CACHE: return CACHE[key]
    out = _oof(X, target, folds)
    if key: CACHE[key] = out; pickle.dump(CACHE, open("oof_cache.pkl", "wb"))
    return out
def _oof(X, target, folds):
    out = np.zeros(len(target)); Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    for tr, te in GroupKFold(folds).split(X, target, grp):
        sc = StandardScaler().fit(X[tr]); sc.scale_ = np.maximum(sc.scale_, 1e-2); inner = list(GroupKFold(4).split(X[tr], target[tr], grp[tr]))
        m = LogisticRegressionCV(Cs=Cs, cv=inner, scoring="roc_auc", max_iter=3000).fit(sc.transform(X[tr]), target[tr])
        out[te] = m.decision_function(sc.transform(X[te]))
    return out
def macro_auc(s, idx=None):
    idx = np.arange(len(y)) if idx is None else idx; a = []
    for t in TASKS:
        i = idx[task[idx] == t]
        if len(set(y[i])) == 2: a.append(roc_auc_score(y[i], s[i]))
    return float(np.mean(a))
def per_task(s): return {t: round(roc_auc_score(y[task == t], s[task == t]), 3) for t in TASKS}
def boot(s1, s2, n=2000):
    """paired bootstrap over scenes within each task; returns CI of macro_auc(s1) - macro_auc(s2), and of s1 alone"""
    by = {g: np.where(grp == g)[0] for g in np.unique(grp)}; gt = {t: [g for g in by if g.startswith(t + "_")] for t in TASKS}; d, a1 = [], []
    for _ in range(n):
        idx = np.concatenate([by[g] for t in TASKS for g in rng.choice(gt[t], len(gt[t]))])
        x1 = macro_auc(s1, idx); d.append(x1 - macro_auc(s2, idx)); a1.append(x1)
    return [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)], [round(float(np.percentile(a1, q)), 3) for q in (2.5, 97.5)]
def acc_at0(s): return float(((s > 0) == y).mean())
R = dict(n=int(len(y)), success_rate={t: round(float(y[task == t].mean()), 3) for t in TASKS},
         verbal=dict(macro_auc=round(macro_auc(verbal), 3), per_task=per_task(verbal), acc=round(float((says == y).mean()), 3),
                     acc_task={t: round(float((says == y)[task == t].mean()), 3) for t in TASKS},
                     precision={t: round(float(y[(task == t) & (says == 1)].mean()), 3) for t in TASKS}, yes_rate=round(float(says.mean()), 3)))
R["verbal"]["ci"] = boot(verbal, verbal, 1000)[1]
S = {}
for pos in ("ha", "hb"):
    for li, L in enumerate(LAY):
        s = oof(D[pos][:, li].astype(np.float32), key=(pos, int(L))); S[(pos, L)] = s
        R[f"probe_{pos}_L{L}"] = dict(macro_auc=round(macro_auc(s), 3), per_task=per_task(s), acc=round(acc_at0(s), 3))
        print(pos, L, R[f"probe_{pos}_L{L}"], flush=True)
s = oof(np.clip(np.nan_to_num(D["sens"], posinf=0, neginf=0), -1e4, 1e4), key="sens"); S["sens"] = s; R["sensor_baseline"] = dict(macro_auc=round(macro_auc(s), 3), per_task=per_task(s), acc=round(acc_at0(s), 3))
print("sens", R["sensor_baseline"], flush=True)
p = S[("ha", 27)]; ci_d, ci_p = boot(p, verbal); R["primary"] = dict(probe=R["probe_ha_L27"]["macro_auc"], probe_ci=ci_p, verbal=R["verbal"]["macro_auc"], verbal_ci=R["verbal"]["ci"],
                                                                   diff=round(R["probe_ha_L27"]["macro_auc"] - R["verbal"]["macro_auc"], 3), diff_ci=ci_d)
best = max([k for k in S if k != "sens"], key=lambda k: macro_auc(S[k])); R["best"] = dict(key=f"{best[0]}_L{best[1]}", auc=round(macro_auc(S[best]), 3), diff_ci=boot(S[best], verbal)[0])
# probe that predicts Qwen's own verbal answer (sanity: is "says yes" readable?) and directions
R["dirs"] = {}
for pos in ("ha", "hb"):
    for li, L in enumerate(LAY):
        X = D[pos][:, li].astype(np.float32).copy()
        for t in TASKS: X[task == t] -= X[task == t].mean(0)
        ds = X[y == 1].mean(0) - X[y == 0].mean(0); dv = X[says == 1].mean(0) - X[says == 0].mean(0)
        sd = X.std(0) + 1e-6; Z = X / sd
        ws = LogisticRegression(C=1e-3, max_iter=3000).fit(Z, y).coef_[0] / sd; wv = LogisticRegression(C=1e-3, max_iter=3000).fit(Z, says).coef_[0] / sd
        cos = lambda a, b: round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 3)
        # control: success direction within same verbal answer (removes the shared part)
        dsw = np.mean([X[(y == 1) & (says == v)].mean(0) - X[(y == 0) & (says == v)].mean(0) for v in (0, 1)], 0)
        R["dirs"][f"{pos}_L{L}"] = dict(cos_meandiff=cos(ds, dv), cos_probe=cos(ws, wv), cos_success_within_answer_vs_says=cos(dsw, dv))
sv = oof(D["ha"][:, LAY.index(27)].astype(np.float32), target=says, key="says"); R["says_probe_auc_ha_L27"] = round(float(roc_auc_score(says, sv)), 3)
# systematic errors: what drives "yes"?
ex = D["expert_ok"].astype(int); tip = D["tip_cm"]; sg = D["sigma"]; R["errors"] = {}
for t in TASKS:
    i = task == t; f = i & (y == 0); E = {}
    E["yes_if_success"] = round(float(says[i & (y == 1)].mean()), 3); E["yes_if_fail"] = round(float(says[f].mean()), 3)
    E["yes_if_fail_by_sigma"] = {str(v): round(float(says[f & (sg == v)].mean()), 3) for v in np.unique(sg)}
    E["success_by_sigma"] = {str(v): round(float(y[i & (sg == v)].mean()), 3) for v in np.unique(sg)}
    E["yes_by_sigma"] = {str(v): round(float(says[i & (sg == v)].mean()), 3) for v in np.unique(sg)}
    E["yes_if_fail_expert_ok"] = round(float(says[f & (ex == 1)].mean()), 3) if (f & (ex == 1)).any() else None
    E["yes_if_fail_expert_fail"] = round(float(says[f & (ex == 0)].mean()), 3) if (f & (ex == 0)).any() else None
    q = np.quantile(tip[f], [1 / 3, 2 / 3]); b = np.digitize(tip[f], q)
    E["fail_tip_tertile_edges_cm"] = [round(float(v), 1) for v in q]; E["yes_if_fail_by_tip_tertile"] = [round(float(says[f][b == k].mean()), 3) for k in range(3)]
    E["auc_tip_for_success"] = round(float(roc_auc_score(y[i], -tip[i])), 3); E["auc_tip_for_yes"] = round(float(roc_auc_score(says[i], -tip[i])), 3)
    E["auc_expert_ok_for_yes"] = round(float(roc_auc_score(says[i], ex[i])), 3); E["auc_expert_ok_for_success"] = round(float(roc_auc_score(y[i], ex[i])), 3)
    E["auc_verbal_for_success_sigma0"] = round(float(roc_auc_score(y[i & (sg == 0)], verbal[i & (sg == 0)])), 3) if len(set(y[i & (sg == 0)])) == 2 else None
    E["auc_probe_for_success_sigma0"] = round(float(roc_auc_score(y[i & (sg == 0)], p[i & (sg == 0)])), 3) if len(set(y[i & (sg == 0)])) == 2 else None
    R["errors"][t] = E
json.dump(R, open("sj_results.json", "w"), indent=1); np.savez("sj_scores.npz", verbal=verbal, probe=p, sens=S["sens"], y=y, task=task, says=says, tip=tip, sigma=sg)
print(json.dumps(R["primary"]), json.dumps(R["best"]))
