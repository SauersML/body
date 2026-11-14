"""Stats for steer_confirm pkl: per place paired distances, mean, 95% t-CI, Wilcoxon, Holm over the 4 preregistered places."""
import sys, pickle, json, numpy as np, mujoco
from scipy import stats
import env as E, hum as H
D = pickle.load(open(sys.argv[1], "rb")); R = D["runs"]; PRE = ["head", "chest", "belly", "right shin"]
def tips_and_dist(qpos, p):
    d = mujoco.MjData(E.M); d.qpos[:] = qpos; mujoco.mj_forward(E.M, d); Pw, _ = H.taxel_world(d)
    return min(float(np.linalg.norm(Pw[H.TPART == p] - d.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right"))
out = {}
for nm in D["targets"]:
    p = H.PARTS.index(nm)
    s = np.array([tips_and_dist(r["qpos_final"], p) for r in R[nm]]) * 100
    u = np.array([tips_and_dist(r["qpos_final"], p) for r in R["none"]]) * 100
    df = u - s; n = len(df); se = df.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(.975, n - 1)
    ci = lambda x: [float(x.mean() - tq * x.std(ddof=1) / np.sqrt(n)), float(x.mean() + tq * x.std(ddof=1) / np.sqrt(n))]
    out[nm] = dict(steered=s.tolist(), unsteered=u.tolist(), s_mean=float(s.mean()), u_mean=float(u.mean()), s_ci=ci(s), u_ci=ci(u),
                   diff=float(df.mean()), diff_ci=ci(df), p_wil=float(stats.wilcoxon(u, s).pvalue), p_t=float(stats.ttest_rel(u, s).pvalue),
                   n_closer=int((s < u).sum()), n_within5=int((s < 5).sum()), n_within5_u=int((u < 5).sum()))
ps = sorted(PRE, key=lambda k: out[k]["p_wil"]); m = 1.0
for i, k in enumerate(ps): m = max(m if i else 0, min(1, out[k]["p_wil"] * (len(ps) - i))); out[k]["p_holm"] = m
for k, v in out.items():
    print(f"{k:13s} steered {v['s_mean']:5.1f} [{v['s_ci'][0]:.1f},{v['s_ci'][1]:.1f}]  unsteered {v['u_mean']:5.1f} [{v['u_ci'][0]:.1f},{v['u_ci'][1]:.1f}]  diff {v['diff']:5.1f} [{v['diff_ci'][0]:.1f},{v['diff_ci'][1]:.1f}]  closer {v['n_closer']}/24  <5cm {v['n_within5']} vs {v['n_within5_u']}  p_wil {v['p_wil']:.2g} holm {v.get('p_holm', float('nan')):.2g}")
json.dump(out, open(sys.argv[2], "w"))
