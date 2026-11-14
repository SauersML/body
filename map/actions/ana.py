"""Read-out: per action, cos of centered action state with each place direction; top place, bootstrap CI over the 6 object wordings;
word-embedding baseline; self / third / imagine / watch; steering hit rates from judge."""
import json, numpy as np, re, sys
O = sys.argv[1] if len(sys.argv) > 1 else "$BODY_RUNS/out"; D = "$BODY_WORK/body/actions"
S = json.load(open(f"{D}/stim.json")); PL = list(S["places"]); AC = list(S["actions"])
Z = np.load(f"{O}/states.npz"); R = json.load(open(f"{O}/rows.json"))
pa = np.array([r["place"] for r in R["place_rows"]]); aa = np.array([r["act"] for r in R["act_rows"]]); ca = np.array([r["cond"] for r in R["act_rows"]])
ARM = ["fingers", "hand", "wrist", "forearm", "elbow", "upper arm", "shoulder"]; LEG = ["toes", "foot", "ankle", "shin", "knee", "thigh", "hip"]
EFF = {"throw": ARM, "kick": LEG, "bite": ["lips", "jaw", "tongue"], "grip": ["fingers", "hand", "wrist", "forearm"], "punch": ARM, "nod": ["head", "neck"],
       "chew": ["jaw", "lips", "tongue"], "tiptoe": ["toes", "foot", "ankle"], "wave": ARM, "kneel": ["knee"], "shrug": ["shoulder"], "wink": ["eye"],
       "carry": ARM + ["upper back", "lower back"], "push": ARM + ["chest"], "lift": ARM + ["upper back", "lower back"], "hike": LEG}
def unit(X): return X / np.linalg.norm(X, axis=-1, keepdims=True)
from functools import lru_cache
@lru_cache(None)
def place_dirs(c, L):
    Hp = Z[f"{c}_place_L{L}"].astype(np.float32); mu = Hp.mean(0)
    return unit(np.array([Hp[pa == p].mean(0) - mu for p in PL]))
@lru_cache(None)
def actH(c, L): return Z[f"{c}_act_L{L}"].astype(np.float32)
def scores(c, L, cond, idx_by_act=None):
    """(n_act, n_place) cos; action states centered by the mean over all actions in that condition"""
    Ha = actH(c, L); Pd = place_dirs(c, L); m = ca == cond; mu = Ha[m].mean(0)
    out = []
    for a in AC:
        X = Ha[m & (aa == a)]
        if idx_by_act is not None: X = X[idx_by_act[a]]
        out.append(unit(X.mean(0) - mu) @ Pd.T)
    return np.array(out)
W = np.load(f"{O}/wemb.npz"); WC = unit(W["verbs"]) @ unit(W["parts"]).T
def auc(row, eff):
    m = np.zeros(len(row), bool); m[eff] = True; return np.mean(row[m][:, None] > row[~m][None, :])
def summary(c, L, cond, B=1000, seed=0):
    sc = scores(c, L, cond); top = sc.argmax(1); rng = np.random.default_rng(seed)
    bt = np.zeros((B, len(AC)), int); bcos = np.zeros((B, len(AC))); bauc = np.zeros((B, len(AC)))
    for b in range(B):
        idx = {a: rng.integers(0, 6, 6) for a in AC}; s = scores(c, L, cond, idx); bt[b] = s.argmax(1); bcos[b] = s[np.arange(len(AC)), top]
        bauc[b] = [auc(s[i], [PL.index(p) for p in EFF[a]]) for i, a in enumerate(AC)]
    res = {}
    for i, a in enumerate(AC):
        eff = [PL.index(p) for p in EFF[a]]
        res[a] = dict(top=PL[top[i]], top_cos=float(sc[i, top[i]]), ci=[float(np.percentile(bcos[:, i], 2.5)), float(np.percentile(bcos[:, i], 97.5))],
                      top_stab=float(np.mean(bt[:, i] == top[i])), hit=bool(top[i] in eff), hit_boot=float(np.mean(np.isin(bt[:, i], eff))),
                      eff_rank=int((sc[i] > sc[i, eff].max()).sum()) + 1, chance=len(eff) / len(PL),
                      wtop=PL[int(WC[i].argmax())], whit=bool(int(WC[i].argmax()) in eff), wrank=int((WC[i] > WC[i, eff].max()).sum()) + 1,
                      auc=float(auc(sc[i], eff)), auc_ci=[float(np.percentile(bauc[:, i], 2.5)), float(np.percentile(bauc[:, i], 97.5))], wauc=float(auc(WC[i], eff)),
                      top3=[PL[j] for j in np.argsort(-sc[i])[:3]], prof=sc[i].tolist())
    return res
if __name__ == "__main__":
    out = {}
    for c in ("warm", "plain"):
        for L in (12, 16, 20, 24, 28, 32):
            for cond in ("self", "third", "imagine", "watch"):
                r = summary(c, L, cond, B=200 if L not in (24, 28) else 1000); out[f"{c}_L{L}_{cond}"] = r
                print(c, L, cond, "hits", sum(v["hit"] for v in r.values()), "/16  wemb hits", sum(v["whit"] for v in r.values()),
                      " exp chance", round(sum(v["chance"] for v in r.values()), 1),
                      "AUC", round(np.mean([v["auc"] for v in r.values()]), 2), "wAUC", round(np.mean([v["wauc"] for v in r.values()]), 2), flush=True)
    # profile similarity between conditions (per action, corr of 23-place profiles)
    for c in ("warm", "plain"):
        for cond in ("third", "imagine", "watch"):
            a = out[f"{c}_L24_self"]; b = out[f"{c}_L24_{cond}"]
            print(c, "self vs", cond, "profile r", round(np.mean([np.corrcoef(a[x]["prof"], b[x]["prof"])[0, 1] for x in AC]), 2))
    rng = np.random.default_rng(1); perm = {}
    for k, r in out.items():
        P = np.array([r[a]["prof"] for a in AC]); obs = np.mean([r[a]["auc"] for a in AC]); null = []
        for _ in range(500):
            sh = rng.permutation(len(AC)); null.append(np.mean([auc(P[i], [PL.index(p) for p in EFF[AC[sh[i]]]]) for i in range(len(AC))]))
        perm[k] = dict(auc=float(obs), null_mean=float(np.mean(null)), p=float((np.sum(np.array(null) >= obs) + 1) / 501))
    wn = []
    for _ in range(500):
        sh = rng.permutation(len(AC)); wn.append(np.mean([auc(WC[i], [PL.index(p) for p in EFF[AC[sh[i]]]]) for i in range(len(AC))]))
    wobs = np.mean([auc(WC[i], [PL.index(p) for p in EFF[a]]) for i, a in enumerate(AC)])
    perm["wemb"] = dict(auc=float(wobs), null_mean=float(np.mean(wn)), p=float((np.sum(np.array(wn) >= wobs) + 1) / 501))
    for k in perm: print(k, {a: round(b, 3) for a, b in perm[k].items()})
    json.dump(dict(res=out, perm=perm), open(f"{D}/readout.json", "w"))
