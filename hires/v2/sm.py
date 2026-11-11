import numpy as np
def spot_means(X, rows, nprobe, tpl_subset=None):
    kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows])
    T = sorted(set(tp)) if tpl_subset is None else tpl_subset
    n = li[kind == "loc"].max() + 1; d = X.shape[1]
    A = np.zeros((n, d)); W = np.zeros((nprobe, d)); isloc = kind == "loc"
    A[li[isloc]]  # all spots present
    pk = [r["kind"] for r in rows if r["kind"] != "loc"]; kinds = sorted(set(pk))
    for t in T:
        sel = tp == t; mu = X[sel & isloc].mean(0)
        a = X[sel & isloc] - mu; A[li[sel & isloc]] += a / len(T)
        sk = sel & ~isloc; np.add.at(W, li[sk], (X[sk] - mu) / len(T))
    return A, W
