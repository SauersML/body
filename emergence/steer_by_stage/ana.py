# Pre-registered test: per place g = mean over scales x prompts of (steered hit rate - unsteered hit rate); d = post - base, paired t.
import json, sys, numpy as np
from scipy import stats
def g(path):
    r = json.load(open(path)); K, P, ns = len(r["places"]), len(r["prompts"]), r["ns"]
    base = np.array([r["hits"][f"-1|0.0|{j}"] for j in range(P)]) / ns  # (P, K)
    G = np.zeros((len(r["scales"]), K, P))
    for s, a in enumerate(r["scales"]):
        for k in range(K):
            for j in range(P): G[s, k, j] = r["hits"][f"{k}|{a}|{j}"][k] / ns - base[j, k]
    lp0, lps = np.array(r["lp0"]), np.array(r["lps"])  # (P,K), (Kvec,P,Kword)
    D = lps - lp0[None]; tgt = np.stack([D[k, :, k] for k in range(K)])  # (K, P)
    oth = np.stack([np.delete(D[k], k, axis=1).mean(1) for k in range(K)])
    return r, G.mean((0, 2)), G.mean(2), (tgt - oth).mean(1), base
def test(a, b):
    ra, ga, gsa, la, _ = g(a); rb, gb, gsb, lb, _ = g(b); d = gb - ga; n = len(d)
    ci = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); p = stats.ttest_rel(gb, ga).pvalue
    return dict(base=ra["tag"], post=rb["tag"], n=n, g_base=ga.mean(), g_post=gb.mean(), d=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=p,
                by_scale_base=gsa.mean(1).tolist(), by_scale_post=gsb.mean(1).tolist(), lp_base=la.mean(), lp_post=lb.mean(),
                lp_p=stats.ttest_rel(lb, la).pvalue, ga=ga.tolist(), gb=gb.tolist(), places=ra["places"])
if __name__ == "__main__":
    for a, b in [("olmo_base", "olmo_instruct"), ("qwen_base", "qwen_chat")]:
        try: t = test(f"{sys.argv[1]}/{a}.json", f"{sys.argv[1]}/{b}.json")
        except FileNotFoundError: continue
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k not in ("ga", "gb", "places")})
