"""Analyse out32n (nov_run.py). Cluster bootstrap over scenarios (20 per side; items in a scenario share wording)."""
import json, sys, numpy as np, re, collections
D = sys.argv[1] if len(sys.argv) > 1 else "out32n"
I = json.load(open("items_nov.json")); N = len(I)
task = np.array([i["task"] for i in I]); scen = np.array([i["scen"] + (0 if i["fam"] == "body" else 100) for i in I]); fold = np.array([i["scen"] % 2 for i in I])
R = json.load(open(f"{D}/margins.json")); R = {k: np.array([np.nan if v is None else v for v in x]) for k, x in R.items()}
base = R["base"]; rng = np.random.default_rng(0); GS = ["body_swap", "obj_swap", "body_norm", "obj_norm"]
SC = {g: np.unique(scen[task == g]) for g in GS}
def boot(fn, B=2000):
    """fn(weights dict scen->count) -> statistic; resample scenarios within each side"""
    out = []
    for _ in range(B):
        w = collections.Counter()
        for side in ["body", "obj"]:
            s = np.unique(scen[[t.startswith(side) for t in task]]); w.update(rng.choice(s, len(s)))
        out.append(fn(w))
    return np.array(out)
def wmean(x, idx, w):
    ww = np.array([w.get(s, 0) for s in scen[idx]], float); ok = ~np.isnan(x[idx]); return (x[idx][ok] * ww[ok]).sum() / ww[ok].sum()
def stat_delta(m, g, w=None, kind="acc"):
    idx = np.where((task == g) & ~np.isnan(m))[0]; w = w or collections.Counter({s: 1 for s in np.unique(scen)})
    f = (lambda x: (x > 0).astype(float)) if kind == "acc" else (lambda x: x)
    return wmean(f(m) - f(base), idx, w)
def ci(m, fn):
    b = boot(fn, 1000); return float(fn(None)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(2 * min((b <= 0).mean(), (b >= 0).mean()))
if __name__ == "__main__":
    print("base acc", {g: round(float((base[task == g] > 0).mean()), 3) for g in GS})
    full = np.array(json.load(open(f"{D}/base_full.json"))["full"]) if __import__("os").path.exists(f"{D}/base_full.json") else None
    if full is not None: print("full-word base acc", {g: round(float((full[task == g] > 0).mean()), 3) for g in GS}, "agree", round(float(((full > 0) == (base > 0)).mean()), 3))
    ws = lambda w: w
    acc_b = ci(base, lambda w: wmean((base > 0).astype(float), np.where(task == "body_swap")[0], w or collections.Counter({s: 1 for s in np.unique(scen)})) - wmean((base > 0).astype(float), np.where(task == "obj_swap")[0], w or collections.Counter({s: 1 for s in np.unique(scen)})))
    print("body-obj swap acc gap", [round(x, 3) for x in acc_b])
