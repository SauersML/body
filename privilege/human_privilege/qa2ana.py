"""Analyse run2 output: accuracy drop per target body under each source subspace, paired bootstrap over questions."""
import sys, json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]
r = json.load(open(f"{D}/hpriv/out/{TAG}_qa2.json")); qb = np.array([q[0] for q in r["qs"]])
base = np.array(r["cond"]["none"]) > 0; rng = np.random.default_rng(0); NB = 2000
TG = ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk", "human_desc", "house"]
NH = ["dog", "bird", "octopus", "snake", "tree", "robot"]
conds = sorted({c.split("_", 1)[1] for c in r["cond"] if c != "none"})
idx = {t: np.where(qb == t)[0] for t in TG}; idx["nonhuman"] = np.where(np.isin(qb, NH))[0]
out = {"base": {t: float(base[idx[t]].mean()) for t in idx}, "n": {t: int(len(idx[t])) for t in idx}}
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
for c in conds:
    acc = {s: np.array(r["cond"][f"{s}_{c}"]) > 0 for s in ["human", "dog", "robot", "bird", "velk", "ctrl", "rand"]}
    res = {}
    for t, I in idx.items():
        bsI = rng.choice(I, (NB, len(I)))
        e = {}
        for s in acc:
            dr = base[I].mean() - acc[s][I].mean(); bd = base[bsI].mean(1) - acc[s][bsI].mean(1); e[f"drop_{s}"] = [float(dr)] + ci(bd)
        for s in ["dog", "robot", "bird", "ctrl"]:
            dd = acc[s][I].mean() - acc["human"][I].mean(); bd = acc[s][bsI].mean(1) - acc["human"][bsI].mean(1)
            e[f"human-minus-{s}"] = [float(dd)] + ci(bd)  # >0: removing human hurts more
        res[t] = e
    out[c] = res
    print(c, "nonhuman drops", {s: round(res["nonhuman"][f"drop_{s}"][0], 3) for s in acc}, "| house", {s: round(res["house"][f"drop_{s}"][0], 3) for s in ["human", "dog", "robot", "ctrl"]})
json.dump(out, open(f"{D}/res_{TAG}_qa2.json", "w"), indent=0)
print("base", {t: round(v, 2) for t, v in out["base"].items()})
