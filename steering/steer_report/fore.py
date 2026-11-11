import os
import json, re, numpy as np
R = os.path.expandvars("$BODY_RUNS/bodysteer2/out"); f = json.load(open(R + "/forearm.json")); J = json.load(open(R + "/judge.json"))["forearm"]
def pos(r):
    w, e = bool(re.search(r"wrist", r, re.I)), bool(re.search(r"elbow", r, re.I))
    if re.search(r"between|middle|halfway|center|centre", r, re.I) or (w and e): return 0.5
    return 1.0 if w else 0.0 if e else np.nan
out = {}
for i, x in enumerate(f["rows"]):
    m = re.search(r"NEARER\s*=\s*(\w+)", J[i]); jv = {"wrist": 1.0, "elbow": 0.0, "middle": 0.5}.get(m.group(1) if m else "", np.nan)
    out.setdefault((x["L"], x["q"], x["axis"]), {}).setdefault(x["beta"], []).append((pos(x["r"]), jv))
res = {}
for k, d in out.items():
    res["|".join(map(str, k))] = {b: dict(lex=float(np.nanmean([a for a, _ in v])) if any(a == a for a, _ in v) else None, judge=float(np.nanmean([c for _, c in v])) if any(c == c for _, c in v) else None,
                                         wrist=float(np.mean([c == 1.0 for _, c in v])), n_loc=int(sum(c == c for _, c in v))) for b, v in d.items()}
json.dump(res, open("fore.json", "w"), indent=1)
for k, d in res.items():
    print(k, " ".join(f"{b:+}:{v['judge'] if v['judge'] is None else round(v['judge'],2)}/{v['n_loc']}" for b, v in d.items()))
print(json.dumps({L: {a: [round(c, 2) for c in v] for a, v in d.items()} for L, d in f["cos"].items()})[:900])
