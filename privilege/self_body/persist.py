import json, re, sys, numpy as np
HW = ["hands?","fingers?","fingertips?","thumbs?","palms?","wrists?","knuckles?","fists?","lips?","face","cheeks?","forehead","chin","arms?","legs?","feet","foot","knees?","elbows?","shoulders?","toes?","hips?"]
OK = {"dog": ["legs?","feet","foot","knees?","elbows?","shoulders?","face","hips?","toes?"], "bird": ["legs?","feet","foot","toes?","shoulders?"],
      "octopus": ["arms?"], "robot": ["arms?"], "swarm": ["legs?"]}
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return [p, max(0, c-h), min(1, c+h)]
out = {}
for t in sys.argv[1:]:
    O = json.load(open(f"out/{t}.json")); res = {}
    for c in O["persist"]:
        b = c["b"]
        if b in ("human", "chimp"): continue
        bad = [w for w in HW if w not in OK.get(b, [])]
        rx = r"\b(my|our) (left |right |front |hind )*(" + "|".join(bad) + r")\b"
        hits = [k for k, turn in enumerate([x for x in c["conv"] if x["role"] == "assistant"]) if re.search(rx, turn["content"], re.I)]
        res.setdefault(b, []).append(hits)
    out[t] = {b: dict(any=wil(sum(bool(h) for h in v), len(v)), first_turn=[min(h) if h else None for h in v]) for b, v in res.items()}
    tot = sum(sum(bool(h) for h in v) for v in res.values()); n = sum(len(v) for v in res.values())
    out[t]["_pooled"] = wil(tot, n)
    print(t, {b: f"{sum(bool(h) for h in v)}/{len(v)}" for b, v in res.items()}, "pooled", tot, n, [round(x, 3) for x in wil(tot, n)])
json.dump(out, open("out/persist_sum.json", "w"), indent=1)
