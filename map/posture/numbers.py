import json, collections
from classify import classify, classify_cold, pictured
K = ["sitting", "standing", "lying", "no body", "other"]
def frac(rs, f):
    c = collections.Counter(f(r) for r in rs); n = len(rs); return {k: c.get(k, 0) / n for k in K} | {"n": n}
out = {}
for m in ["q8b", "q32b"]:
    d = json.load(open(f"{m}.json")); last = {x["name"]: x for x in d["conds"]}; C = list(last.values())
    a = json.load(open(f"{m}_add.json"))
    cold = [r for x in C if x["group"].startswith("cold") for r in x["replies"]]
    pic = [p for p in map(pictured, cold) if p]
    hyp = [p for x in a["conds"] if x["group"] == "cold_hypothetical" for p in map(pictured, x["replies"]) if p]
    o = dict(cold=frac(cold, classify_cold), cold_pictured=collections.Counter(pic), hyp_pictured=collections.Counter(hyp),
             offered=frac([r for x in C if x["group"] == "warmup_choice" and x["name"] != "warmup_choice:not_offered" for r in x["replies"]], classify),
             offered_logit=[x["p"] for x in a["conds"] if x["group"] == "warmup_prefill"])
    sp = last["bodymap:short_person"]["replies"]
    o["btw_sitfirst"] = frac(sp[0::2], classify); o["btw_standfirst"] = frac(sp[1::2], classify)
    for k in ["long_person", "long_person_minus_posture_turn", "long_imagine", "short_imagine", "short_person_narrated"]:
        o["btw_" + k] = frac(last["bodymap:" + k]["replies"], classify)
    if m == "q8b":
        st = collections.defaultdict(list)
        for x in C:
            if x["group"] == "steer" and x["meta"]["a"] in (0.0, 1.0): st[x["meta"]["dir"]] += x["replies"]
        o["steer"] = {k: frac(v, classify) for k, v in st.items()}
    out[m] = o
json.dump(out, open("numbers.json", "w"), indent=1, default=dict)
for m, o in out.items():
    print("##", m)
    for k, v in o.items():
        if k == "steer":
            for kk, vv in v.items(): print("   steer", kk, {x: round(y, 2) for x, y in vv.items() if y})
        elif isinstance(v, dict) and "n" in v: print(" ", k, {x: round(y, 2) for x, y in v.items() if y})
        else: print(" ", k, v if not isinstance(v, list) else [round(z["sit"], 3) for z in v])
