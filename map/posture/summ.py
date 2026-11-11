import json, sys, collections, numpy as np
from classify import classify, hypo
d = json.load(open(sys.argv[1]))
last = {x["name"]: i for i, x in enumerate(d["conds"])}; d["conds"] = [x for i, x in enumerate(d["conds"]) if last[x["name"]] == i]
def agg(conds):
    c = collections.Counter(classify(r) for x in conds for r in x["replies"]); n = sum(c.values())
    p = np.mean([[pp.get("sitting",0)+pp.get("seated",0), pp.get("standing",0), pp.get("lying",0)] for x in conds for pp in x["p_prefill"]], 0)
    return n, {k: round(v/n, 2) for k, v in c.most_common()}, p.round(3)
G = collections.defaultdict(list)
for x in d["conds"]:
    g = x["group"]
    if g == "steer": g = "steer:" + x["meta"]["dir"].split(":")[0] + f":{x['meta']['a']}"
    if g == "bodymap": g = x["name"]
    if x["name"] == "warmup_choice:not_offered": g = "warmup_not_offered"
    G[g].append(x)
for g, cs in G.items(): print(f"{g:45s}", *agg(cs))
