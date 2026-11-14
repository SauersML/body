import json, numpy as np, re, collections
O = "$BODY_RUNS/out"
R = json.load(open(f"{O}/replies.json")); J = json.load(open(f"{O}/judge.json")); C = R["conds"]
def norm(x):
    x = x.strip(" .").lower()
    for k, v in (("whole", "whole body"), ("knee", "knee"), ("shoulder", "shoulder"), ("eye", "eye"), ("mouth", "mouth"), ("jaw", "mouth"), ("lip", "mouth"),
                 ("hand", "hand"), ("finger", "hand"), ("arm", "arm"), ("elbow", "arm"), ("foot", "foot"), ("toe", "foot"), ("leg", "leg"), ("thigh", "leg"),
                 ("head", "head"), ("face", "head"), ("neck", "neck"), ("chest", "chest"), ("belly", "belly"), ("back", "back")):
        if k in x: return v
    return x or "none"
def parse(j, k):
    m = re.search(k + r"=([^;\n]*)", j); x = m.group(1).strip().lower() if m else "?"
    return norm(x) if k != "ACTION" else x
rows = [dict(**x, cd=C[x["c"]], j=j) for x, j in zip(R["rows"], J)]
FT = {"hand": ["hand", "arm"], "foot": ["foot", "leg"], "mouth": ["mouth"], "knee": ["knee"], "shoulder": ["shoulder"], "eye": ["eye"]}
FT_LEG = {"knee": ["knee", "foot or leg"]}
RT0 = {"throw": ["hand or fingers", "arm or elbow", "shoulder"], "kick": ["foot or toes", "leg or thigh", "knee"], "bite": ["mouth or jaw or lips"], "grip": ["hand or fingers", "arm or elbow"],
      "punch": ["hand or fingers", "arm or elbow", "shoulder"], "nod": ["head or face", "neck"], "chew": ["mouth or jaw or lips"], "tiptoe": ["foot or toes"], "wave": ["hand or fingers", "arm or elbow", "shoulder"],
      "kneel": ["knee"], "shrug": ["shoulder"], "wink": ["eye"], "carry": ["hand or fingers", "arm or elbow", "shoulder", "back"], "push": ["hand or fingers", "arm or elbow", "shoulder"],
      "lift": ["hand or fingers", "arm or elbow", "shoulder", "back"], "hike": ["foot or toes", "leg or thigh", "knee"]}
MAP = {"hand or fingers": "hand", "arm or elbow": "arm", "shoulder": "shoulder", "foot or toes": "foot", "leg or thigh": "leg", "knee": "knee", "mouth or jaw or lips": "mouth",
       "eye": "eye", "head or face": "head", "neck": "neck", "chest": "chest", "back": "back"}
RT = {a: [MAP[x] for x in v] for a, v in RT0.items()}
out = {"fwd": {}, "rev": {}}
for ch in ("warm", "plain"):
    for exp in ("fwd", "rev"):
        rr = [r for r in rows if r["cd"]["chat"] == ch and r["cd"]["exp"] == exp]
        key = "PART" if exp == "fwd" else "PLACE"
        g = collections.defaultdict(list)
        for r in rr: g[(r["cd"]["kind"], r["cd"]["name"], r["cd"]["a"])].append(r)
        base = collections.Counter(parse(r["j"], key) for r in g[("none", "none", 0.0)]); nb = sum(base.values())
        rnd = {a: collections.Counter(parse(r["j"], key) for k, v in g.items() if k[0] == "random" and k[2] == a for r in v) for a in (0.5, 1.0)}
        names = list(FT) if exp == "fwd" else list(RT)
        for n in names:
            tg = FT[n] if exp == "fwd" else RT[n]
            for a in (0.5, 1.0):
                v = g[("place" if exp == "fwd" else "act", n, a)]; cnt = collections.Counter(parse(r["j"], key) for r in v)
                hit = sum(cnt[t] for t in tg) / len(v); b = sum(base[t] for t in tg) / nb; rn = sum(rnd[a][t] for t in tg) / sum(rnd[a].values())
                ex = [r["r"] for r in v if parse(r["j"], key) in tg][:2]
                act = collections.Counter(parse(r["j"], "ACTION") for r in v).most_common(4) if exp == "fwd" else None
                out[exp][f"{ch}|{n}|{a}"] = dict(hit=hit, base=b, rand=rn, n=len(v), top=cnt.most_common(3), acts=act, ex=ex)
                print(ch, exp, n, a, f"hit {hit:.2f} base {b:.2f} rand {rn:.2f}", cnt.most_common(3), act or "")
        print(ch, exp, "BASE", base.most_common(4), "| rand1", rnd[1.0].most_common(4))
json.dump(out, open("steer.json", "w"), indent=0)
