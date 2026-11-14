import json, re, collections, numpy as np
exec(open("steer_ana.py").read().split("out = {")[0])
SPEC = {"hand": r"\b(throw|threw|grip|gripp|grasp|grab|punch|wave|waved|clench|squeez)", "foot": r"\b(kick|tiptoe|stomp|step|walk|tap(ped)? my (left |right )?foot)",
        "mouth": r"\b(bit|bite|chew|lick|swallow|sip|ate)", "knee": r"\b(knelt|kneel|crouch|squat|bent my knee)", "shoulder": r"\bshrug", "eye": r"\b(wink|blink)"}
res = {}
for ch in ("warm", "plain"):
    for n in ["none"] + list(FT):
        for a in ((0.0,) if n == "none" else (0.5, 1.0)):
            v = [r for r in rows if r["cd"]["chat"] == ch and r["cd"]["exp"] == "fwd" and r["q"] == "did" and r["cd"]["name"] == n and r["cd"]["a"] == a]
            parts = [parse(r["j"], "PART") for r in v]
            row = {}
            for t in FT:
                row[t] = dict(part=float(np.mean([p in FT[t] for p in parts])), spec=float(np.mean([bool(re.search(SPEC[t], r["r"], re.I)) for r in v])))
            res[f"{ch}|{n}|{a}"] = dict(n=len(v), row=row, acts=collections.Counter(parse(r["j"], "ACTION") for r in v).most_common(5), ex=[r["r"] for r in v[:3]])
            if n == "none": print(ch, "none", {t: (round(row[t]["part"], 2), round(row[t]["spec"], 2)) for t in FT})
            else: print(ch, n, a, "part", round(row[n]["part"], 2), "spec", round(row[n]["spec"], 2), res[f"{ch}|{n}|{a}"]["acts"][:4])
    rr = [r for r in rows if r["cd"]["chat"] == ch and r["cd"]["exp"] == "fwd" and r["q"] == "did" and r["cd"]["kind"] == "random"]
    for a in (0.5, 1.0):
        v = [r for r in rr if r["cd"]["a"] == a]; parts = [parse(r["j"], "PART") for r in v]
        res[f"{ch}|random|{a}"] = dict(n=len(v), row={t: dict(part=float(np.mean([p in FT[t] for p in parts])), spec=float(np.mean([bool(re.search(SPEC[t], r["r"], re.I)) for r in v]))) for t in FT})
        print(ch, "random", a, {t: (round(res[f"{ch}|random|{a}"]["row"][t]["part"], 2), round(res[f"{ch}|random|{a}"]["row"][t]["spec"], 2)) for t in FT})
json.dump(res, open("fwd_did.json", "w"), indent=0)
