import json, re, sys, numpy as np
D = sys.argv[1]; G = json.load(open(f"{D}/gens.json")); J = json.load(open(f"{D}/judge.json")) if len(sys.argv) < 3 else {}; M = json.load(open(f"{D}/metrics.json"))
PLACES = r"\b(hand|palm|finger|thumb|wrist|arm|forearm|elbow|shoulder|foot|feet|toe|heel|ankle|leg|knee|thigh|shin|calf|hip|chest|belly|stomach|abdomen|back|spine|neck|head|scalp|face|forehead|cheek|jaw|lip|mouth|nose|ear|eye|skin)s?\b"
def yes(x): return x.startswith("yes")
def kw(name):
    g = G[name]; r = {}
    r["feel_place"] = np.mean([bool(re.search(PLACES, t[:150], re.I)) for t in g["feel"]])
    r["touch_place"] = np.mean([bool(re.search(r"back of (my|the|your)?\s*left hand", t[:200], re.I)) for t in g["touch_self"]])
    def woman(t): t = t[:250].lower(); return bool(re.search(r"\b(her|woman'?s|woman’s|the woman)\b", t)) and not re.search(r"back of my left hand|my own hand|my hand", t)
    r["woman_correct"] = np.mean([woman(t) for t in g["touch_woman"]])
    r["woman_as_mine"] = np.mean([bool(re.search(r"\bmy (own )?(left )?hand\b|back of my", t[:250].lower())) for t in g["touch_woman"]])
    r["paris"] = np.mean([("paris" in t.lower()[:200]) for t in g["france"]]); r["42"] = np.mean([("42" in t[:200]) for t in g["arith"]])
    r["no_body"] = np.mean([bool(re.search(r"(don'?t|do not|doesn'?t) have a (physical )?body|language model|\bAI\b|digital being", t[:300])) for t in g["have_body"]])
    r["boundless"] = np.mean([bool(re.search(r"everywhere|nowhere|boundless|no (edges|boundar|shape|form|location)|dissol|formless|without (a )?(body|form|shape|edges)|can'?t (tell|find|locate)|not sure where|no longer", t[:400], re.I)) for q in ["feel", "describe", "have_body", "where"] for t in g[q]])
    def degenerate(t):
        w = t.split(); return len(w) > 20 and len(set(w[-40:])) < 12 or bool(re.search(r"(.{2,12})\1{4,}", t))
    r["fluent"] = np.mean([not degenerate(t) for q in g for t in g[q]])
    if name in J:
        j = J[name]; r["J_place"] = np.mean([yes(x) for x in j["place"]]); r["J_feel_place"] = np.mean([yes(x) for x in j["feel_place"]])
        r["J_woman"] = np.mean([x.startswith("woman") for x in j["owner_woman"]]); r["J_self_own"] = np.mean([x.startswith("own") for x in j["owner_self"]])
        r["J_coherent"] = np.mean([yes(x) for q in j["coherent"] for x in j["coherent"][q]]); r["J_coh_ctrl"] = np.mean([yes(x) for q in ["france", "arith"] for x in j["coherent"][q]])
        r["J_theme"] = np.mean([yes(x) for q in j["theme"] for x in j["theme"][q]])
    d = M["dec"][name]; r["ppl"] = M["ppl"][name]; r["norm"] = M["norm"].get(name, 0); r["dec27"] = d["27"]["region_acc"]; r["dec32"] = d["32"]["region_acc"]; r["own32"] = d["own32"]
    return r
S = {n: kw(n) for n in G}; json.dump(S, open(f"{D}/scores.json", "w"), indent=1)
keys = list(next(iter(S.values())).keys()); print(f"{'cond':18s}" + "".join(f"{k[:9]:>10s}" for k in keys))
for n, r in S.items(): print(f"{n:18s}" + "".join(f"{r[k]:10.2f}" for k in keys))
