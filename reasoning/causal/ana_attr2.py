import json, numpy as np, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "attr32d"
P = json.load(open(f"{OUT}/patch.json")); I = {it["id"]: it for it in json.load(open("items_attr2.json"))}
rng = np.random.default_rng(0); B = 5000
def units(task, key):
    u = {}
    for k, d in P.items():
        it = I[int(k)]
        if it["task"] not in task: continue
        g = d["ldt"] - d["ld0"]
        if abs(g) < 1 or key not in d: continue
        u.setdefault(it["unit"], []).append((d[key] - d["ld0"]) / g)
    return {k: np.mean(v) for k, v in u.items()}
def val(task, sk, band):
    if sk == "R":
        rs = [units(task, f"swap|{band}|R{r}") for r in range(10)]; ks = rs[0].keys(); return {k: np.mean([r[k] for r in rs]) for k in ks}
    return units(task, f"swap|{band}|{sk}")
def boot(x): x = np.array(list(x.values())); return x.mean(), np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
def ci(a): return f"{100*np.mean(a):.0f} ({100*np.percentile(a,2.5):.0f} to {100*np.percentile(a,97.5):.0f})"
BODY = {"reach", "adjacent", "posture"}; res = {}
for band in ["q1", "q2", "q3", "q4"]:
    line = [band]
    for sk in ["full", "PUd", "OUd", "PUl", "OUl", "R"]:
        for nm, t in [("abc", BODY), ("reach", {"reach"}), ("adj", {"adjacent"}), ("pos", {"posture"}), ("let", {"letter"})]:
            m, b = boot(val(t, sk, band)); res[f"{band}|{sk}|{nm}"] = (m, list(np.percentile(b, [2.5, 97.5])))
            line.append(f"{sk}.{nm}={100*m:.0f}")
    print(" ".join(line))
    A, O, R = val(BODY, "PUd", band), val(BODY, "OUd", band), val(BODY, "R", band); ks = sorted(A)
    a, o, r = (np.array([d[k] for k in ks]) for d in (A, O, R)); ix = rng.integers(0, len(ks), (B, len(ks)))
    b_abc = a[ix].mean(1); b_o = o[ix].mean(1); b_r = r[ix].mean(1); _, b_let = boot(val({"letter"}, "PUd", band))
    for nm, d in [("abc-2let", b_abc - 2 * b_let), ("abc-2obj", b_abc - 2 * b_o), ("abc-2rand", b_abc - 2 * b_r), ("abc-let", b_abc - b_let)]:
        res[f"{band}|bar|{nm}"] = ci(d); print("  ", band, nm, ci(d))
n = {t: len(val({t}, "PUd", "q3")) for t in ["reach", "adjacent", "posture", "letter"]}; print("units kept", n)
res["n"] = n; json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1, default=float)
