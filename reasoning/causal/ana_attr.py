"""Analyse attribution-over-text runs. python ana_attr.py outdir  -> outdir/stats.json"""
import json, sys, numpy as np, torch
D = sys.argv[1]; rng = np.random.default_rng(0)
P = json.load(open(f"{D}/patch.json")); A = torch.load(f"{D}/atp.pt", weights_only=False)
IT = {it["id"]: it for it in A["items"]}; SN = A["SN"]; RN = [s for s in SN if s.startswith("R")]
ids = [int(k) for k, v in P.items() if abs(v["ldt"] - v["ld0"]) >= 1]
fam = {i: IT[i]["fam"] for i in ids}; task = {i: IT[i]["task"] for i in ids}; half = {i: (i // 2) % 2 for i in ids}
sites = sorted({k.rsplit("|", 1)[0] for k in P[str(ids[0])] if "|" in k})
def fr(i, site, s):
    v = P[str(i)]; return (v[f"{site}|{s}"] - v["ld0"]) / (v["ldt"] - v["ld0"])
F = {i: {site: dict(PU=fr(i, site, "PU"), OU=fr(i, site, "OU"), R=np.mean([fr(i, site, r) for r in RN]), full=fr(i, site, "full"),
                    Rall=[fr(i, site, r) for r in RN]) for site in sites} for i in ids}
def boot(sel, f, n=4000):
    """mean and 95% CI of f(i) over items, resampling pairs (i//2)."""
    pairs = sorted({i // 2 for i in sel}); by = {p: [i for i in sel if i // 2 == p] for p in pairs}
    vals = {p: [f(i) for i in by[p]] for p in pairs}; m = np.mean([x for p in pairs for x in vals[p]]); bs = []
    for _ in range(n):
        s = rng.choice(len(pairs), len(pairs)); xs = [x for j in s for x in vals[pairs[j]]]; bs.append(np.mean(xs))
    return [float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def boot2(sa, fa, sb, fb, n=4000):
    """difference of means (a - b), independent pair resampling."""
    pa = sorted({i // 2 for i in sa}); pb = sorted({i // 2 for i in sb})
    va = {p: [fa(i) for i in sa if i // 2 == p] for p in pa}; vb = {p: [fb(i) for i in sb if i // 2 == p] for p in pb}
    m = np.mean([x for v in va.values() for x in v]) - np.mean([x for v in vb.values() for x in v]); bs = []
    for _ in range(n):
        a = rng.choice(len(pa), len(pa)); b = rng.choice(len(pb), len(pb))
        bs.append(np.mean([x for j in a for x in va[pa[j]]]) - np.mean([x for j in b for x in vb[pb[j]]]))
    bs = np.array(bs); return [float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))]
out = dict(n=dict(body=sum(fam[i] == "body" for i in ids), ctl=sum(fam[i] == "ctl" for i in ids), dropped=len(P) - len(ids)))
# site table, both halves pooled (descriptive)
out["sites"] = {}
for site in sites:
    row = {}
    for fm in ["body", "ctl"]:
        sel = [i for i in ids if fam[i] == fm]
        row[fm] = {k: float(np.mean([F[i][site][k] for i in sel])) for k in ["PU", "OU", "R", "full"]}
    out["sites"][site] = row
# preregistered: choose site on half A body (max PU - R), test on half B
selA = [i for i in ids if fam[i] == "body" and half[i] == 0]
best = max(sites, key=lambda s: np.mean([F[i][s]["PU"] - F[i][s]["R"] for i in selA])); out["chosen_site"] = best
bB = [i for i in ids if fam[i] == "body" and half[i] == 1]; cB = [i for i in ids if fam[i] == "ctl" and half[i] == 1]
pr = dict(PU=boot(bB, lambda i: F[i][best]["PU"]), R=boot(bB, lambda i: F[i][best]["R"]), OU=boot(bB, lambda i: F[i][best]["OU"]),
          full=boot(bB, lambda i: F[i][best]["full"]),
          PU_minus_R=boot(bB, lambda i: F[i][best]["PU"] - F[i][best]["R"]), PU_minus_OU=boot(bB, lambda i: F[i][best]["PU"] - F[i][best]["OU"]),
          ctl_PU=boot(cB, lambda i: F[i][best]["PU"]), ctl_R=boot(cB, lambda i: F[i][best]["R"]), ctl_OU=boot(cB, lambda i: F[i][best]["OU"]),
          body_vs_ctl=boot2(bB, lambda i: F[i][best]["PU"] - F[i][best]["R"], cB, lambda i: F[i][best]["PU"] - F[i][best]["R"]))
# random-subspace null for the mean: how many of 10 random subspaces reach the PU mean on half B body items
pr["rand_means"] = [float(np.mean([F[i][best]["Rall"][r] for i in bB])) for r in range(len(RN))]
def bootp(sel, f, n=4000):
    pairs = sorted({i // 2 for i in sel}); vals = {p: [f(i) for i in sel if i // 2 == p] for p in pairs}; bs = []
    for _ in range(n):
        s_ = rng.choice(len(pairs), len(pairs)); bs.append(np.mean([x for j in s_ for x in vals[pairs[j]]]))
    bs = np.array(bs); return float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
pr["p_PU_R_body"] = bootp(bB, lambda i: F[i][best]["PU"] - F[i][best]["R"]); pr["p_PU_R_ctl"] = bootp(cB, lambda i: F[i][best]["PU"] - F[i][best]["R"])
pr["p_PU_OU_body"] = bootp(bB, lambda i: F[i][best]["PU"] - F[i][best]["OU"])
pr["n_body"] = len(bB); pr["n_ctl"] = len(cB)
pr["pass"] = dict(a=pr["PU"][0] >= .15, b=pr["PU_minus_R"][1] > 0, c=pr["PU_minus_OU"][1] > 0, d=pr["body_vs_ctl"][1] > 0)
out["prereg"] = pr
# per task at chosen site and at all|all (both halves)
out["tasks"] = {}
for t in sorted(set(task.values())):
    sel = [i for i in ids if task[i] == t]
    out["tasks"][t] = {s: {k: boot(sel, lambda i, s=s, k=k: F[i][s][k], 1000) for k in ["PU", "OU", "R", "full"]} for s in [best, "all|all", "swap|all", "last|all"]}
    out["tasks"][t]["n"] = len(sel)
# AtP: share of the gap attributed per layer (summed over tokens), per subspace, normalised by the gap
res = {r["id"]: r for r in A["res"]}
def layer_prof(sel, key):
    xs = []
    for i in sel:
        r = res[i]; g = r["ldt"] - r["ld"]
        if abs(g) < 1: continue
        if key == "full": xs.append(r["full"].astype(np.float64).sum(-1) / g)
        elif key == "R": xs.append(np.mean([r["sub"][SN.index(s)].astype(np.float64).sum(-1) for s in RN], 0) / g)
        else: xs.append(r["sub"][SN.index(key)].astype(np.float64).sum(-1) / g)
    return np.array(xs)
out["atp_layer"] = {fm: {k: layer_prof([i for i in ids if fam[i] == fm], k).mean(0).tolist() for k in ["PU", "OU", "R", "full"]} for fm in ["body", "ctl"]}
out["atp_layer_task"] = {t: {k: layer_prof([i for i in ids if task[i] == t], k).mean(0).tolist() for k in ["PU", "OU", "R", "full"]} for t in sorted(set(task.values()))}
json.dump(out, open(f"{D}/stats.json", "w"), indent=1)
print("n", out["n"], "chosen", best)
for k, v in pr.items(): print(k, np.round(v, 3) if isinstance(v, list) else v)
print("sites (body PU/OU/R/full | ctl PU/R):")
for s in sites: r = out["sites"][s]; print(f"  {s:10s} {r['body']['PU']:.2f} {r['body']['OU']:.2f} {r['body']['R']:.2f} {r['body']['full']:.2f} | {r['ctl']['PU']:.2f} {r['ctl']['R']:.2f} {r['ctl']['full']:.2f}")
