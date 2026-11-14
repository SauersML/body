import json, numpy as np, sys
from collections import Counter
D = "$BODY_WORK/foot/"
PARTS = ["feet", "hands", "back", "shoulders", "neck", "knees", "belly", "face"]
ALL = PARTS + ["chest", "head", "legs", "arms", "stomach", "toes"]
rng = np.random.default_rng(0)
C = json.load(open(D + "res_chat.json")); B = json.load(open(D + "res_base.json")); G = json.load(open(D + "res_corpus.json"))
S = {}
def spec(M, nb=4000):
    """M: wordings x 8 parts. feet minus mean of controls per wording, in units of the pooled sd across parts; bootstrap over wordings."""
    M = np.asarray(M, float); sd = M.std(1, ddof=1).mean()
    e = (M[:, 0] - M[:, 1:].mean(1)) / sd
    bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(nb)]
    pm = M.mean(0); allz = [((M[:, j] - np.delete(M, j, 1).mean(1)) / sd).mean() for j in range(8)]
    return dict(d=float(e.mean()), ci=np.percentile(bs, [2.5, 97.5]).round(2).tolist(), rank=int((pm > pm[0]).sum()) + 1,
                means=dict(zip(PARTS, pm.round(3).tolist())), all_d=dict(zip(PARTS, np.round(allz, 2).tolist())), n=len(e))
def pr(k, v): S[k] = v; print(k, {a: b for a, b in v.items() if a != "means"} if isinstance(v, dict) else v)

# L1a base rates
if "L1a" in C:
    rows = []
    for name, r in C["L1a"].items():
        c = Counter(r["cls"]); n = len(r["cls"])
        rows.append((name, r["calm"], c["feet"] / n, np.mean([c[p] / n for p in PARTS[1:]]), dict(c.most_common(4))))
        print(f"  {name:32s} calm={r['calm']} feet={c['feet']/n:.2f} top={dict(c.most_common(4))}")
    f = np.array([r[2] for r in rows]); o = np.array([r[3] for r in rows]); calm = np.array([r[1] for r in rows])
    e = f - o; bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(4000)]
    pr("L1a_feet_minus_ctrl_share", dict(d=float(e.mean()), ci=np.percentile(bs, [2.5, 97.5]).round(3).tolist(),
        feet_calm=float(f[calm == 1].mean()), feet_nocalm=float(f[calm == 0].mean()), table=[(r[0], round(r[2], 2), r[4]) for r in rows]))
# L1b basin
if "L1b" in C:
    out = {}
    for sn, r in C["L1b"].items():
        b0 = ALL[int(np.argmax(r["none"]))]; p0 = float(np.exp(r["none"][0]) / np.exp(r["none"]).sum())
        row = {"unsteered_top": b0, "unsteered_pfeet": round(p0, 3)}
        for k, v in r.items():
            if k == "none": continue
            v = np.array(v); top = Counter(ALL[i] for i in v.argmax(1))
            pn = np.exp(v) / np.exp(v).sum(1, keepdims=True)
            row[k] = dict(frac_feet_top=round(top["feet"] / len(v), 2), mean_pfeet=round(float(pn[:, 0].mean()), 3), tops=dict(top.most_common(3)))
        out[sn] = row; print("  L1b", sn, row)
    S["L1b"] = out
# L1c norms
for tag, R in [("chat", C), ("base", B)]:
    dn = np.array([R["L1c_dirnorm"][p] for p in PARTS]); sp = np.array([R["L1c_spread"][p] for p in PARTS])
    ls = slice(15, 31)
    S[f"L1c_{tag}"] = dict(dirnorm_rank=int((dn[:, ls].mean(1) > dn[0, ls].mean()).sum()) + 1,
        dirnorm_ratio=float(dn[0, ls].mean() / dn[1:, ls].mean()), spread_ratio=float(sp[0, ls].mean() / sp[1:, ls].mean()),
        dirnorm_L15_30=dict(zip(PARTS, dn[:, ls].mean(1).round(1).tolist())))
    print("L1c", tag, S[f"L1c_{tag}"])
# L2
if "L2" in C:
    lift = {p: [] for p in PARTS}
    for it in C["L2"]:
        for p, s in zip(it["ps"], it["p"]): lift[p].append(4 * s)
    # per item: feet lift - mean control lift present in the item (items with feet)
    e = []
    for it in C["L2"]:
        if "feet" in it["ps"]:
            l = dict(zip(it["ps"], it["p"])); e.append(4 * l["feet"] - np.mean([4 * l[q] for q in it["ps"] if q != "feet"]))
    e = np.array(e); bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(4000)]
    pr("L2_interest", dict(feet_minus_others_lift=float(e.mean()), ci=np.percentile(bs, [2.5, 97.5]).round(2).tolist(), n=len(e),
        lift=dict((p, round(float(np.mean(v)), 2)) for p, v in lift.items()), mass=float(np.mean([it["mass"] for it in C["L2"]]))))
# L3 / L4
for tag, R in [("chat", C), ("base", B)]:
    pr(f"L3_raw_pleasant_{tag}", spec(R["L3_raw_pleasant"]))
    pr(f"L4_raw_sensual_{tag}", spec(R["L4_raw_sensual"]))
    pr(f"L4_raw_romantic_{tag}", spec(R["L4_raw_romantic"]))
    for L in [15, 20, 25]:
        for k in ["proj_valence", "proj_sensual", "proj_sensual_orth"]:
            M = np.array(R[k])[:, :, L]; s = spec(M)
            null = np.array(R["proj_null"])[:, L]
            s["null_pct"] = float((null < (M[:, 0] - M[:, 1:].mean(1)).mean() / M.std((0, 1))).mean())
            pr(f"{k}_L{L}_{tag}", s)
    print("dir acc", tag, "val", np.round(np.array(R["dir_acc"]["valence"])[[15, 20, 25]], 2), "sens", np.round(np.array(R["dir_acc"]["sensual"])[[15, 20, 25]], 2))
if "L3_chat_PNU" in C:
    M = np.array(C["L3_chat_PNU"]); pr("L3_chat_pleasant", spec(M[:, :, 0] - M[:, :, 2]))
    W = np.array(C["L3_chat_want"]); P = np.exp(W) / np.exp(W).sum(-1, keepdims=True); pr("L3_chat_want", spec((P * np.arange(1, 6)).sum(-1)))
    M = np.array(C["L4_chat_RP"]); pr("L4_chat_romantic", spec(M[:, :, 0] - M[:, :, 1]))
    print("tokcheck", C["chat_tokcheck"])
S["corpus"] = G
json.dump(S, open(D + "summary.json", "w"), indent=1, default=str)
