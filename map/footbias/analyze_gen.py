import json, numpy as np
D = "$BODY_WORK/body/footbias/"
K = ["feet", "hands", "back", "shoulders", "neck", "knees", "belly", "face", "lips", "eyes", "hair", "chest", "head", "arms", "legs", "skin"]
rng = np.random.default_rng(0); out = {}
def tab(R):  # opening x part counts, per condition
    T = {}
    for cond, rows in R.items():
        M = np.zeros((50, len(K)))
        for r in rows: M[r["op"]] += [r["c"][k] for k in K]
        T[cond] = M
    return T
def lift(T, idx):
    r = T["romantic"][idx].sum(0); c = T["control"][idx].sum(0)
    sr = r / r.sum(); sc = c / c.sum()
    return sr, sc, np.log2((sr + 1e-4) / (sc + 1e-4))
Ts = {}
for tag in ["chat", "base"]:
    R = json.load(open(D + f"counts_{tag}.json")); T = Ts[tag] = tab(R)
    sr, sc, l = lift(T, np.arange(50))
    bs = np.array([lift(T, rng.integers(0, 50, 50))[2] for _ in range(2000)])
    n0 = {c: float(np.mean([sum(r["c"].values()) == 0 for r in rows])) for c, rows in R.items()}
    out[tag] = dict(n=len(R["romantic"]), total={c: int(T[c].sum()) for c in T}, frac_samples_no_part=n0,
        share_rom=dict(zip(K, sr.round(4).tolist())), share_ctl=dict(zip(K, sc.round(4).tolist())),
        log2lift=dict(zip(K, l.round(2).tolist())), ci=dict(zip(K, np.percentile(bs, [2.5, 97.5], 0).T.round(2).tolist())),
        feet_rank_lift=int((l > l[0]).sum()) + 1, feet_count={c: int(T[c][:, 0].sum()) for c in T})
    print(tag, json.dumps(out[tag]["total"]), "feet", out[tag]["feet_count"], "rank", out[tag]["feet_rank_lift"], "nopart", n0)
    for k in K: print(f"   {k:10s} rom {sr[K.index(k)]:.3f} ctl {sc[K.index(k)]:.3f} log2lift {l[K.index(k)]:+.2f} {out[tag]['ci'][k]}")
# chat minus base lift difference for feet, and feet's difference vs mean of other parts' differences
bs = []
for _ in range(2000):
    i = rng.integers(0, 50, 50)
    lc = lift(Ts["chat"], i)[2]; lb = lift(Ts["base"], i)[2]; bs.append(lc - lb)
bs = np.array(bs); dlt = lift(Ts["chat"], np.arange(50))[2] - lift(Ts["base"], np.arange(50))[2]
out["chat_minus_base_log2lift"] = dict(zip(K, dlt.round(2).tolist())); out["chat_minus_base_ci"] = dict(zip(K, np.percentile(bs, [2.5, 97.5], 0).T.round(2).tolist()))
print("chat-base log2 lift", out["chat_minus_base_log2lift"]["feet"], out["chat_minus_base_ci"]["feet"], "rank", int((dlt > dlt[0]).sum()) + 1)
json.dump(out, open("$BODY_WORK/foot/summary_gen.json", "w"), indent=1)
