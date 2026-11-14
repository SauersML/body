"""stats.py DIR -> DIR/stats.json + printed table. Reads unify_{arm}.json."""
import sys, json, os, numpy as np
D = sys.argv[1]; ARMS = [a for a in ("ctrl", "words", "pair", "align", "both") if os.path.exists(f"{D}/unify_{a}.json")]
TRN = ["head", "chest", "right shin", "left thigh"]; HLD = ["belly", "left forearm", "right thigh", "left shin"]
PARTS = ['head', 'chest', 'belly', 'waist', 'hips', 'left upper arm', 'left forearm', 'left hand', 'right upper arm', 'right forearm', 'right hand', 'left thigh', 'left shin', 'left foot', 'right thigh', 'right shin', 'right foot']
NONHAND = [p for p in range(17) if not PARTS[p].endswith("hand")]
rng = np.random.default_rng(0)
def wil(x):
    x = np.asarray(x, float); n = len(x); p = x.mean(); z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]
def boot(x, n=10000):
    x = np.asarray(x, float); m = x[rng.integers(len(x), size=(n, len(x)))].mean(1); return [round(x.mean(), 2), round(np.percentile(m, 2.5), 2), round(np.percentile(m, 97.5), 2)]
S = {}; J = {a: json.load(open(f"{D}/unify_{a}.json")) for a in ARMS}
def b1_gain(a, tg, cond):  # random - cond distance, cm, per (target, pose) pair
    R = J[a]["b1"]; return np.concatenate([100 * (np.array(R[f"{t}|random"]["dist"]) - np.array(R[f"{t}|{cond}"]["dist"])) for t in tg])
def b2(a, tg, cond, al):
    R = J[a]["b2"]; base = np.array(R["base"]); hits, dl = [], []
    for t in tg:
        j = NONHAND.index(PARTS.index(t)); L = np.array(R[f"{t}|{cond}|{al}"])
        ls = L - np.logaddexp.reduce(L, 1, keepdims=True); lb = base - np.logaddexp.reduce(base, 1, keepdims=True)
        hits += list(L.argmax(1) == j); dl += list(ls[:, j] - lb[:, j])
    return hits, dl
for a in ARMS:
    s = {}; r = J[a].get("rsa")
    if r: s["rsa"] = [round(r["rsa"], 3), r["rsa_p"]]; s.update({k: wil(r[k]) for k in ("dec_w2s_trained", "dec_w2s_held", "dec_s2w_trained", "dec_s2w_held")})
    if "b1" in J[a] and all(f"{t}|random" in J[a]["b1"] for t in TRN + HLD):
        for nm, tg in (("trained", TRN), ("held", HLD)):
            for c in ("word", "sensor"):
                s[f"b1_{c}_{nm}_gain_cm"] = boot(b1_gain(a, tg, c))
                s[f"b1_{c}_{nm}_dist_cm"] = round(float(np.mean([100 * np.mean(J[a]["b1"][f"{t}|{c}"]["dist"]) for t in tg])), 1)
                s[f"b1_{c}_{nm}_touch"] = wil(np.concatenate([J[a]["b1"][f"{t}|{c}"]["touch"] for t in tg]))
            s[f"b1_random_{nm}_dist_cm"] = round(float(np.mean([100 * np.mean(J[a]["b1"][f"{t}|random"]["dist"]) for t in tg])), 1)
            s[f"b1_none_{nm}_dist_cm"] = round(float(np.mean([100 * np.mean(J[a]["b1"]["none"]["dist"][t]) for t in tg])), 1)
    if "b2" in J[a]:
        for nm, tg in (("trained", TRN), ("held", HLD)):
            base = np.array(J[a]["b2"]["base"]); s[f"b2_base_{nm}_hit"] = wil([base[i].argmax() == NONHAND.index(PARTS.index(t)) for t in tg for i in range(len(base))])
            for c in ("sensor", "word", "random"):
                for al in ("1.5", "3.0"):
                    if f"{tg[-1]}|{c}|{al}" in J[a]["b2"]:
                        h, dl = b2(a, tg, c, al); s[f"b2_{c}{al}_{nm}_hit"] = wil(h); s[f"b2_{c}{al}_{nm}_dlogp"] = boot(dl)
    if "control" in J[a]: s.update({f"ctl_{k}": wil(v) for k, v in J[a]["control"].items()})
    S[a] = s
# arm vs ctrl on b1 word gain (paired by target x pose)
if "ctrl" in S:
    for a in ARMS:
        if a == "ctrl" or "b1_word_held_gain_cm" not in S[a] or "b1_word_held_gain_cm" not in S["ctrl"]: continue
        for nm, tg in (("trained", TRN), ("held", HLD)):
            S[a][f"b1_word_{nm}_vs_ctrl"] = boot(b1_gain(a, tg, "word") - b1_gain("ctrl", tg, "word"))
json.dump(S, open(f"{D}/stats.json", "w"), indent=1)
keys = sorted({k for s in S.values() for k in s})
for k in keys: print(k.ljust(30), " | ".join(f"{a}:{S[a].get(k)}" for a in ARMS))
