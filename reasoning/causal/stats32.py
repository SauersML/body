"""Qwen3-32B body-causal numbers: bootstrap 95% CIs over items, permutation p for body-vs-control differences. Reads out32/scores.json, writes stats32.json."""
import json, re, numpy as np
I = json.load(open("items32.json")); R = {k: np.array(v) for k, v in json.load(open("out32/scores.json")).items()}; rng = np.random.default_rng(0)
BODY = ["perspective", "reach", "effector", "pain", "limb", "novel", "reach2"]; CTL = ["perspective_ctl", "reach_ctl", "effector_ctl", "limb_ctl", "geo", "arith", "lr_word", "hf_word", "novel_ctl", "reach2_ctl"]
PAIRS = [("perspective", "perspective_ctl"), ("reach", "reach_ctl"), ("effector", "effector_ctl"), ("limb", "limb_ctl"), ("novel", "novel_ctl"), ("reach2", "reach2_ctl")]
def ix(ts): return np.array([k for k, i in enumerate(I) if i["task"] in ts])
def ci(x): x = np.asarray(x, float); b = rng.choice(x, (4000, len(x))).mean(1); return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def perm(x, y, n=20000):
    obs = x.mean() - y.mean(); z = np.concatenate([x, y]).astype(float); c = 0
    for _ in range(n): rng.shuffle(z); c += abs(z[:len(x)].mean() - z[len(x):].mean()) >= abs(obs)
    return float((c + 1) / (n + 1))
def mg(c): return R[c][:, 0] - R[c][:, 1]
MB = mg("base"); ACC0 = MB > 0
OUT = {"n": {t: int(len(ix([t]))) for t in BODY + CTL}, "base_acc": {t: ci(ACC0[ix([t])]) for t in BODY + CTL}}
# perspective: mirror (answer is the opposite of the side named) vs same-side items
def named(q):
    m = re.search(r"(their|its own) (left|right)", q, re.I); return m.group(2) if m else None
for t in ["perspective", "perspective_ctl"]:
    ks = ix([t]); mir = [k for k in ks if named(I[k]["q"]) and named(I[k]["q"]) != I[k]["a"]]; same = [k for k in ks if named(I[k]["q"]) == I[k]["a"]]
    OUT[f"{t}_mirror_vs_same"] = dict(mirror=ci(ACC0[mir]), same=ci(ACC0[same]), n=(len(mir), len(same)))
def dacc(c, ts): return ((mg(c) > 0).astype(float) - ACC0)[ix(ts)]
def dm(c, ts): return (mg(c) - MB)[ix(ts)] * np.where(ACC0[ix(ts)], 1, 1)  # margin change toward correct
def flip(c, ts): return (np.sign(mg(c)) != np.sign(MB))[ix(ts)].astype(float)
# 1+2: subspace removal, all body tasks vs all control tasks, and per pair
abl = {}
for c in [k for k in R if "_abl_" in k]:
    xb, xc = dacc(c, BODY), dacc(c, CTL)
    abl[c] = dict(body=ci(xb), ctl=ci(xc), p=perm(xb, xc), pairs={f"{a}|{b}": dict(body=ci(dacc(c, [a])), ctl=ci(dacc(c, [b])), p=perm(dacc(c, [a]), dacc(c, [b]), 5000)) for a, b in PAIRS})
OUT["ablation"] = abl
# stem swaps: flip rate and margin change, perspective vs perspective_ctl and all body vs all control
st = {}
for c in [k for k in R if k.startswith("stem_refl")]:
    st[c] = dict(persp=ci(flip(c, ["perspective"])), persp_ctl=ci(flip(c, ["perspective_ctl"])), p_persp=perm(flip(c, ["perspective"]), flip(c, ["perspective_ctl"])),
                 persp_dacc=ci(dacc(c, ["perspective"])), persp_ctl_dacc=ci(dacc(c, ["perspective_ctl"])),
                 body=ci(flip(c, BODY)), ctl=ci(flip(c, CTL)), p=perm(flip(c, BODY), flip(c, CTL)),
                 hf=ci(flip(c, ["effector", "pain"])), hf_word=ci(flip(c, ["hf_word"])))
OUT["stem"] = st
# answer pushes: shift of logit(hand)-logit(foot) (or left-right) toward the pushed side
def side(S, ks, key, pos): return np.array([(S[k, 0] - S[k, 1]) * (1 if I[k][key] == pos else -1) for k in ks])
def push(n, a, ts, key, pos):
    ks = ix(ts); B = R["base"]; return np.concatenate([side(R[f"ans_{n}_+{a}"], ks, key, pos) - side(B, ks, key, pos), -(side(R[f"ans_{n}_-{a}"], ks, key, pos) - side(B, ks, key, pos))])
def flips_to(n, a, ts, key, pos):
    ks = ix(ts); f = []
    for c, s in [(f"ans_{n}_+{a}", 1), (f"ans_{n}_-{a}", -1)]:
        h0, h1 = side(R["base"], ks, key, pos) * s, side(R[c], ks, key, pos) * s; f += list((h1[h0 < 0] > 0).astype(float))
    return ci(f) + [len(f)]
hp = {}
for a in [3, 4]:
    for n in ["hf", "rand", "randB"]:
        xb, xw = push(n, a, ["effector", "pain"], "hf", "hand"), push(n, a, ["hf_word"], "hf", "hand")
        hp[f"{n} x{a}"] = dict(body=ci(xb), word=ci(xw), p=perm(xb, xw), flips_body=flips_to(n, a, ["effector", "pain"], "hf", "hand"), flips_word=flips_to(n, a, ["hf_word"], "hf", "hand"))
OUT["hf_push"] = hp
lp = {}
for n in ["lr_body", "lr_obj", "rand_lr"]:
    v = {t: push(n, 3, [t], "lr", "left") for t in ["perspective", "perspective_ctl", "lr_word"]}
    lp[n] = {t: ci(x) for t, x in v.items()}; lp[n]["p_body_vs_car"] = perm(v["perspective"], v["perspective_ctl"])
OUT["lr_push"] = lp
json.dump(OUT, open("stats32.json", "w"), indent=1)
r = lambda v: np.round(v, 2).tolist()
print("base acc", {t: round(v[0], 2) for t, v in OUT["base_acc"].items()})
for t in ["perspective", "perspective_ctl"]: print(t, {k: r(v) if isinstance(v, list) else v for k, v in OUT[f"{t}_mirror_vs_same"].items()})
for c, v in abl.items(): print(c, "body", r(v["body"]), "ctl", r(v["ctl"]), "p", round(v["p"], 3), {k: (round(x["body"][0], 2), round(x["ctl"][0], 2), round(x["p"], 3)) for k, x in v["pairs"].items()})
for c, v in st.items(): print(c, {k: r(x) if isinstance(x, list) else round(x, 3) for k, x in v.items()})
for c, v in hp.items(): print(c, {k: r(x) if isinstance(x, list) else round(x, 4) for k, x in v.items()})
for c, v in lp.items(): print(c, {k: r(x) if isinstance(x, list) else round(x, 4) for k, x in v.items()})
# run 32b: bigger doses (6, 8, 12x) at layers 39-52 ("late") and 21-52 ("wide")
RB = {k: np.array(v) for k, v in json.load(open("out32/b/scores.json")).items()}
def pushb(n, bn, a, ts):
    ks = ix(ts); B = RB["base"]; return np.concatenate([side(RB[f"ans_{n}_{bn}_+{a}"], ks, "hf", "hand") - side(B, ks, "hf", "hand"), -(side(RB[f"ans_{n}_{bn}_-{a}"], ks, "hf", "hand") - side(B, ks, "hf", "hand"))])
def flipsb(n, bn, a, ts):
    ks = ix(ts); f = []
    for c, s in [(f"ans_{n}_{bn}_+{a}", 1), (f"ans_{n}_{bn}_-{a}", -1)]:
        h0, h1 = side(RB["base"], ks, "hf", "hand") * s, side(RB[c], ks, "hf", "hand") * s; f += list((h1[h0 < 0] > 0).astype(float))
    return ci(f) + [len(f)]
hb = {}
for bn in ["late", "wide"]:
    for a in [6, 8, 12]:
        for n in ["hf", "rand"]:
            xb, xw = pushb(n, bn, a, ["effector", "pain"]), pushb(n, bn, a, ["hf_word"])
            hb[f"{n} {bn} x{a}"] = dict(body=ci(xb), word=ci(xw), p=perm(xb, xw), flips_body=flipsb(n, bn, a, ["effector", "pain"]), flips_word=flipsb(n, bn, a, ["hf_word"]))
OUT["hf_push_b"] = hb; json.dump(OUT, open("stats32.json", "w"), indent=1)
for c, v in hb.items(): print(c, {k: r(x) if isinstance(x, list) else round(x, 4) for k, x in v.items()})
