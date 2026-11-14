import json, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "out2_8b"
I = json.load(open("items.json")); R = {k: np.array(v) for k, v in json.load(open(f"{D}/scores.json")).items()}
rng = np.random.default_rng(0)
def T(t, cond=lambda i: True): return np.array([k for k, i in enumerate(I) if i["task"] == t and cond(i)])
def ci(x, n=4000):
    b = rng.choice(x, (n, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])
B = R["base"]; mb = B[:, 0] - B[:, 1]
BODY = ["perspective", "reach", "effector", "pain", "limb"]; CTL = ["perspective_ctl", "reach_ctl", "effector_ctl", "limb_ctl", "geo", "arith", "lr_word", "hf_word"]
print("base acc", {t: round(float((mb[T(t)] > 0).mean()), 2) for t in BODY + CTL})
print("\n== ablation: accuracy change (95% CI) ==")
for c in [k for k in R if k.startswith("ablate")]:
    m = R[c][:, 0] - R[c][:, 1]; out = []
    for t in BODY + CTL:
        ix = T(t); d = (m[ix] > 0).astype(float) - (mb[ix] > 0); lo, hi = ci(d); out.append(f"{t[:8]}:{d.mean():+.2f}")
    bix = np.concatenate([T(t) for t in BODY]); cix = np.concatenate([T(t) for t in CTL])
    db = ((m[bix] > 0).astype(float) - (mb[bix] > 0)); dc = ((m[cix] > 0).astype(float) - (mb[cix] > 0))
    print(c.ljust(18), f"BODY {db.mean():+.2f} CTL {dc.mean():+.2f} |", " ".join(out))
def sh(c, key, val, t, cond=lambda i: True):
    ix = T(t, lambda i: key in i and cond(i)); sg = np.array([1 if I[k][key] == val else -1 for k in ix])
    d = ((R[c][ix, 0] - R[c][ix, 1]) - mb[ix]) * sg   # change of logit(val) - logit(other)
    nb = (R[c][ix, 0] - R[c][ix, 1]) * sg > 0          # now answers val
    was = mb[ix] * sg > 0
    return d.mean(), nb[~was].mean() if (~was).any() else np.nan
print("\n== steering: change in logit(HAND)-logit(FOOT) nats / share of foot-answers turned hand ==")
for c in [k for k in R if k.startswith("steer_hf") or k.startswith("steer_rand")]:
    print(c.ljust(22), " ".join(f"{t[:8]}:{sh(c, 'hf', 'hand', t)[0]:+6.1f}/{sh(c, 'hf', 'hand', t)[1]:.2f}" for t in ["effector", "pain", "hf_word"]),
          " wordfree eff:", f"{sh(c, 'hf', 'hand', 'effector', lambda i: i['wf'])[0]:+.1f}")
print("\n== steering: change in logit(LEFT)-logit(RIGHT) / share of right-answers turned left ==")
for c in [k for k in R if k.startswith("steer_lr") or k.startswith("steer_rand")]:
    print(c.ljust(26), " ".join(f"{t[:11]}:{sh(c, 'lr', 'left', t)[0]:+6.1f}/{sh(c, 'lr', 'left', t)[1]:.2f}" for t in ["perspective", "perspective_ctl", "lr_word"]))
