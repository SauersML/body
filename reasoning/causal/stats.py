"""final numbers for body-causal: bootstrap 95% CIs over items, permutation p for body-vs-control differences. Writes stats.json."""
import json, numpy as np
I = json.load(open("items.json")); rng = np.random.default_rng(0)
def load(d, f="scores.json"): return {k: np.array(v) for k, v in json.load(open(f"{d}/{f}")).items()}
BODY = ["perspective", "reach", "effector", "pain", "limb"]; CTL = ["perspective_ctl", "reach_ctl", "effector_ctl", "limb_ctl", "geo", "arith", "lr_word", "hf_word"]
def ix(ts): return np.array([k for k, i in enumerate(I) if i["task"] in ts])
def ci(x): b = rng.choice(x, (4000, len(x))).mean(1); return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def perm(x, y, n=20000):
    obs = x.mean() - y.mean(); z = np.concatenate([x, y]); c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(x)].mean() - z[len(x):].mean()) >= abs(obs)
    return float((c + 1) / (n + 1))
OUT = {}
# 1. ablation: accuracy change, body tasks vs controls
abl = {}
for d, c, lab in [("out2_8b", "ablate_body150", "body-place map (150 dims)"), ("out2_8b", "ablate_matched150", "non-body, same variance"), ("out2_8b", "ablate_iso150", "random (150 dims)"),
                  ("out3_8b", "abl_parts11_all", "body-part axes (11 dims)"), ("out3_8b", "abl_objs11_all", "object axes (11 dims)"), ("out3_8b", "abl_rand11_all", "random (11 dims)")]:
    R = load(d); mb = R["base"][:, 0] - R["base"][:, 1]; m = R[c][:, 0] - R[c][:, 1]
    da = lambda ts: ((m[ix(ts)] > 0).astype(float) - (mb[ix(ts)] > 0))
    xb, xc = da(BODY), da(CTL); abl[lab] = dict(body=ci(xb), ctl=ci(xc), p=perm(xb, xc), per_task={t: float(da([t]).mean()) for t in BODY + CTL})
OUT["ablation"] = abl
# 2. hand<->foot push at the answer (late layers): shift of logit(hand)-logit(foot) toward the pushed side
R = load("out4_8b"); B = R["base"]
def hm(S, ks): return np.array([(S[k, 0] - S[k, 1]) * (1 if I[k]["hf"] == "hand" else -1) for k in ks])
def push(cpos, cneg, ts):
    ks = ix(ts); return np.concatenate([hm(R[cpos], ks) - hm(B, ks), -(hm(R[cneg], ks) - hm(B, ks))])
def flips(cpos, cneg, ts):  # share of answers that switch to the pushed side, among those that started on the other side
    ks = ix(ts); f = []
    for c, s in [(cpos, 1), (cneg, -1)]:
        h0, h1 = hm(B, ks) * s, hm(R[c], ks) * s; f += list((h1[h0 < 0] > 0).astype(float))
    return float(np.mean(f)), len(f)
hfp = {}
for a in [3, 4]:
    for nm, cp, cn in [("hand/foot body direction", f"steer_hf_late_+{a}", f"steer_hf_late_-{a}"), ("random direction", f"steer_rand_late_+{a}", f"steer_rand_late_-{a}"), ("random direction B", f"steer_randB_late_+{a}", f"steer_randB_late_-{a}")]:
        xb, xw = push(cp, cn, ["effector", "pain"]), push(cp, cn, ["hf_word"])
        hfp[f"{nm} x{a}"] = dict(body=ci(xb), word=ci(xw), p=perm(xb, xw), flips_body=flips(cp, cn, ["effector", "pain"]), flips_word=flips(cp, cn, ["hf_word"]))
OUT["hf_push"] = hfp
# 3. left<->right push at the answer: shift of logit(left)-logit(right) toward pushed side
def lm(S, ks): return np.array([(S[k, 0] - S[k, 1]) * (1 if I[k]["lr"] == "left" else -1) for k in ks])
def lpush(cp, cn, ts):
    ks = ix(ts); return np.concatenate([lm(R[cp], ks) - lm(B, ks), -(lm(R[cn], ks) - lm(B, ks))])
lrp = {}
for nm, d in [("left/right on bodies", "lr_body"), ("left/right on objects", "lr_obj"), ("random direction", "rand")]:
    cp, cn = f"steer_{d}_late_+3", f"steer_{d}_late_-3"
    lrp[nm] = {t: ci(lpush(cp, cn, [t])) for t in ["perspective", "perspective_ctl", "lr_word"]}
OUT["lr_push"] = lrp
# 4. edits at the question text only (stem tokens, layers 2-30): change of margin toward the correct answer
R5 = load("out5_8b"); mb = R5["base"][:, 0] - R5["base"][:, 1]; st = {}
for c, lab in [("refl_hf_all", "swap hand<->foot"), ("refl_lr_body_all", "swap left<->right"), ("abl_parts_all", "remove body-part axes"), ("refl_rand_all", "swap along random axis")]:
    m = R5[c][:, 0] - R5[c][:, 1]; xb, xc = (m - mb)[ix(BODY)], (m - mb)[ix(CTL)]
    fb = (np.sign(m) != np.sign(mb))[ix(BODY)].astype(float); fc = (np.sign(m) != np.sign(mb))[ix(CTL)].astype(float)
    st[lab] = dict(body=ci(xb), ctl=ci(xc), p=perm(xb, xc), flip_body=ci(fb), flip_ctl=ci(fc))
OUT["stem"] = st
# base accuracy
R0 = load("out2_8b"); m0 = R0["base"][:, 0] - R0["base"][:, 1]
OUT["base_acc"] = {t: float((m0[ix([t])] > 0).mean()) for t in BODY + CTL}; OUT["n"] = {t: int(len(ix([t]))) for t in BODY + CTL}
json.dump(OUT, open("stats.json", "w"), indent=1)
for k in ["ablation", "hf_push", "stem"]:
    for lab, v in OUT[k].items(): print(k, lab, {kk: (np.round(vv, 2).tolist() if isinstance(vv, list) else (round(vv, 4) if isinstance(vv, float) else vv)) for kk, vv in v.items() if kk != "per_task"})
for lab, v in lrp.items(): print("lr", lab, {t: np.round(x, 1).tolist() for t, x in v.items()})
