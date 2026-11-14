import json, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "out8b"
I = json.load(open("items.json")); R = {k: np.array(v) for k, v in json.load(open(f"{D}/scores.json")).items()}
tasks = ["perspective", "reach", "effector", "pain", "limb", "perspective_ctl", "reach_ctl", "effector_ctl", "limb_ctl", "geo", "arith", "lr_word", "hf_word"]
TI = {t: np.array([k for k, i in enumerate(I) if i["task"] == t]) for t in tasks}
rng = np.random.default_rng(0)
def m(S): return S[:, 0] - S[:, 1]
def boot(x, n=2000):
    b = rng.choice(x, (n, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])
mode = sys.argv[2] if len(sys.argv) > 2 else "acc"
B = m(R["base"])
print("cond".ljust(24), " ".join(t[:9].rjust(9) for t in tasks))
print("base acc".ljust(24), " ".join(f"{(B[TI[t]] > 0).mean():9.2f}" for t in tasks))
for c, S in R.items():
    if c == "base": continue
    M = m(S)
    if mode == "acc": row = [(M[TI[t]] > 0).mean() - (B[TI[t]] > 0).mean() for t in tasks]
    elif mode == "margin": row = [np.mean(M[TI[t]] - B[TI[t]]) / np.mean(np.abs(B[TI[t]])) for t in tasks]  # change in margin, in units of mean |base margin|
    elif mode == "flip": row = [np.mean(np.sign(M[TI[t]]) != np.sign(B[TI[t]])) for t in tasks]
    print(c.ljust(24), " ".join(f"{v:+9.2f}" for v in row))
