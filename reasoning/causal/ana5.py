"""analysis of stem interventions: per task accuracy change, margin change, flip rate, with bootstrap CIs and exact sign-test p-values."""
import json, sys, numpy as np
from scipy.stats import binomtest
D = sys.argv[1] if len(sys.argv) > 1 else "out5_8b"; PAT = sys.argv[2] if len(sys.argv) > 2 else ""
I = json.load(open("items.json")); R = {k: np.array(v) for k, v in json.load(open(f"{D}/scores.json")).items()}
rng = np.random.default_rng(0)
TASKS = ["perspective", "reach", "effector", "pain", "limb", "perspective_ctl", "reach_ctl", "effector_ctl", "limb_ctl", "geo", "arith", "lr_word", "hf_word"]
BODY = TASKS[:5]
def ix(t): return np.array([k for k, i in enumerate(I) if i["task"] == t])
mb = R["base"][:, 0] - R["base"][:, 1]
def stats(c, idx):
    m = R[c][:, 0] - R[c][:, 1]
    acc0, acc1 = (mb[idx] > 0), (m[idx] > 0)
    flip = (np.sign(m[idx]) != np.sign(mb[idx]))
    up, dn = int((~acc0 & acc1).sum()), int((acc0 & ~acc1).sum())
    p = binomtest(dn, up + dn).pvalue if up + dn else 1.0
    d = acc1.astype(float) - acc0; b = rng.choice(d, (2000, len(d))).mean(1)
    return dict(dacc=d.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), p=p, flip=flip.mean(), dmar=(m[idx] - mb[idx]).mean(), acc0=acc0.mean())
if __name__ == "__main__":
    print("base acc", {t: round(float((mb[ix(t)] > 0).mean()), 2) for t in TASKS})
    print("cells: d(margin, nats toward correct) / flip-rate")
    for c in R:
        if c == "base" or PAT not in c: continue
        row = [f"{t[:7]}:{stats(c, ix(t))['dmar']:+5.1f}/{stats(c, ix(t))['flip']:.2f}" for t in TASKS]
        print(c.ljust(22), " ".join(row))
