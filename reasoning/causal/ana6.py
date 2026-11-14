import json, sys, numpy as np
from scipy.stats import binomtest
D = sys.argv[1]; I = json.load(open("items.json")); R = {k: np.array(v) for k, v in json.load(open(f"{D}/scores6.json")).items()}
EFF = [k for k, i in enumerate(I) if i["task"] == "effector"]; B = EFF[1::2]
T = {"effB(held-out)": B, "pain": [k for k, i in enumerate(I) if i["task"] == "pain"], "hf_word": [k for k, i in enumerate(I) if i["task"] == "hf_word"]}
CT = [k for k, i in enumerate(I) if i["task"] in ("geo", "arith", "effector_ctl", "reach_ctl")]
mb = R["base"][:, 0] - R["base"][:, 1]
def hand_margin(S, ks): return np.array([(S[k, 0] - S[k, 1]) * (1 if I[k]["hf"] == "hand" else -1) for k in ks])  # logit(hand)-logit(foot)
def row(c):
    out = []
    for t, ks in T.items():
        h0 = hand_margin(R["base"], ks); h1 = hand_margin(R[c], ks)
        tgt = "hand" if c.endswith("tohand") else "foot"; src = [i for i, k in enumerate(ks) if (h0[i] > 0) != (tgt == "hand")]  # items that started on the other side
        fl = np.mean([(h1[i] > 0) == (tgt == "hand") for i in src]) if src else np.nan
        out.append(f"{t[:7]}:{(h1 - h0).mean():+5.1f}/{fl:.2f}")
    m = R[c][CT, 0] - R[c][CT, 1]; out.append(f"ctl acc {((m > 0).mean() - (mb[CT] > 0).mean()):+.2f}")
    return " ".join(out)
for c in R:
    if c != "base" and (len(sys.argv) < 3 or sys.argv[2] in c): print(c.ljust(28), row(c))
