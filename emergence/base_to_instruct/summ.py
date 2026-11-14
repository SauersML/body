# Print key measures per checkpoint: mean over relative depth 0.5-0.8 (layers), steering at depth .65 a=.5
import json, glob, sys, os, numpy as np
fs = sys.argv[1:] or sorted(glob.glob("$BODY_RUNS*/out/*.json"))
K = ["rsa_touch", "rsa_third", "rsa_list", "arm_touch", "arm_list", "sharp", "own_d", "own_locus", "lr_d", "wing_grid", "glorp_grid", "wing_arm", "phantom_hand", "phantom_rank"]
print("tag fmt " + " ".join(k[:9] for k in K) + " st1spec st1acc st3spec st3acc")
for f in fs:
    r = json.load(open(f)); nl = r["nl"]; L = [l for l in range(nl + 1) if .5 <= l / nl <= .8]
    for fmt, R in r["fmt"].items():
        v = [np.nanmean([R[k][l] for l in L]) for k in K]; st = R["steer"]
        g = lambda fr, m: st.get(f"{fr}_d0.65_a0.5", {}).get(m, np.nan)
        print(f"{r['tag']:16s} {fmt:10s} " + " ".join(f"{x:9.2f}" for x in v) + f" {g('first','spec'):7.2f} {g('first','acc'):6.2f} {g('third','spec'):7.2f} {g('third','acc'):6.2f}")
