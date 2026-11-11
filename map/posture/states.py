import numpy as np, sys, json
S = np.load(sys.argv[1]); P = sorted({k.split("__")[0] for k in S.files})
out = {}
for p in P:
    t, si, st, li = [S[f"{p}__{k}"].astype(np.float32) for k in ["touch", "sit", "stand", "lie"]]
    n = len(t); sc = []; scl = []; scs = []
    for i in range(n):  # leave this touch out of the axis
        m = np.arange(n) != i
        a, b = si[m].mean(0), st[m].mean(0); w = (a - b) / np.linalg.norm(a - b)
        f = lambda h: ((h - b) @ w) / ((a - b) @ w)
        sc.append(f(t[i])); scl.append(f(li[i])); scs.append((f(si[i]), f(st[i])))
    out[p] = dict(touch=float(np.mean(sc)), touch_sd=float(np.std(sc)), lie=float(np.mean(scl)), heldout_sit=float(np.mean([x[0] for x in scs])), heldout_stand=float(np.mean([x[1] for x in scs])))
    print(f"{p:35s} plain touch {np.mean(sc):+.2f} (sd {np.std(sc):.2f})  lie {np.mean(scl):+.2f}  held-out sit {out[p]['heldout_sit']:+.2f} stand {out[p]['heldout_stand']:+.2f}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
