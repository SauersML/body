"""Test 3: centroid displacement toward each reference body X (pull_X = mean over other bodies b of [mean dist(b, others) - dist(b, X)]).
Human privilege predicts pull_human > pull_X for every other X. Bootstrap over templates. usage: python hub.py TAG"""
import sys, json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]; f = f"{D}/hpriv/out/{TAG}_states.npz"
if not os.path.exists(f): f = f"{D}/hpemrg/out/{TAG}_states.npz"
Z = np.load(f, allow_pickle=True); meta = Z["meta"]; LS = [int(x) for x in Z["layers"]]
BOD = ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk"]
bo = np.array([m[0] for m in meta]); tj = np.array([int(m[2]) for m in meta]); rng = np.random.default_rng(0); NB = 1000
R = {}
for l in LS:
    X = Z[f"L{l}"].astype(np.float32)
    M = np.stack([[X[(bo == b) & (tj == t)].mean(0) for t in range(8)] for b in BOD])  # [8 bodies, 8 templates, d]
    def pulls(T):
        c = M[:, T].mean(1); c = c - c.mean(0); c /= np.linalg.norm(c, axis=1, keepdims=True); Dm = 1 - c @ c.T; n = len(BOD); p = np.zeros(n)
        for x in range(n):
            v = []
            for b in range(n):
                if b == x: continue
                o = [k for k in range(n) if k not in (b, x)]; v.append(Dm[b, o].mean() - Dm[b, x])
            p[x] = np.mean(v)
        return p
    full = pulls(np.arange(8)); bs = np.array([pulls(rng.choice(8, 8)) for _ in range(NB)])
    oth = bs[:, 1:].max(1)
    R[str(l)] = {"pull": {b: [float(full[i]), float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))] for i, b in enumerate(BOD)},
                 "human-minus-max": [float(full[0] - full[1:].max()), float(np.percentile(bs[:, 0] - oth, 2.5)), float(np.percentile(bs[:, 0] - oth, 97.5))],
                 "rank_human": int((full > full[0]).sum() + 1)}
    print(l, {b: round(full[i], 3) for i, b in enumerate(BOD)}, "rank", R[str(l)]["rank_human"], flush=True)
json.dump(R, open(f"{D}/res_hub_{TAG}.json", "w"), indent=0)
