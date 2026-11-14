"""States analyses (tests 1-3) for human privilege. usage: python ana.py TAG  (reads hpriv/out/TAG_states.npz or hpemrg/out)."""
import sys, json, os, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__) + "/hpriv"); from bodies import B, xyz, names
TAG = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__))
f = f"{D}/hpriv/out/{TAG}_states.npz"
if not os.path.exists(f): f = f"{D}/hpemrg/out/{TAG}_states.npz"
Z = np.load(f, allow_pickle=True); LS = list(Z["layers"]); meta = Z["meta"]
BOD = ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk"]  # named bodies; human_desc/velk described
ALL = BOD + ["human_desc", "ctrl"]
bo = np.array([m[0] for m in meta]); pi = np.array([int(m[1]) for m in meta]); tj = np.array([int(m[2]) for m in meta])
rng = np.random.default_rng(0); NB = 1000

def cube(l):  # body -> [16 parts, 8 templates, d]
    X = Z[f"L{l}"].astype(np.float32); out = {}
    for b in ALL:
        m = bo == b; C = np.zeros((16, 8, X.shape[1]), np.float32); C[pi[m], tj[m]] = X[m]; out[b] = C
    return out

def geo(b): c = xyz(b); c = c / np.ptp(c, 0).max(); return np.linalg.norm(c[:, None] - c[None], axis=-1)
iu = np.triu_indices(16, 1)
def rsa(C, b, T):  # part means over templates T, cosine distance after centring, Spearman vs 3-D distance
    P = C[:, T].mean(1); P = P - P.mean(0); P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return spearmanr(1 - (P @ P.T)[iu], geo(b)[iu])[0]
def decode(C):  # leave-one-template-out nearest centroid (cosine, centred), chance 1/16
    ok = []
    for t in range(8):
        tr = [s for s in range(8) if s != t]; M = C[:, tr].mean(1); mu = C[:, tr].mean((0, 1))
        M = M - mu; M /= np.linalg.norm(M, axis=1, keepdims=True); Q = C[:, t] - mu; Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        ok.append((Q @ M.T).argmax(1) == np.arange(16))
    return np.mean(ok)
def lr_pairs(b):
    nm = names(b); pr = []
    for i, n in enumerate(nm):
        if "left" in n:
            r = n.replace("left", "right")
            if r in nm: pr.append((i, nm.index(r)))
    return pr
def lr(C, b):  # left/right axis generalising to an unseen pair: diff-of-means on other pairs, test held-out pair; per template
    pr = lr_pairs(b)
    if len(pr) < 2: return np.nan
    ok = []
    for k, (li, ri) in enumerate(pr):
        oth = [p for j, p in enumerate(pr) if j != k]
        w = np.mean([C[r].mean(0) - C[l].mean(0) for l, r in oth], 0)
        mid = np.mean([(C[r].mean(0) + C[l].mean(0)) / 2 for l, r in oth], 0)
        # sign test relative to the held-out pair's own midpoint removes body-part offset: is right-left along w positive?
        ok += list(((C[ri] - C[li]) @ w) > 0)
    return np.mean(ok)

def boot(fn, n=NB):
    v = [fn(rng.choice(8, 8, replace=True)) for _ in range(n)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

R = {"layers": [int(l) for l in LS], "rsa": {}, "dec": {}, "lr": {}, "ve": {}, "hub": {}}
K = 4
for l in LS:
    C = cube(l); L = str(int(l))
    R["rsa"][L] = {b: [float(rsa(C[b], b, list(range(8))))] + boot(lambda T: rsa(C[b], b, T), 300) for b in BOD + ["human_desc"]}
    R["dec"][L] = {b: float(decode(C[b])) for b in ALL}
    R["lr"][L] = {b: float(lr(C[b], b)) for b in BOD + ["human_desc"]}
    # (2) reuse: variance of target's centred part means (templates 4-7) explained by source's top-K PCs (templates 0-3); split halves both ways
    def ve(src, tgt, A, Bt):
        S = C[src][:, A].mean(1); S -= S.mean(0); U = np.linalg.svd(S, full_matrices=False)[2][:K]
        T = C[tgt][:, Bt].mean(1); T -= T.mean(0); return np.sum((T @ U.T) ** 2) / np.sum(T ** 2)
    srcs = BOD + ["ctrl"]; VE = {}
    for s in srcs:
        for t in BOD:
            VE[(s, t)] = (ve(s, t, [0, 1, 2, 3], [4, 5, 6, 7]) + ve(s, t, [4, 5, 6, 7], [0, 1, 2, 3])) / 2
    d = C["human"].shape[2]; Ur = np.linalg.qr(rng.standard_normal((d, K)))[0]
    rnd = float(np.mean([(lambda T: np.sum((T @ Ur) ** 2) / np.sum(T ** 2))(C[t].mean(1) - C[t].mean(1).mean(0)) for t in BOD]))
    # mean VE each source gives to the other named bodies (excluding itself); also velk excluded as a target for human (same layout) -> report both
    out = {}
    for s in srcs:
        tg = [t for t in BOD if t != s and t != "velk" and not (s == "velk" and t == "human")]
        out[s] = float(np.mean([VE[(s, t)] for t in tg]))
    out["rand"] = rnd; out["own"] = {t: float(VE[(t, t)]) for t in BOD}
    out["matrix"] = {f"{s}>{t}": float(VE[(s, t)]) for s in srcs for t in BOD}
    # bootstrap over targets for the human-minus-best-other contrast
    R["ve"][L] = out
    # (3) hub / centroid displacement: centroid of each body (all its sentences), global mean of the 8 bodies removed
    cen = {b: C[b].mean((0, 1)) for b in BOD}; g = np.mean([cen[b] for b in BOD], 0)
    cn = {b: (cen[b] - g) / np.linalg.norm(cen[b] - g) for b in BOD}
    dist = {f"{a}~{b}": float(1 - cn[a] @ cn[b]) for a in BOD for b in BOD if a < b}
    nn = {}
    for a in BOD:
        o = [b for b in BOD if b != a]; nn[a] = min(o, key=lambda b: 1 - cn[a] @ cn[b])
    meand = {a: float(np.mean([1 - cn[a] @ cn[b] for b in BOD if b != a])) for a in BOD}
    # part-level: each non-human part state (template-avg) -> which body's part-centroid set is nearest (excluding own)
    R["hub"][L] = {"nn": nn, "meandist": meand, "dist": dist}
    print(L, "rsa", {b: round(R['rsa'][L][b][0], 2) for b in BOD}, "dec", {b: round(R['dec'][L][b], 2) for b in ["human", "dog", "robot"]}, flush=True)
json.dump(R, open(f"{D}/res_{TAG}.json", "w"), indent=0)
