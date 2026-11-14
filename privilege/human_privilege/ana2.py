"""Paired bootstrap contrasts (over the 8 touch templates) for tests 1-3. usage: python ana2.py TAG LAYERS(comma)"""
import sys, json, os, numpy as np
from scipy.stats import spearmanr
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D + "/hpriv"); from bodies import xyz
TAG = sys.argv[1]; f = f"{D}/hpriv/out/{TAG}_states.npz"
if not os.path.exists(f): f = f"{D}/hpemrg/out/{TAG}_states.npz"
Z = np.load(f, allow_pickle=True); meta = Z["meta"]
LS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [int(x) for x in Z["layers"]]
BOD = ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk"]; NH = [b for b in BOD if b != "human"]
bo = np.array([m[0] for m in meta]); pi = np.array([int(m[1]) for m in meta]); tj = np.array([int(m[2]) for m in meta])
iu = np.triu_indices(16, 1); K = 4; NB = int(os.environ.get("NB", 300)); rng = np.random.default_rng(0)
G = {b: (lambda c: np.linalg.norm(c[:, None] - c[None], axis=-1)[iu])(xyz(b) / np.ptp(xyz(b), 0).max()) for b in BOD}
def stats(C, T):
    """C: body -> [16,8,d]; T: template indices (resample). returns dict of scalars"""
    h = len(T) // 2; A, Bt = T[:h], T[h:]; out = {}
    PM = {b: C[b][:, T].mean(1) for b in C}
    for b in BOD:
        P = PM[b] - PM[b].mean(0); P = P / np.linalg.norm(P, axis=1, keepdims=True); out[f"rsa_{b}"] = spearmanr(1 - (P @ P.T)[iu], G[b])[0]
    U = {}
    for s in BOD + ["ctrl"]:
        S = C[s][:, A].mean(1); S -= S.mean(0); U[s] = np.linalg.svd(S, full_matrices=False)[2][:K]
    Tg = {t: (lambda X: X - X.mean(0))(C[t][:, Bt].mean(1)) for t in BOD}
    ve = lambda s, t: np.sum((Tg[t] @ U[s].T) ** 2) / np.sum(Tg[t] ** 2)
    for s in BOD + ["ctrl"]:  # targets: the 6 bodies that are neither source nor human-layout twin
        tg = [t for t in BOD if t not in (s, "human", "velk")]
        out[f"ve_{s}"] = np.mean([ve(s, t) for t in tg])
    for s in [x for x in BOD if x not in ("human", "velk")] + ["ctrl"]:  # paired, common targets: non-human bodies other than s
        tg = [t for t in BOD if t not in (s, "human", "velk")]
        out[f"vp_{s}"] = np.mean([ve("human", t) - ve(s, t) for t in tg])
    cen = {b: C[b][:, T].mean((0, 1)) for b in BOD}; g = np.mean(list(cen.values()), 0)
    cn = {b: (cen[b] - g) / np.linalg.norm(cen[b] - g) for b in BOD}
    for b in NH:  # human pull: distance to other non-human bodies minus distance to human (>0 = closer to human)
        out[f"pull_{b}"] = np.mean([1 - cn[b] @ cn[o] for o in NH if o != b]) - (1 - cn[b] @ cn["human"])
    for b in BOD: out[f"meand_{b}"] = np.mean([1 - cn[b] @ cn[o] for o in BOD if o != b])
    return out
R = {}
for l in LS:
    X = Z[f"L{l}"].astype(np.float32); C = {}
    for b in BOD + ["ctrl"]:
        m = bo == b; c = np.zeros((16, 8, X.shape[1]), np.float32); c[pi[m], tj[m]] = X[m]; C[b] = c
    full = stats(C, np.arange(8))
    bs = [stats(C, rng.choice(8, 8, replace=True)) for _ in range(NB)]
    keys = list(full); arr = {k: np.array([s[k] for s in bs]) for k in keys}
    def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    r = {k: [float(full[k])] + ci(arr[k]) for k in keys}
    for b in NH: r[f"rsaD_human-{b}"] = [float(full["rsa_human"] - full[f"rsa_{b}"])] + ci(arr["rsa_human"] - arr[f"rsa_{b}"])
    oth = [s for s in BOD if s != "human"]
    vo = [s for s in oth if s != "velk"]
    r["veD_human-minpaired"] = [float(min(full[f"vp_{s}"] for s in vo))] + ci(np.min([arr[f"vp_{s}"] for s in vo], 0))
    r["veD_human-bestother"] = r["veD_human-minpaired"]
    r["pull_mean"] = [float(np.mean([full[f"pull_{b}"] for b in NH]))] + ci(np.mean([arr[f"pull_{b}"] for b in NH], 0))
    R[str(l)] = r; print(l, {k: round(r[k][0], 3) for k in ["rsa_human", "rsa_dog", "ve_human", "ve_dog", "ve_ctrl", "veD_human-bestother", "pull_mean"]}, flush=True)
json.dump(R, open(f"{D}/res2_{TAG}{os.environ.get('SUF','')}.json", "w"), indent=0)
