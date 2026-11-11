import sys, os, json, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from scipy.stats import spearmanr, rankdata
H = np.load("H.npy").astype(np.float64); rows = json.load(open("rows.json")); LAYS = [15, 20, 25, 30]
tk = json.load(open(os.path.expandvars("$BODY_REPO/figures/bodychain4/tk.json"))); SK = np.load(os.path.expandvars("$BODY_REPO/figures/bodychain4/skin_cm.npy"))
Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tk}; it = np.triu_indices(len(tk), 1)
def cos_d(M): M = M - M.mean(0); n = M / np.linalg.norm(M, axis=1, keepdims=True); return 1 - n @ n.T
def rdm(cond, li):
    sel = {(r["li"]): [] for r in rows}
    for k, r in enumerate(rows):
        if r["cond"] == cond: sel[r["li"]].append(k)
    return cos_d(np.stack([H[sum((sel[i] for i in Tidx[t]), []), li].mean(0) for t in tk]))[it]
def pr(x, y, z):  # partial Spearman of x,y given z
    x, y, z = map(rankdata, (x, y, z)); a = np.corrcoef([x, y, z]); return (a[0,1]-a[0,2]*a[1,2]) / np.sqrt((1-a[0,2]**2)*(1-a[1,2]**2))
def mantel(D, B, n=2000, rng=np.random.default_rng(0)):
    full = np.zeros((32, 32)); full[it] = D; full = full + full.T; obs = spearmanr(D, B)[0]; c = 0
    for _ in range(n):
        p = rng.permutation(32); c += spearmanr(full[np.ix_(p, p)][it], B)[0] >= obs
    return (c + 1) / (n + 1)
B = SK[it]
for li, L in enumerate(LAYS):
    R = {c: rdm(c, li) for c in ["touch", "textbook", "list", "third"]}
    line = f"L{L} " + " ".join(f"{c} {spearmanr(R[c], B)[0]:.3f}" for c in R)
    line += " | partial touch|" + " ".join(f"{c} {pr(R['touch'], B, R[c]):.3f}" for c in ["textbook", "list", "third"])
    line += " | touch~base " + " ".join(f"{c} {spearmanr(R['touch'], R[c])[0]:.2f}" for c in ["textbook", "list", "third"])
    if L == 25: line += " | mantel p " + " ".join(f"{c} {mantel(R[c], B, 500):.3f}" for c in R)
    print(line)
