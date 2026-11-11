exec(open("anal.py").read().split("r = lambda")[0])
r = lambda D, B: spearmanr(D[it], B[it])[0]
SK = np.load("skin_cm.npy"); ST = np.load("straight_cm.npy")
BODY = {"steps": Gsteps, "skin_cm": SK, "straight_cm_mesh": ST, "path_cm_graph": Gpath}
print("body-metric agreement:", {k: round(r(B, Gsteps), 2) for k, B in BODY.items()})
W = np.load(f"{SO}/wordemb.npy").astype(np.float64); Wt = np.stack([W[Tidx[t]].mean(0) for t in tk])
print("word cos_c:", {k: round(r(cos_d(Wt), B), 3) for k, B in BODY.items()}, " word euc:", {k: round(r(euc(Wt), B), 3) for k, B in BODY.items()})
res = {}
for L in [15, 18, 20, 22, 25, 28, 30, 33, 36]:
    M = typemeans(L); C = cos_d(M); E = euc(M); CV = cveuc(L)
    res[L] = {k: (round(r(C, B), 3), round(r(E, B), 3), round(r(CV, B), 3)) for k, B in BODY.items()}
    print(L, res[L])
# permutation (Mantel) p at L25, skin cm
rng = np.random.default_rng(0); C25 = cos_d(typemeans(25)); n = len(tk)
def mantel(Dm, B, k=5000):
    obs = r(Dm, B); c = 0
    for _ in range(k):
        p = rng.permutation(n); c += spearmanr(Dm[np.ix_(p, p)][it], B[it])[0] >= obs
    return obs, (c + 1) / (k + 1)
print("mantel L25 skin", mantel(C25, SK), "word skin", mantel(cos_d(Wt), SK))
# bootstrap over wordings: resample 8 templates
a = np.asarray(H[idx, 25], np.float64); bs = []
for _ in range(500):
    tpl = rng.integers(0, 8, 8); M = np.stack([a[Tidx[t]][:, tpl].reshape(-1, a.shape[-1]).mean(0) for t in tk]); bs.append(r(cos_d(M), SK))
print("bootstrap over wordings L25 skin: 95% CI", np.percentile(bs, [2.5, 97.5]).round(3))
