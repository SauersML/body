exec(open("anal.py").read().split("r = lambda")[0])
from scipy.stats import rankdata
r = lambda D, B: spearmanr(D[it], B[it])[0]
LAY = 25
ORDER = ["thumb","index","middle","ring","little","palm","backhand","wrist","forearm","elbow","upperarm","shoulder",
         "neckB","neckF","crown","forehead","ear","cheek","lips","tongue","chest","upback","belly","lowback",
         "hip","thigh","knee","shin","ankle","heel","sole","toes"]
o = [tk.index(t) for t in ORDER]
SK = np.load("skin_cm.npy"); C = cos_d(typemeans(LAY))
W = np.load(f"{SO}/wordemb.npy").astype(np.float64); Wt = np.stack([W[Tidx[t]].mean(0) for t in tk]); CW = cos_d(Wt)
def pct(D):
    D = D[np.ix_(o, o)]; iu = np.triu_indices(len(o), 1); R = np.zeros_like(D); q = rankdata(D[iu]) / len(iu[0]); R[iu] = q; R = R + R.T; np.fill_diagonal(R, np.nan); return R
def mds(D, k=2):
    D = D ** 2; n = len(D); J = np.eye(n) - 1 / n; B = -.5 * J @ D @ J; w, v = np.linalg.eigh(B); i = np.argsort(w)[::-1]
    return v[:, i[:k]] * np.sqrt(np.maximum(w[i[:k]], 0)), w[i] / w[w > 0].sum()
dep = {t: d for t, _, d in S.ARM} | {t: d for t, _, d in S.LEG}
maps = {}
for nm, sel in [("arm", ORDER[:12]), ("leg", ORDER[24:])]:
    ii = [tk.index(t) for t in sel]; Y, ev = mds(C[np.ix_(ii, ii)])
    rr = spearmanr(Y[:, 0], [dep[t] for t in sel])[0]
    if rr < 0: Y[:, 0] *= -1
    maps[nm] = {t: Y[k].tolist() for k, t in enumerate(sel)}; maps[nm + "_rho"] = abs(rr); maps[nm + "_ev"] = ev[:2].tolist()
    print(nm, "dim1 vs depth rho", round(abs(rr), 3), "var expl", ev[:2].round(2))
out = dict(order=ORDER, layer=LAY, body=pct(SK).tolist(), qwen=pct(C).tolist(), word=pct(CW).tolist(),
           rho_qwen=r(C, SK), rho_word=r(CW, SK), maps=maps,
           thumb_shoulder_cm=float(SK[tk.index("thumb"), tk.index("shoulder")]))
json.dump(out, open("figdata.json", "w"), default=lambda x: None if x != x else x)
print({k: v for k, v in out.items() if k.startswith(("rho", "thumb"))})
