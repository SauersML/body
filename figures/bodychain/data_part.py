import os
import sys, os, json, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse, FancyBboxPatch
plt.rcParams.update({"font.family": ["Avenir Next", "Arial Unicode MS"], "text.color": "#111111"})
SO = os.path.expandvars("$BODY_REPO/map/somato/out2_8b")
# ---------------- data ----------------
H = np.load(f"{SO}/H_rs.npy", mmap_mode="r"); rows = json.load(open(f"{SO}/rows.json"))
nloc = len(S.LOCS); idx = np.zeros((nloc, 8), int)
for k, r in enumerate(rows):
    if r["cond"] == "self": idx[r["li"], r["tpl"]] = k
tk = list(dict.fromkeys(l["type"] for l in S.LOCS)); Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tk}
G = S.model_rdms()["body path distance"]; rep = [Tidx[t][0] for t in tk]; Gt = G[np.ix_(rep, rep)]; it = np.triu_indices(len(tk), 1)
W = np.load(f"{SO}/wordemb.npy").astype(np.float32); Wt = np.stack([W[Tidx[t]].mean(0) for t in tk]); Wt -= Wt.mean(0)
Dw = ((Wt[:, None] - Wt[None]) ** 2).sum(-1)
def rdm(L):
    a = np.asarray(H[idx, L], np.float32); a = a - a.mean(0, keepdims=True)
    at = np.stack([a[Tidx[t]].mean(0) for t in tk]); A, B = at[:, 0::2].mean(1), at[:, 1::2].mean(1)
    return np.einsum("ijd,ijd->ij", A[:, None] - A[None], B[:, None] - B[None])
def resid(u, z): u, z = rankdata(u), rankdata(z); return u - np.polyval(np.polyfit(z, u, 1), z)
def ps(x, y, z): return spearmanr(resid(x, z), resid(y, z))[0]
NL = H.shape[1]; Ds = [rdm(L) for L in range(NL)]
bylayer = [spearmanr(Gt[it], resid(Ds[L][it], Dw[it]))[0] if L else np.nan for L in range(NL)]
rho_word = spearmanr(Dw[it], Gt[it])[0]
D = Ds[25]; x = Gt[it]; y = resid(D[it], Dw[it]); y = (y - y.min()) / (y.max() - y.min())
rho_main = spearmanr(x, y)[0]; rho_partial_full = ps(D[it], Gt[it], Dw[it])
def mds(D, k=2):
    D = np.maximum((D + D.T) / 2, 0); n = len(D); J = np.eye(n) - 1 / n; B = -.5 * J @ D @ J; w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    return v[:, o[:k]] * np.sqrt(np.maximum(w[o[:k]], 0))
ARM = tk[:12]; LEG = tk[12:20]
dep = {t: d for t, _, d in S.ARM} | {t: d for t, _, d in S.LEG}
maps = {}
for nm, sel in [("arm", ARM), ("leg", LEG)]:
    ii = [tk.index(t) for t in sel]; Y = mds(D[np.ix_(ii, ii)])
    if spearmanr(Y[:, 0], [dep[t] for t in sel])[0] > 0: Y[:, 0] *= -1   # proximal = +dim1
    maps[nm] = dict(zip(sel, Y)); maps[nm + "_rho"] = abs(spearmanr(Y[:, 0], [dep[t] for t in sel])[0])
nums = dict(rho_semipartial_L25=rho_main, rho_partial_L25=rho_partial_full, rho_raw_L25=spearmanr(D[it], Gt[it])[0], rho_word=rho_word, peak_layer=int(np.nanargmax(bylayer)),
            arm_order_rho=maps["arm_rho"], leg_order_rho=maps["leg_rho"], n_pairs=len(x), bylayer=[None if np.isnan(b) else round(b, 3) for b in bylayer])
json.dump(nums, open(os.path.expandvars("$BODY_REPO/figures/bodychain/numbers.json"), "w"), indent=1); print({k: v for k, v in nums.items() if k != "bylayer"})

