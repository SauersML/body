"""v2 analyses on H_rs (reply start): word-free alignment, layer-19-22 dip, posture, movement. Usage: analyze2.py OUTDIR TAG STEERLAYER"""
import sys, os, json, numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S, stimuli2 as S2
OUT, TAG, SL = sys.argv[1], sys.argv[2], int(sys.argv[3])
FIG = os.path.join(os.path.dirname(__file__), "fig")
plt.rcParams.update({"font.size": 18, "axes.titlesize": 21, "axes.labelsize": 19, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 3.5})
H = np.load(f"{OUT}/H_rs.npy", mmap_mode="r"); rows = json.load(open(f"{OUT}/rows.json")); NL = H.shape[1]
keys = [l["key"] for l in S.LOCS]; nloc = len(keys); T = 8; WK = list(S2.WORDFREE)
def sel(**kw): return [k for k, r in enumerate(rows) if all(r.get(a) == b for a, b in kw.items())]
idx = np.zeros((nloc, T), int)
for k in sel(cond="self"): idx[rows[k]["li"], rows[k]["tpl"]] = k
out = {}
def lay(L, ix): return np.asarray(H[ix, L], np.float32)

# ---------------- A. word-free directions vs named directions
widx = np.zeros((12, 6, T), int)
for k in sel(cond="wordfree"): r = rows[k]; widx[WK.index(r["loc"]), r["desc"], r["tpl"]] = k
nidx = [[keys.index(x) for x in S2.NAMED[w]] for w in WK]
def named_means(L):
    a = lay(L, idx.ravel()).reshape(nloc, T, -1); a -= a.mean(0, keepdims=True); m = a.mean(1)
    N = np.stack([m[ix].mean(0) for ix in nidx]); return N - N.mean(0)
def wf_means(L, descs=slice(None)):
    a = lay(L, widx[:, descs].ravel()).reshape(12, -1, T, H.shape[-1]).mean(1); a -= a.mean(0, keepdims=True); F = a.mean(1); return F - F.mean(0)
def cosm(A, B): A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True); return A @ B.T
acc, gap, half = [], [], []
for L in range(NL):
    C = cosm(wf_means(L), named_means(L)); acc.append(float(np.mean(C.argmax(1) == np.arange(12))))
    gap.append(float(np.diag(C).mean() - C[~np.eye(12, dtype=bool)].mean()))
    C2 = cosm(wf_means(L, slice(0, 3)), wf_means(L, slice(3, 6))); half.append(float(np.mean(C2.argmax(1) == np.arange(12))))
# lexical baseline: description input embeddings vs named-phrase input embeddings
DE = np.load(f"{OUT}/descemb.npy").mean(1); WE = np.load(f"{OUT}/wordemb.npy"); NE = np.stack([WE[ix].mean(0) for ix in nidx])
Cl = cosm(DE - DE.mean(0), NE - NE.mean(0)); lexacc = float(np.mean(Cl.argmax(1) == np.arange(12)))
out["wordfree"] = dict(acc=acc, gap=gap, half=half, lexacc=lexacc)
print("wordfree->named match acc by layer:", [round(x, 2) for x in acc[::2]], "lexical baseline", lexacc)
# RSA: word-free 12-loc geometry vs body path
G = S.model_rdms()["body path distance"]; Gw = np.array([[G[np.ix_(a, b)].mean() for b in nidx] for a in nidx]); it = np.triu_indices(12, 1)
def cvr(A, B): dA = A[:, None] - A[None]; dB = B[:, None] - B[None]; return np.einsum("ijd,ijd->ij", dA, dB)
DL = ((DE[:, None] - DE[None]) ** 2).sum(-1)
def pspear(x, y, z):
    x, y, z = rankdata(x), rankdata(y), rankdata(z); res = lambda u: u - np.polyval(np.polyfit(z, u, 1), z); return np.corrcoef(res(x), res(y))[0, 1]
wrsa = [float(pspear(cvr(wf_means(L, slice(0, 3)), wf_means(L, slice(3, 6)))[it], Gw[it], DL[it])) for L in range(NL)]
out["wordfree_rsa_partial"] = wrsa; print("wordfree RSA vs body path (desc-embedding partialled):", [round(x, 2) for x in wrsa[::2]])
C = cosm(wf_means(SL), named_means(SL))
fig, axs = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
ax.plot(acc, color="k", label="described without naming → matched to named part"); ax.plot(half, color="tab:gray", ls="--", label="ceiling: half the descriptions → other half")
ax.axhline(1 / 12, color="tab:red", lw=2, ls=":", label="chance (1 in 12)"); ax.axhline(lexacc, color="tab:blue", lw=2, ls=":", label="same test on the input words only")
ax.set_xlabel("layer"); ax.set_ylabel("fraction of the 12 places\nmatched correctly"); ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=14, loc="lower right")
ax.set_title("Does 'where your watch sits' land\nwhere 'your wrist' does?")
ax = axs[1]; im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(12)); ax.set_xticklabels(WK, rotation=45, ha="right"); ax.set_yticks(range(12)); ax.set_yticklabels([f"'{S2.WORDFREE[w][0]}'" for w in WK], fontsize=12)
ax.set_xlabel("place named outright ('your wrist')"); ax.set_ylabel("place described without its name")
ax.set_title(f"layer {SL}: similarity of directions"); plt.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle(f"{TAG}: body places described without naming them use the same directions", fontsize=24); fig.tight_layout()
fig.savefig(f"{FIG}/{TAG}_wordfree.png", dpi=100); plt.close()

# ---------------- B. layer 19-22 dip: body-path correlation (30 parts, word-emb partialled) under different distance measures
tkeys = list(dict.fromkeys(l["type"] for l in S.LOCS)); Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tkeys}
rep = [Tidx[t][0] for t in tkeys]; Gt = G[np.ix_(rep, rep)]; i30 = np.triu_indices(len(tkeys), 1)
Wt = np.stack([WE[Tidx[t]].mean(0) for t in tkeys]); Wt -= Wt.mean(0); Dw = ((Wt[:, None] - Wt[None]) ** 2).sum(-1)
allself = idx.ravel()
def parts(a): return np.stack([a[Tidx[t]].mean(0) for t in tkeys])  # a: nloc x T x d
def score(a):
    a = a - a.mean(0, keepdims=True); A, B = parts(a[:, 0::2].mean(1)[:, None])[:, 0], parts(a[:, 1::2].mean(1)[:, None])[:, 0]
    return float(pspear(cvr(A, B)[i30], Gt[i30], Dw[i30]))
var = {k: [] for k in ["raw (v1)", "top 1% dims removed", "each dim standardised", "unit length per sentence", "rank-transformed dims"]}
massive = []
for L in range(NL):
    a = lay(L, allself).reshape(nloc, T, -1)
    var["raw (v1)"].append(score(a))
    mabs = np.abs(a).reshape(-1, a.shape[-1]).mean(0); top = np.argsort(mabs)[::-1]
    massive.append(float(np.sort(np.abs(a).reshape(-1, a.shape[-1]).max(0))[::-1][:4].mean() / np.median(np.abs(a))))
    keep = np.ones(a.shape[-1], bool); keep[top[: a.shape[-1] // 100]] = False
    var["top 1% dims removed"].append(score(a[..., keep]))
    flat = a.reshape(-1, a.shape[-1]); sd = flat.std(0) + 1e-6
    var["each dim standardised"].append(score(a / sd))
    c = a - flat.mean(0); var["unit length per sentence"].append(score(c / np.linalg.norm(c, axis=-1, keepdims=True)))
    var["rank-transformed dims"].append(score(np.apply_along_axis(rankdata, 0, flat).reshape(a.shape)))
# share of the location signal carried by top dims
share = []
for L in range(NL):
    a = lay(L, allself).reshape(nloc, T, -1); m = a.mean(1); m -= m.mean(0); v = (m ** 2).sum(0); share.append(float(np.sort(v)[::-1][: a.shape[-1] // 100].sum() / v.sum()))
out["dip"] = dict(var, massive=massive, top1pct_share=share)
for k, v in var.items(): print(f"{k:28s}", [round(x, 2) for x in v[12:30]])
print("top-1% dims share of location variance, layers 12-29:", [round(x, 2) for x in share[12:30]])
fig, axs = plt.subplots(1, 2, figsize=(22, 8))
for (k, v), col in zip(var.items(), ["k", "tab:red", "tab:blue", "tab:green", "tab:orange"]): axs[0].plot(v, color=col, label=k, lw=4 if k.startswith("raw") else 3)
axs[0].set_xlabel("layer"); axs[0].set_ylabel("match to distance along the body\n(rank corr., word meaning partialled)"); axs[0].legend(fontsize=14); axs[0].axhline(0, color="k", lw=.5)
axs[0].set_title("Body-map strength under different distance measures")
axs[1].plot(share, color="k"); axs[1].set_xlabel("layer"); axs[1].set_ylabel("share of location differences\ncarried by the top 1% of dimensions")
axs[1].set_title("Is the location signal concentrated in a few dimensions?"); axs[1].set_ylim(0, 1)
fig.suptitle(f"{TAG}: what is the dip in the middle layers?", fontsize=24); fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_dip.png", dpi=100); plt.close()

# ---------------- C. posture: does the touched hand's representation move toward where the hand is?
def pmean(L, **kw): return lay(L, sel(cond="posture", **kw)).mean(0)
side_word = {"left": "L", "right": "R"}
def axis(L, place, s):
    a = lay(L, allself).reshape(nloc, T, -1).mean(1)
    tgt = {"head": ["crown", "forehead"], "belly": ["belly"], "knee": ["knee" + s], "side": ["hip" + s, "thigh" + s]}[place]
    hand = a[[keys.index("backhand" + s)]].mean(0); t = a[[keys.index(x) for x in tgt]].mean(0); v = t - hand; return v / np.linalg.norm(v)
P = list(S2.PLACES); post = np.zeros((NL, 4, 4))
for L in range(NL):
    for i, pl in enumerate(P):
        for s in ["left", "right"]:
            o = "right" if s == "left" else "left"
            diff = pmean(L, loc=pl, posed=s, touched=s) - pmean(L, loc=pl, posed=o, touched=s)  # same words except which hand was placed
            for j, ax_pl in enumerate(P): post[L, i, j] += diff @ axis(L, ax_pl, side_word[s]) / 2
out["posture"] = post.tolist()
pn = post / np.linalg.norm(post, axis=2, keepdims=True)
print(f"posture L{SL} (rows: where the touched hand was put; cols: projection on hand->place axis):\n", np.round(post[SL], 2))
diag = [float(np.mean([(post[L] - post[L].mean(0))[i, i] for i in range(4)]) / (np.abs(post[L]).mean() + 1e-9)) for L in range(NL)]
out["posture_diag"] = diag
fig, axs = plt.subplots(1, 2, figsize=(21, 8))
pc = post[SL] - post[SL].mean(0, keepdims=True)
im = axs[0].imshow(pc, cmap="RdBu_r", vmin=-np.abs(pc).max(), vmax=np.abs(pc).max())
lab = ["on head", "on belly", "on knee", "at side"]
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["→ head", "→ belly", "→ knee", "→ hip"]); axs[0].set_yticks(range(4)); axs[0].set_yticklabels(["hand " + x for x in lab])
for i in range(4):
    for j in range(4): axs[0].text(j, i, f"{pc[i, j]:+.1f}", ha="center", va="center", fontsize=15)
axs[0].set_title(f"layer {SL}: shift of a touched hand when it\nis the placed hand (each column centred)"); plt.colorbar(im, ax=axs[0], fraction=0.04)
axs[1].plot(diag, color="k"); axs[1].axhline(0, color="k", lw=.5); axs[1].set_xlabel("layer"); axs[1].set_ylabel("shift toward its own place,\nbeyond other postures (relative)")
axs[1].set_title("By layer")
fig.suptitle(f"{TAG}: 'rest your left hand on your head' — does the left hand move on the map?", fontsize=23); fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_posture.png", dpi=100); plt.close()

# ---------------- D. movement sentences: which touched location does each movement look like?
mv = sel(cond="move"); mkeys = list(dict.fromkeys(rows[k]["loc"] for k in mv))
tgtm = {"arm": ["upperarm", "elbow", "forearm", "shoulder"], "fingers": ["thumb", "index", "middle", "ring", "little"], "knee": ["knee", "thigh", "shin"], "foot": ["toes", "sole", "heel", "ankle"], "head": ["neckB", "neckF", "crown", "forehead"]}
mres = {}
for L in range(NL):
    a = lay(L, allself).reshape(nloc, T, -1).mean(1); a -= a.mean(0)
    M = np.stack([lay(L, [k for k in mv if rows[k]["loc"] == m]).mean(0) for m in mkeys]); M -= M.mean(0)
    C = cosm(M, a)
    # per movement: rank-percentile of its own body region among the 54 locations, and left/right correctness
    pct, lr = [], []
    for i, m in enumerate(mkeys):
        part, s = m[:-1], m[-1]; tg = [keys.index(x + (s if x + s in keys else "")) for x in tgtm[part]]
        r = rankdata(C[i]) / nloc; pct.append(float(r[tg].mean()))
        if part != "head": lr.append(float(C[i][[keys.index(x + s) for x in tgtm[part]]].mean() > C[i][[keys.index(x + ("R" if s == "L" else "L")) for x in tgtm[part]]].mean()))
    mres[L] = (float(np.mean(pct)), float(np.mean(lr)))
out["move"] = mres
print("movement -> touch map: mean percentile of moved region, L/R correct:", {L: tuple(round(x, 2) for x in mres[L]) for L in range(0, NL, 3)})
json.dump(out, open(f"{OUT}/v2_summary.json", "w"))
