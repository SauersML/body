"""RSA, axes, unsupervised maps. Usage: analyze.py OUTDIR TAG"""
import sys, os, json, numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S
OUT, TAG = sys.argv[1], sys.argv[2]
FIG = os.path.join(os.path.dirname(__file__), "fig"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
H = np.load(f"{OUT}/H.npy", mmap_mode="r"); rows = json.load(open(f"{OUT}/rows.json"))
NL = H.shape[1]; nloc = len(S.LOCS); T = len(S.TEMPLATES)
conds = ["self", "other", "object"]; POSN = ["stimulus end", "reply start"]
idx = {c: np.zeros((nloc, T), int) for c in conds}
for k, r in enumerate(rows):
    if r["cond"] in idx: idx[r["cond"]][r["li"], r["tpl"]] = k
R = S.model_rdms(); iu = np.triu_indices(nloc, 1)
COL = {"armL": "tab:blue", "armR": "tab:cyan", "legL": "tab:red", "legR": "tab:orange", "head": "tab:green", "torso": "tab:purple"}

def X(c, layer, p):  # nloc x T x d, float32, template mean removed (wording offset)
    a = np.asarray(H[idx[c], layer, p], np.float32)
    return a - a.mean(0, keepdims=True)

def cv_rdm(a):  # cross-validated squared distance between location means (even vs odd wordings)
    A, B = a[:, 0::2].mean(1), a[:, 1::2].mean(1)
    dA = A[:, None] - A[None]; dB = B[:, None] - B[None]
    return np.einsum("ijd,ijd->ij", dA, dB)

def partial_spearman(x, y, z):
    x, y, z = rankdata(x), rankdata(y), rankdata(z)
    res = lambda u: u - np.polyval(np.polyfit(z, u, 1), z)
    return np.corrcoef(res(x), res(y))[0, 1]

res = {}
for c in conds:
    for p in range(2):
        for L in range(NL):
            D = cv_rdm(X(c, L, p))[iu]
            r = {k: spearmanr(D, v[iu])[0] for k, v in R.items()}
            r["body path | lexical"] = partial_spearman(D, R["body path distance"][iu], R["shared words (lexical)"][iu])
            # within-limb only (removes left/right and limb-identity effects): path distance inside each limb
            m = (R["same limb"][iu] == 0)
            r["path within limb"] = spearmanr(D[m], R["body path distance"][iu][m])[0]
            res[(c, p, L)] = r
json.dump({f"{c}|{p}|{L}": v for (c, p, L), v in res.items()}, open(f"{OUT}/rsa.json", "w"), indent=0)

# ---- fig 1: RSA by layer
keys = ["body path distance", "path within limb", "body path | lexical", "left vs right", "same part, either side"]
fig, axs = plt.subplots(2, len(keys), figsize=(5.2 * len(keys), 10), sharex=True)
for p in range(2):
    for j, k in enumerate(keys):
        ax = axs[p, j]
        for c, col in zip(conds, ["k", "tab:red", "tab:gray"]):
            ax.plot([res[(c, p, L)][k] for L in range(NL)], color=col, lw=3, label={"self": "your body", "other": "a woman's body", "object": "a doll's body"}[c])
        ax.axhline(0, color="k", lw=.5); ax.set_title(k if p == 0 else "")
        if j == 0: ax.set_ylabel(f"{POSN[p]}\nrank correlation with\nmodel's distances")
        if p == 1: ax.set_xlabel("layer")
axs[0, 0].legend(fontsize=14)
fig.suptitle(f"{TAG}: does the model's geometry of body locations match the body?", fontsize=22)
fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_rsa_by_layer.png", dpi=110); plt.close()

# best layer = max self body-path-within-limb at reply start, from middle layers on
score = lambda L, p=1: res[("self", p, L)]["path within limb"]
best = int(max(range(2, NL), key=score)); print("best layer (reply start, path within limb):", best, score(best))

# ---- axes: left-right and distal-proximal at best layer
pairs = [(i, j) for i, a in enumerate(S.LOCS) for j, b in enumerate(S.LOCS) if a["side"] == 1 and b["side"] == -1 and a["type"] == b["type"]]
axinfo = {}
for c in conds:
    for p in range(2):
        a = X(c, best, p).mean(1)
        diffs = np.array([a[i] - a[j] for i, j in pairs])
        # leave-one-pair-out consistency: cosine of each pair's L-R difference with mean of the others
        cs = [np.dot(diffs[k], diffs[np.arange(len(diffs)) != k].mean(0)) / np.linalg.norm(diffs[k]) / np.linalg.norm(diffs[np.arange(len(diffs)) != k].mean(0)) for k in range(len(diffs))]
        axinfo[(c, p)] = float(np.mean(cs))
print("L-R difference consistency (mean LOO cosine):", {f"{c}|{POSN[p]}": round(v, 3) for (c, p), v in axinfo.items()})

# cross-condition: do location geometries agree? (rank corr between condition RDMs)
cross = {}
for p in range(2):
    for L in range(NL):
        Ds = {c: cv_rdm(X(c, L, p))[iu] for c in conds}
        cross[(p, L)] = {"self~other": spearmanr(Ds["self"], Ds["other"])[0], "self~object": spearmanr(Ds["self"], Ds["object"])[0],
                         "other~object": spearmanr(Ds["other"], Ds["object"])[0]}
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for p in range(2):
    for k, col in zip(["self~other", "self~object", "other~object"], ["tab:red", "tab:gray", "tab:brown"]):
        axs[p].plot([cross[(p, L)][k] for L in range(NL)], lw=3, color=col, label=k.replace("self", "your").replace("other", "woman's").replace("object", "doll's").replace("~", " vs "))
    axs[p].set_title(POSN[p]); axs[p].set_xlabel("layer")
axs[0].set_ylabel("agreement of location geometry\n(rank correlation)"); axs[0].legend(fontsize=14)
fig.suptitle(f"{TAG}: is the body map the same for your body, a woman's, a doll's?"); fig.tight_layout()
fig.savefig(f"{FIG}/{TAG}_self_other_object.png", dpi=110); plt.close()

# ---- body-agnostic map: PCA + classical MDS of cv distances, connect body neighbours
def edges():
    G = R["body path distance"]; return [(i, j) for i, j in zip(*iu) if 0 < G[i, j] <= 1.0]
def draw(ax, Y, title):
    for i, j in edges(): ax.plot(Y[[i, j], 0], Y[[i, j], 1], color="0.8", lw=1, zorder=0)
    for i, l in enumerate(S.LOCS):
        ax.scatter(*Y[i, :2], color=COL[l["limb"]], s=90)
        ax.annotate(l["key"], Y[i, :2], fontsize=10, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title); ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
def mds(D):
    D = np.maximum(D, 0); n = len(D); J = np.eye(n) - 1 / n
    B = -0.5 * J @ D @ J; w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    return v[:, o[:3]] * np.sqrt(np.maximum(w[o[:3]], 0))
for p in range(2):
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    for ax, c in zip(axs, conds):
        Dfull = cv_rdm(X(c, best, p)); draw(ax, mds((Dfull + Dfull.T) / 2), f"{ {'self':'your body','other':'a woman','object':'a doll'}[c] }")
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], marker="o", ls="", color=v) for v in COL.values()],
                  ["left arm", "right arm", "left leg", "right leg", "head/neck", "torso"], fontsize=12)
    fig.suptitle(f"{TAG}, layer {best}, {POSN[p]}: unlabeled 2-D map of the 54 locations (grey lines = neighbours on the body)")
    fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_map_{p}.png", dpi=100); plt.close()

# chain test (unlabeled): are body neighbours each other's nearest neighbours more than chance?
nn = {}
for c in conds:
    for p in range(2):
        Dm = cv_rdm(X(c, best, p)); np.fill_diagonal(Dm, np.inf)
        G = R["body path distance"]; hit = np.mean([G[i, np.argmin(Dm[i])] for i in range(nloc)])
        rng = np.random.default_rng(0); null = np.mean([np.mean([G[i, rng.choice([j for j in range(nloc) if j != i])] for i in range(nloc)]) for _ in range(200)])
        nn[(c, p)] = (float(hit), float(null))
print("mean body-path distance to nearest representational neighbour (vs random):", {f"{c}|{POSN[p]}": tuple(round(x, 2) for x in v) for (c, p), v in nn.items()})
json.dump(dict(best=best, lr_consistency={f"{c}|{p}": v for (c, p), v in axinfo.items()},
               nn={f"{c}|{p}": v for (c, p), v in nn.items()},
               cross={f"{p}|{L}": v for (p, L), v in cross.items()}), open(f"{OUT}/summary.json", "w"), indent=0)
for c in conds:
    for p in range(2):
        r = res[(c, p, best)]; print(c, POSN[p], {k: round(v, 3) for k, v in r.items()})

# ================= v2 additions: per-layer axis tests =================
side = np.array([l["side"] for l in S.LOCS]); depth = np.array([l["depth"] for l in S.LOCS])
types = sorted(set(l["type"] for l in S.LOCS if l["side"] == 1))
limbs = np.array([l["limb"] for l in S.LOCS])
def lr_acc(a):
    # a: nloc x T x d. hold out one homologous pair; direction from the other pairs (wording means);
    # test: each held-out wording's left item projects higher than its right item
    m = a.mean(1); ok = []
    for k, (i, j) in enumerate(pairs):
        dvec = np.mean([m[p] - m[q] for kk, (p, q) in enumerate(pairs) if kk != k], 0)
        ok += list((a[i] @ dvec) > (a[j] @ dvec))
    return float(np.mean(ok))
def pd_corr(a):
    # proximal-distal on arms+legs: ridge from location mean to depth, leave one part type out (both sides)
    m = a.mean(1); sel = np.where(np.isin(limbs, ["armL", "armR", "legL", "legR"]))[0]
    tys = np.array([S.LOCS[i]["type"] for i in sel]); pred = np.zeros(len(sel))
    for t in set(tys):
        tr, te = sel[tys != t], sel[tys == t]
        Xtr = m[tr] - m[tr].mean(0); y = depth[tr] - depth[tr].mean()
        w = Xtr.T @ np.linalg.solve(Xtr @ Xtr.T + 1e-2 * np.trace(Xtr @ Xtr.T) / len(tr) * np.eye(len(tr)) + 1e-6 * np.eye(len(tr)), y)
        pred[tys == t] = (m[te] - m[tr].mean(0)) @ w
    return float(spearmanr(pred, depth[sel])[0])
ax_res = {}
for c in conds:
    for p in range(2):
        for L in range(NL):
            a = X(c, L, p); ax_res[(c, p, L)] = (lr_acc(a), pd_corr(a))
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for p in range(2):
    for j, nm in enumerate(["left/right of a held-out body part\n(accuracy, chance 0.5)", "near/far from the trunk, held-out part\n(rank correlation)"]):
        for c, col in zip(conds, ["k", "tab:red", "tab:gray"]):
            axs[p, j].plot([ax_res[(c, p, L)][j] for L in range(NL)], lw=3, color=col, label={"self": "your body", "other": "a woman's body", "object": "a doll's body"}[c])
        axs[p, j].set_title(nm if p == 0 else ""); axs[p, j].set_ylabel(POSN[p])
        if p: axs[p, j].set_xlabel("layer")
axs[0, 0].legend(fontsize=14); fig.suptitle(f"{TAG}: a left-right axis and a trunk-to-fingertip axis, tested on held-out parts")
fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_axes_by_layer.png", dpi=110); plt.close()

# side-collapsed map: average left & right of each type -> body map without side
tkeys = [t for t in dict.fromkeys(l["type"] for l in S.LOCS)]
Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tkeys}
G = R["body path distance"]; rep = [Tidx[t][0] for t in tkeys]  # left-side (or midline) member for geometry
Gt = G[np.ix_(rep, rep)]; it = np.triu_indices(len(tkeys), 1)
Wl = [set(S.LOCS[i]["phrase"].replace("the ", "").replace("left ", "").replace("right ", "").split()) - {"of"} for i in rep]
Lt = np.array([[1 - len(a & b) / len(a | b) for b in Wl] for a in Wl])
tres = {}
for c in conds:
    for p in range(2):
        for L in range(NL):
            a = X(c, L, p); at = np.stack([a[Tidx[t]].mean(0) for t in tkeys])
            D = cv_rdm(at)[it]
            tres[(c, p, L)] = (spearmanr(D, Gt[it])[0], partial_spearman(D, Gt[it], Lt[it]), spearmanr(D, Lt[it])[0])
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for p in range(2):
    for c, col in zip(conds, ["k", "tab:red", "tab:gray"]):
        axs[p].plot([tres[(c, p, L)][0] for L in range(NL)], lw=3, color=col, label={"self": "your body", "other": "a woman's", "object": "a doll's"}[c])
        axs[p].plot([tres[(c, p, L)][1] for L in range(NL)], lw=2, ls="--", color=col)
    axs[p].set_title(POSN[p]); axs[p].set_xlabel("layer")
axs[0].set_ylabel("rank correlation with distance along the body\n(solid: raw, dashed: word overlap removed)"); axs[0].legend(fontsize=14)
fig.suptitle(f"{TAG}: body parts, left and right merged (30 parts)"); fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_parts_by_layer.png", dpi=110); plt.close()
bestT = int(max(range(2, NL), key=lambda L: tres[("self", 1, L)][1]))
print("best layer (parts, partial lexical, reply start):", bestT, [round(x, 3) for x in tres[("self", 1, bestT)]])
for L in range(0, NL, max(1, NL // 12)):
    print(L, "parts raw/partial self,other,object @reply:", [tuple(round(x, 2) for x in tres[(c, 1, L)][:2]) for c in conds],
          "LR acc:", [round(ax_res[(c, 1, L)][0], 2) for c in conds], "PD:", [round(ax_res[(c, 1, L)][1], 2) for c in conds])
json.dump(dict(axes={f"{c}|{p}|{L}": v for (c, p, L), v in ax_res.items()}, parts={f"{c}|{p}|{L}": v for (c, p, L), v in tres.items()}, bestT=bestT),
          open(f"{OUT}/summary2.json", "w"))
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
for ax, c in zip(axs, conds):
    a = X(c, bestT, 1); at = np.stack([a[Tidx[t]].mean(0) for t in tkeys]); Dm = cv_rdm(at); Y = mds((Dm + Dm.T) / 2)
    for i, j in zip(*it):
        if 0 < Gt[i, j] <= 1: ax.plot(Y[[i, j], 0], Y[[i, j], 1], color="0.8", lw=1.5, zorder=0)
    for i, t in enumerate(tkeys):
        ax.scatter(*Y[i, :2], color=COL[S.LOCS[rep[i]]["limb"]], s=110); ax.annotate(t.rstrip("L"), Y[i, :2], fontsize=12, xytext=(3, 3), textcoords="offset points")
    ax.set_title({"self": "your body", "other": "a woman's body", "object": "a doll's body"}[c]); ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
fig.suptitle(f"{TAG}, layer {bestT}, reply start: unlabeled map of 30 body parts (blue arm, red leg, green head, purple torso; grey = body neighbours)")
fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_partmap.png", dpi=100); plt.close()

# ================= v3: word-embedding baseline + cross-condition transfer =================
W = np.load(f"{OUT}/wordemb.npy").astype(np.float32)
Wt = np.stack([W[Tidx[t]].mean(0) for t in tkeys]); Wt -= Wt.mean(0)
Dw = ((Wt[:, None] - Wt[None]) ** 2).sum(-1)
print("word-embedding baseline: parts RDM vs body path rho =", round(spearmanr(Dw[it], Gt[it])[0], 3))
DwF = ((W[:, None] - W[None]) ** 2).sum(-1)
print("word-embedding baseline: 54-loc RDM vs body path rho =", round(spearmanr(DwF[iu], R["body path distance"][iu])[0], 3))
# partial out word embedding
pw = {c: [partial_spearman(cv_rdm(np.stack([X(c, L, 1)[Tidx[t]].mean(0) for t in tkeys]))[it], Gt[it], Dw[it]) for L in range(NL)] for c in conds}
print("parts vs body path, word-embedding partialled, reply start, by layer:", {c: [round(x, 2) for x in v[::3]] for c, v in pw.items()})
# transfer: nearest-centroid location decoding, centroids from condition A (even wordings) -> test on B (odd wordings)
def transfer(L, p, A, B):
    a = X(A, L, p); b = X(B, L, p); C = a[:, 0::2].mean(1); C = C - C.mean(0)
    te = b[:, 1::2]; te = te - te.reshape(-1, te.shape[-1]).mean(0)
    d = ((te[:, :, None, :] - C[None, None]) ** 2).sum(-1); pred = d.argmin(-1)
    return float((pred == np.arange(nloc)[:, None]).mean())
tr = {(A, B): [transfer(L, 1, A, B) for L in range(NL)] for A in conds for B in conds}
print("location decoding (54-way, chance 0.019) reply start, train->test, by layer every 3:")
for k, v in tr.items(): print("  ", k, [round(x, 2) for x in v[::3]])
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
for c, col in zip(conds, ["k", "tab:red", "tab:gray"]):
    axs[0].plot(pw[c], lw=3, color=col, label={"self": "your body", "other": "a woman's", "object": "a doll's"}[c])
axs[0].axhline(0, color="k", lw=.5); axs[0].set_xlabel("layer"); axs[0].set_ylabel("rank corr. with distance along body\n(word meaning partialled out)")
axs[0].set_title("body map beyond word similarity"); axs[0].legend(fontsize=14)
for (A, B), st, col in [(("self", "self"), "-", "k"), (("self", "other"), "-", "tab:red"), (("self", "object"), "-", "tab:gray"), (("other", "object"), "--", "tab:brown")]:
    axs[1].plot(tr[(A, B)], ls=st, lw=3, color=col, label=f"learn on {A}, test on {B}".replace("self", "you").replace("other", "woman").replace("object", "doll"))
axs[1].axhline(1 / nloc, color="k", lw=.5); axs[1].set_xlabel("layer"); axs[1].set_ylabel("which of 54 locations? (accuracy)")
axs[1].set_title("same location code for you, a woman, a doll?"); axs[1].legend(fontsize=12)
fig.suptitle(f"{TAG}, model state at reply start"); fig.tight_layout(); fig.savefig(f"{FIG}/{TAG}_partial_transfer.png", dpi=110); plt.close()
json.dump(dict(partial_wordemb=pw, transfer={f"{a}>{b}": v for (a, b), v in tr.items()}), open(f"{OUT}/summary3.json", "w"))
