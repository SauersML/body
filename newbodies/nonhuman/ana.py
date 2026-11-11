"""Analyse run.py output. usage: ana.py OUT LAYER  -> OUT/ana_L{LAYER}.json"""
import sys, os, json, re, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stimuli as S, bodies as BD
OUT, LAYER = sys.argv[1], int(sys.argv[2])
rows = json.load(open(f"{OUT}/rows.json")); LAYERS = json.load(open(f"{OUT}/layers.json")); li = LAYERS.index(LAYER)
H = np.load(f"{OUT}/H.npy", mmap_mode="r")
PE = json.load(open(f"{OUT}/phrase_emb.json"))
chat_a = np.array([r["chat"] for r in rows]); loc_a = np.array([r["loc"] for r in rows]); tpl_a = np.array([r["tpl"] for r in rows])
X = {c: np.asarray(H[chat_a == c, li], np.float32) for c in dict.fromkeys(chat_a)}
L = {c: loc_a[chat_a == c] for c in X}; TP = {c: tpl_a[chat_a == c] for c in X}
rng = np.random.default_rng(0)
def unit(M): return M / np.linalg.norm(M, axis=1, keepdims=True)
def cents(c, keys, tpls=None, labels=None):
    lab = L[c] if labels is None else labels; m = np.ones(len(lab), bool) if tpls is None else np.isin(TP[c], tpls)
    idx = {k: i for i, k in enumerate(keys)}; li_ = np.array([idx.get(x, -1) for x in lab]); ok = m & (li_ >= 0)
    O = np.zeros((len(keys), len(lab)), np.float32); O[li_[ok], np.nonzero(ok)[0]] = 1; O /= O.sum(1, keepdims=True)
    M = O @ X[c]; return M - M.mean(0)
def rdm(M): U = unit(M); return 1 - U @ U.T
iu = lambda n: np.triu_indices(n, 1)
def rsa(A, B): i = iu(len(A)); return float(spearmanr(A[i], B[i])[0])
def perm_p(A, B, n=2000):
    r0 = rsa(A, B); c = 0
    for _ in range(n):
        p = rng.permutation(len(A)); c += rsa(A[np.ix_(p, p)], B) >= r0
    return (c + 1) / (n + 1)
def partial(A, B, C):  # spearman of A,B controlling C
    i = iu(len(A)); from scipy.stats import rankdata
    a, b, c = (rankdata(M[i]) for M in (A, B, C))
    ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])
HK = [l["key"] for l in S.LOCS]; HG = S.model_rdms()["body path distance"]
HM = cents("human", HK); HMu = unit(HM); HMo = unit(cents("human|other", HK))
ARM = {k for k in HK if S.LOCS[HK.index(k)]["limb"].startswith("arm")}; LEG = {k for k in HK if S.LOCS[HK.index(k)]["limb"].startswith("leg")}
HANDS = {k for k in ARM if S.LOCS[HK.index(k)]["depth"] >= 5}; FEET = {k for k in LEG if S.LOCS[HK.index(k)]["depth"] >= 5}
out = dict(layer=LAYER, bodies={})
# human reference
o = [0, 1, 2, 3]; e = [4, 5, 6, 7]
hA, hB = cents("human", HK, o), cents("human", HK, e)
out["human"] = dict(rsa_true=rsa(rdm(HM), HG), rsa_split=(rsa(rdm(hB), HG) + rsa(rdm(hA), HG)) / 2, reliab=rsa(rdm(hA), rdm(hB)),
                    emb_rsa=rsa(rdm(np.array([PE["your " + l["phrase"]] for l in S.LOCS]) - np.mean([PE["your " + l["phrase"]] for l in S.LOCS], 0)), HG),
                    rsa_other=rsa(rdm(cents("human|other", HK)), HG))
print(f"human: RSA true {out['human']['rsa_true']:.2f}  split {out['human']['rsa_split']:.2f}  reliab {out['human']['reliab']:.2f}  emb {out['human']['emb_rsa']:.2f}  other {out['human']['rsa_other']:.2f}")
for b in BD.ORDER:
    if b not in X: continue
    parts = BD.B[b]["parts"]; ks = [p["key"] for p in parts]; G = BD.geodesic(b); r = dict(n=len(ks), keys=ks)
    for cond, c, ref in [("self", b, HMu), ("nonce", b + "|nonce", HMu), ("other", b + "|other", HMo)]:
        C = cents(c, ks); S_ = unit(C) @ ref.T  # parts x 54
        best = S_.argmax(1)
        nul = []
        for _ in range(200):
            lab = L[c][rng.permutation(len(L[c]))]; nul += list((unit(cents(c, ks, labels=lab)) @ ref.T).max(1))
        n95 = float(np.percentile(nul, 95)); n50 = float(np.median(nul))
        R = rdm(C); ph = {k: rows_ph for k, rows_ph in zip(L[c], [rr["phrase"] for rr in rows if rr["chat"] == c])}
        EM = np.array([PE[ph[k]] for k in ks]); EM -= EM.mean(0); ER = rdm(EM)
        # cross-validated human folding: map parts to the nearest human place with one half of the wordings, test on the other half
        fold, true_cv = [], []
        for A_, B_ in [(o, e), (e, o)]:
            ref_s = unit(cents("human" if cond != "other" else "human|other", HK, A_))
            mp = (unit(cents(c, ks, A_)) @ ref_s.T).argmax(1); F = HG[np.ix_(mp, mp)]
            RB = rdm(cents(c, ks, B_)); fold.append(rsa(RB, F) if np.ptp(F[iu(len(ks))]) > 0 else 0.0); true_cv.append(rsa(RB, G))
        r[cond] = dict(best=[HK[j] for j in best], best_cos=S_.max(1).round(3).tolist(), null50=n50, null95=n95,
                       frac_above95=float((S_.max(1) > n95).mean()), sims=S_.round(3).tolist(),
                       rsa_true=rsa(R, G), p_true=perm_p(R, G, 1000), rsa_emb_true=rsa(ER, G), rsa_partial=partial(R, G, ER),
                       rsa_true_cv=float(np.mean(true_cv)), rsa_fold_cv=float(np.mean(fold)),
                       reliab=rsa(rdm(cents(c, ks, o)), rdm(cents(c, ks, e))),
                       pull=[float(S_[i, [HK.index(k) for k in ARM]].mean() - S_[i, [HK.index(k) for k in LEG]].mean()) for i in range(len(ks))],
                       rdm=R.round(4).tolist())
        print(f"{b:9s} {cond:5s} best>null95 {r[cond]['frac_above95']:.2f}  RSA true {r[cond]['rsa_true']:.2f} (p {r[cond]['p_true']:.3f}) emb {r[cond]['rsa_emb_true']:.2f} partial {r[cond]['rsa_partial']:.2f}"
              f"  cv true {r[cond]['rsa_true_cv']:.2f} vs human-folded {r[cond]['rsa_fold_cv']:.2f}  reliab {r[cond]['reliab']:.2f}")
    print("   self map:", ", ".join(f"{k}->{h} {c:.2f}" for k, h, c in zip(ks, r["self"]["best"], r["self"]["best_cos"])))
    r["geo"] = G.round(3).tolist(); out["bodies"][b] = r
# imagining / reframing in the model's reply to the body description
IM = re.compile(r"imagin|(don't|do not|doesn't) (actually )?have a (physical )?body|as an ai|if i were|pretend|i'm an ai|i am an ai|in my mind|metaphor", re.I)
pre = json.load(open(f"{OUT}/prefixes.json"))
out["imagine"] = {c: bool(IM.search(m[3]["content"])) for c, m in pre.items()}
out["intro_reply"] = {c: m[3]["content"][:400] for c, m in pre.items()}
rep = json.load(open(f"{OUT}/replies.json"))
out["imagine_touch"] = {}
for c in dict.fromkeys(x["chat"] for x in rep):
    xs = [x["reply"] for x in rep if x["chat"] == c]; out["imagine_touch"][c] = [sum(bool(IM.search(x)) for x in xs), len(xs)]
print("imagine in intro reply:", {c: v for c, v in out["imagine"].items()})
json.dump(out, open(f"{OUT}/ana_L{LAYER}.json", "w"))
