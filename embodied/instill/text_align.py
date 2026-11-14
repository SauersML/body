"""(b) Does the embodied skin map reuse plain Qwen's text body map? Offline tests on saved states.
For each run, checkpoint, layer: (1) segment classifier trained on plain-Qwen sentences (upper arm / forearm / hand words)
applied to embodied skin centroids, accuracy vs permutation null; (2) share of embodied arc variance inside the text
body-part subspace vs the text wording subspace (same text, not about place) and vs a random subspace;
(3) embodied arc direction applied to the 9 part words: Spearman with true order, exact permutation p."""
import json, glob, os, itertools, numpy as np
from scipy.stats import spearmanr
SM = os.environ.get("SMOKE")
R = "$BODY_RUNS{}{}"
ARCS = np.linspace(.03, 1.12, 23); BOUND = [.5, .95]
seg_e = np.array([int(a > BOUND[0]) + int(a > BOUND[1]) for a in ARCS] * 2)
PSEG = {0: 0, 1: 0, 2: 0, 3: None, 4: 1, 5: None, 6: 2, 7: 2, 8: 2}  # shoulder upper-arm biceps | elbow | forearm | wrist | palm knuckles fingertips
PARC = [.02, .25, .30, .50, .72, .95, 1.02, 1.07, 1.13]
perms9 = np.array([np.random.default_rng(i).permutation(9) for i in range(500)])


def lda_fit(X, y, al=1e1):
    """ridge one-vs-rest on standardised features (dual form)."""
    mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd; K = Z @ Z.T
    Y = np.stack([(y == c).astype(float) * 2 - 1 for c in range(3)], 1)
    A = np.linalg.solve(K + al * np.eye(len(Z)), Y - Y.mean(0)); W = Z.T @ A
    return lambda Xb, own=True: ((Xb - (Xb.mean(0) if own else mu)) / sd) @ W


def analyse(nz):
    tm = nz["tmeta"]; steps = nz["steps"]; out = {}
    for l in ["12", "18", "27", "36"]:
        TX = nz["text_" + l].astype(float); cent = nz["cent_" + l].astype(float)
        keep = np.array([PSEG[int(p)] is not None for p in tm[:, 2]]); ys = np.array([PSEG[int(p)] if PSEG[int(p)] is not None else -1 for p in tm[:, 2]])
        # text-internal sanity: leave-one-wording-out segment acc
        wid = np.tile(np.repeat(np.arange(8), 1), 18)[:len(tm)] if False else np.array([i % 8 for i in range(len(tm))])
        acc_t = np.mean([(lda_fit(TX[keep & (wid != w)], ys[keep & (wid != w)])(TX[keep & (wid == w)], False).argmax(1) == ys[keep & (wid == w)]).mean() for w in range(8)])
        f = lda_fit(TX[keep], ys[keep])
        acc = [float((f(cent[t]).argmax(1) == seg_e).mean()) for t in range(len(cent))]
        rng = np.random.default_rng(0); null = []
        for _ in range(50):
            pm = rng.permutation(9); ysh = np.array([PSEG[int(pm[int(p)])] if PSEG[int(pm[int(p)])] is not None else -1 for p in tm[:, 2]]); k2 = ysh >= 0
            fn = lda_fit(TX[k2], ysh[k2]); null.append((fn(cent[-1]).argmax(1) == seg_e).mean())
        # subspaces
        C = np.stack([TX[(tm[:, 0] == s) & (tm[:, 2] == p)].mean(0) for s in range(2) for p in range(9)]); C -= C.mean(0)
        Wd = np.stack([TX[wid == w].mean(0) for w in range(8)]); Wd -= Wd.mean(0)
        Ub = np.linalg.svd(C, full_matrices=False)[2][:7]; Uw = np.linalg.svd(Wd, full_matrices=False)[2][:7]
        Ur = np.linalg.qr(np.random.default_rng(1).normal(size=(TX.shape[1], 7)))[0].T
        def share(M, U): M = M - M.mean(0); return float((np.linalg.norm(M @ U.T) ** 2) / np.linalg.norm(M) ** 2)
        sb = [share(c, Ub) for c in cent]; sw = [share(c, Uw) for c in cent]; sr = [share(c, Ur) for c in cent]
        # arc-only variance (remove arm mean so left/right split is not counted)
        def arcshare(M, U):
            M = M.copy(); M[:23] -= M[:23].mean(0); M[23:] -= M[23:].mean(0); return float((np.linalg.norm(M @ U.T) ** 2) / np.linalg.norm(M) ** 2)
        ab = [arcshare(c, Ub) for c in cent]; aw = [arcshare(c, Uw) for c in cent]
        # reverse: embodied arc direction -> order of the 9 words
        rev, revp = [], []
        for c in cent:
            Mc = c - c.mean(0); y = np.tile(ARCS, 2); w = Mc.T @ np.linalg.solve(Mc @ Mc.T + 1e-2 * np.trace(Mc @ Mc.T) / 46 * np.eye(46), y - y.mean())
            pr = np.array([(TX[tm[:, 2] == p] @ w).mean() for p in range(9)]); r = spearmanr(pr, PARC)[0]
            nr = np.array([spearmanr(pr, np.array(PARC)[pp])[0] for pp in perms9]); rev.append(float(r)); revp.append(float((np.abs(nr) >= abs(r)).mean()))
        out[l] = dict(text_internal_seg_acc=float(acc_t), seg_acc=acc, seg_null95=float(np.percentile(null, 95)), seg_null_mean=float(np.mean(null)),
                      share_body=sb, share_wording=sw, share_random=sr, arcshare_body=ab, arcshare_wording=aw, rev=rev, rev_p=revp)
    return steps.tolist(), out


res = {}
for a, an in [("named", "named"), ("namel", "nameless")]:
    for s in range(3):
        f = glob.glob(R.format(a, s) + f"/form_Qwen3-8B_{an}_s{s}.npz") if not SM else glob.glob("$BODY_RUNS/form_*.npz")
        if f: res[f"{an}_s{s}"] = analyse(np.load(f[0]))
json.dump(res, open("$BODY_WORK/body_instill/text_align.json", "w"))
for k, (st, o) in res.items():
    print("==", k, st)
    for l in ["18", "27"]:
        d = o[l]; print(l, "textseg", round(d["text_internal_seg_acc"], 2), "null95", round(d["seg_null95"], 2), "nullmean", round(d["seg_null_mean"], 2))
        for key in ["seg_acc", "share_body", "share_wording", "share_random", "arcshare_body", "arcshare_wording", "rev", "rev_p"]:
            print(f"  {key:16s}", " ".join(f"{x:.3f}" if 'share' in key else f"{x:5.2f}" for x in d[key]))
