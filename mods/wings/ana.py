"""python ana.py OUTDIR  -> OUTDIR/ana.json and printed summary."""
import json, sys, re, itertools, numpy as np
from scipy.stats import spearmanr
O = sys.argv[1]; R = json.load(open(f"{O}/probes.json"))
KEEP = [16, 20, 22, 24, 25, 26, 27, 28, 30, 32] if "8b" in O else [36, 40, 44, 48, 50, 52, 54, 56, 60]
kind = np.array([r["kind"] for r in R]); key = np.array([r["key"] for r in R]); side = np.array([r["side"] for r in R]); tpl = np.array([r["tpl"] for r in R])
HW = np.load(f"{O}/H_W.npy").astype(np.float32); HN = np.load(f"{O}/H_N.npy").astype(np.float32); LEX = np.load(f"{O}/lex.npy").astype(np.float32)[None]
CH = ["root", "base", "inner", "mid", "outer", "longf", "tip"]; TT = dict(root=0, base=.1, inner=.25, mid=.5, outer=.7, longf=.88, tip=1)
SPL = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
def cent(H, k, kd, s, ts=(0, 1, 2, 3)): return H[(kind == kd) & (key == k) & (side == s) & np.isin(tpl, ts)].mean(0)
def cvd(H, kd, keys, s):
    """cross-validated squared distance matrix between spots (unbiased over template splits)."""
    n = len(keys); Dm = np.zeros((n, n))
    for a, b in SPL:
        A = np.stack([cent(H, k, kd, s, a) for k in keys]); B = np.stack([cent(H, k, kd, s, b) for k in keys])
        for i, j in itertools.combinations(range(n), 2): Dm[i, j] += (A[i] - A[j]) @ (B[i] - B[j]) / 3
    return Dm + Dm.T
def chain_rho(H, kd="wing"):
    x, y = [], []
    for s in ("left", "right"):
        Dm = cvd(H, kd, CH, s)
        for i, j in itertools.combinations(range(7), 2): x.append(Dm[i, j]); y.append(abs(TT[CH[i]] - TT[CH[j]]))
    return spearmanr(x, y)[0], x
def perm_p(H, kd="wing", n=2000):
    r0, x = chain_rho(H, kd); rng = np.random.default_rng(0); x = np.array(x); null = []
    pairs = list(itertools.combinations(range(7), 2))
    for _ in range(n):  # permute spot labels
        p = rng.permutation(7); t = np.array([TT[k] for k in CH])[p]
        y = [abs(t[i] - t[j]) for i, j in pairs] * 2; null.append(spearmanr(x, y)[0])
    return r0, float(np.mean(np.array(null) >= r0))
HUM = sorted(set(key[kind == "hum"]))
def hum_cent(H):
    C = {}
    for k in HUM:
        for s in set(side[(kind == "hum") & (key == k)]): C[f"{k}:{s}"] = cent(H, k, "hum", s)
    return C
ARM = ["shtop", "uarm", "elbow", "farm", "wrist", "bhand", "ftip"]; LEG = ["hip", "thigh", "knee", "foot"]
res = {}
for li, L in enumerate(KEEP):
    r = {}
    for cn, H in (("W", HW[li]), ("N", HN[li])):
        mu = H[np.isin(kind, ["wing", "hum"])].mean(0); Hc = H - mu
        r[f"chain_{cn}"], r[f"chain_p_{cn}"] = perm_p(Hc)
        if cn == "W": r["chain_her_W"], r["chain_her_p_W"] = perm_p(Hc, "owing")
        C = hum_cent(Hc); names = list(C); M = np.stack([C[n] for n in names]); Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        near = {}
        for kd in ("wing", "owing", "glorp"):
            for k in sorted(set(key[kind == kd])):
                for s in ("left", "right"):
                    v = cent(Hc, k, kd, s); c = Mn @ (v / np.linalg.norm(v)); o = np.argsort(-c)[:4]
                    near[f"{kd}:{k}:{s}"] = [(names[i], round(float(c[i]), 3)) for i in o]
        r[f"near_{cn}"] = near
        # arm axis projection (wings as arms?) and leg control
        for ax, (a0, a1) in {"arm": ("shtop", "ftip"), "leg": ("hip", "foot")}.items():
            rh = []
            for s in ("left", "right"):
                d = cent(Hc, a1, "hum", s) - cent(Hc, a0, "hum", s); d /= np.linalg.norm(d)
                rh += [(TT[k], float(cent(Hc, k, "wing", s) @ d)) for k in CH]
            r[f"{ax}proj_{cn}"] = spearmanr(*zip(*rh))[0]
        # left/right: separation index (cv distance same spot other side / cv distance other spot same side), wings vs arm parts
        def sep(kd, keys):
            out = []
            for k in keys:
                for a, b in SPL:
                    Ls = cent(Hc, k, kd, "left", a) - cent(Hc, k, kd, "right", a); Lb = cent(Hc, k, kd, "left", b) - cent(Hc, k, kd, "right", b)
                    out.append(Ls @ Lb)
            return float(np.mean(out))
        wsp = cvd(Hc, "wing", CH, "left"); asp = cvd(Hc, "hum", ARM, "left")
        r[f"sep_wing_{cn}"] = sep("wing", CH) / float(np.mean(wsp[np.triu_indices(7, 1)]))
        r[f"sep_arm_{cn}"] = sep("hum", ARM) / float(np.mean(asp[np.triu_indices(7, 1)]))
        r[f"lr_sq_wing_{cn}"] = sep("wing", CH); r[f"lr_sq_arm_{cn}"] = sep("hum", ARM)
        # does the human left-right axis (from non-arm human parts) sort wings by side?
        hl = [k for k in HUM if k not in ARM and (kind[(key == k)] == "hum").any() and "left" in set(side[key == k])]
        ax = np.mean([cent(Hc, k, "hum", "left") - cent(Hc, k, "hum", "right") for k in hl], 0); ax /= np.linalg.norm(ax)
        acc = np.mean([(cent(Hc, k, "wing", "left", (t,)) @ ax > cent(Hc, k, "wing", "right", (t,)) @ ax) for k in CH + ["lead", "under", "trail"] for t in range(4)])
        r[f"lr_axis_acc_{cn}"] = float(acc)
        wd = np.mean([np.linalg.norm(cent(Hc, k, "wing", "left") - cent(Hc, k, "wing", "right")) for k in CH])
        r[f"lr_cos_wing_{cn}"] = float(np.mean([((cent(Hc, k, "wing", "left") - cent(Hc, k, "wing", "right")) @ ax) / np.linalg.norm(cent(Hc, k, "wing", "left") - cent(Hc, k, "wing", "right")) for k in CH]))
        r[f"lr_cos_arm_{cn}"] = float(np.mean([((cent(Hc, k, "hum", "left") - cent(Hc, k, "hum", "right")) @ ax) / np.linalg.norm(cent(Hc, k, "hum", "left") - cent(Hc, k, "hum", "right")) for k in ARM]))
    # self->other transfer: nearest-centroid decoding of wing spot (10 keys x side), train own wing W, test her wing W
    Hc = HW[li] - HW[li][np.isin(kind, ["wing", "hum"])].mean(0)
    keys = sorted(set(key[kind == "wing"])); labs = [(k, s) for k in keys for s in ("left", "right")]
    Cs = np.stack([cent(Hc, k, "wing", s) for k, s in labs]); ok = []; oks = []
    for i, (k, s) in enumerate(labs):
        for t in range(4):
            v = Hc[(kind == "owing") & (key == k) & (side == s) & (tpl == t)][0]; j = np.argmin(np.linalg.norm(Cs - v, axis=1)); ok.append(j == i)
            # within self, leave-one-template-out
            C2 = np.stack([cent(Hc, k2, "wing", s2, tuple(x for x in range(4) if x != t)) for k2, s2 in labs])
            v2 = Hc[(kind == "wing") & (key == k) & (side == s) & (tpl == t)][0]; oks.append(np.argmin(np.linalg.norm(C2 - v2, axis=1)) == i)
    r["decode_self"] = float(np.mean(oks)); r["decode_self_to_her"] = float(np.mean(ok)); r["decode_chance"] = 1 / len(labs)
    res[L] = r
lr, lp = chain_rho(LEX[0] - LEX[0][np.isin(kind, ["wing", "hum"])].mean(0)), None
res["lex_chain"] = perm_p(LEX[0] - LEX[0].mean(0))
json.dump(res, open(f"{O}/ana.json", "w"), indent=1)
print("lex chain", res["lex_chain"])
for L in KEEP:
    r = res[L]; print(L, " ".join(f"{k}={v:.2f}" for k, v in r.items() if isinstance(v, float)))
for L in KEEP[len(KEEP)//2-1:len(KEEP)//2+2]:
    for k in ("wing:root:left", "wing:mid:left", "wing:tip:left", "wing:tip:right", "wing:lead:left", "wing:under:left", "owing:tip:left", "glorp:gtip:left", "glorp:groot:left"):
        print(L, k, res[L]["near_W"][k], "| N:", res[L]["near_N"][k][:2])
