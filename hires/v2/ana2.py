"""Unsupervised body map metrics, identical for v1 (body/hires) and v2 data. Labels/coords used only for scoring."""
import json, numpy as np, sys, os, importlib.util
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
def load_stim(path):
    sp = importlib.util.spec_from_file_location("stim_" + str(abs(hash(path))), path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
class Body:
    def __init__(s, S):
        s.S = S; s.L = S.LOCS; s.n = len(s.L); s.P = np.array([l["xyz"] for l in s.L]); s.side = np.array([l["side"] for l in s.L])
        Pf = s.P.copy(); Pf[:, 0] = abs(Pf[:, 0]); s.DF = squareform(pdist(Pf)); s.iu = np.triu_indices(s.n, 1)
        ph = [l["phrase"] for l in s.L]; s.ph = ph
        s.pairs = np.array([(i, ph.index(p.replace("left", "right"))) for i, p in enumerate(ph) if "left" in p and p.replace("left", "right") in ph])
        s.shafts = {}
        for i, p in enumerate(ph):
            if ", " in p and any(w in p for w in ["upper arm", "forearm", "thigh", "shin", "calf", "lower leg"]): s.shafts.setdefault(p.split(", ")[0], []).append(i)
        s.chains = s._chains()
    def _chains(s):
        out = []
        for sd in (1, -1):
            reg = lambda r: [i for i, l in enumerate(s.L) if l["region"] == r + ("L" if sd > 0 else "R")]
            arm, hand, leg, foot = reg("arm"), reg("hand"), reg("leg"), reg("foot")
            g = lambda ix: [i for i in ix]
            sh = [i for i in arm if "shoulder" in s.ph[i].split(",")[0] or "armpit" in s.ph[i]]
            ua = {}; fa = {}
            for i in arm:
                if "upper arm" in s.ph[i]: ua.setdefault(round(abs(s.P[i, 0])), []).append(i)
                if "forearm" in s.ph[i]: fa.setdefault(round(abs(s.P[i, 0])), []).append(i)
            el = [i for i in arm if "elbow" in s.ph[i].split(",")[0]]
            wr = [i for i in hand if "wrist" in s.ph[i]]; palm = [i for i in hand if "wrist" not in s.ph[i] and not any(f in s.ph[i] for f in ["finger", "thumb", "knuckle", "nail", "tip"])]
            out.append([sh] + [ua[k] for k in sorted(ua)] + [el] + [fa[k] for k in sorted(fa)] + [wr, palm])
            hip = [i for i in leg if "hip" in s.ph[i].split(",")[0] or "buttock" in s.ph[i]]
            th = {}; sn = {}
            for i in leg:
                if "thigh" in s.ph[i]: th.setdefault(round(s.P[i, 1]), []).append(i)
                elif ", " in s.ph[i]: sn.setdefault(round(s.P[i, 1]), []).append(i)
            kn = [i for i in leg if "knee" in s.ph[i].split(",")[0]]
            an = [i for i in foot if "ankle" in s.ph[i] or "Achilles" in s.ph[i]]; toes = [i for i in foot if "toe" in s.ph[i]]; ft = [i for i in foot if i not in an and i not in toes]
            out.append([hip] + [th[k] for k in sorted(th, reverse=True)] + [kn] + [sn[k] for k in sorted(sn, reverse=True)] + [an, ft, toes])
        return out
    def metrics(s, Z, ZW=None, probes=None):
        DZ = squareform(pdist(Z)); np.fill_diagonal(DZ, np.inf); nn = np.argsort(DZ, 1)[:, :5]
        m = dict(nn5_cm=np.mean([s.DF[i, nn[i]].mean() for i in range(s.n)]))
        m["rho_folded"] = spearmanr(pdist(Z), s.DF[s.iu])[0]
        m["other_side_nn"] = np.mean([np.mean(s.side[nn[i]] == -s.side[i]) for i in np.where(s.side != 0)[0]])
        ch = []
        for c in s.chains:
            C = np.array([Z[g].mean(0) for g in c]); ch.append(spearmanr(np.linalg.norm(C - C[0], axis=1), np.arange(len(c)))[0])
        m["chain_rho"] = np.mean(ch)
        so = []
        for k, ix in s.shafts.items():
            ix = sorted(ix, key=lambda i: np.linalg.norm(s.P[i] - s.P[ix[0]]))  # first listed = most proximal
            so.append(spearmanr(np.linalg.norm(Z[ix] - Z[ix[0]], axis=1)[1:], np.arange(1, len(ix)))[0])
        m["shaft_rho"] = np.mean(so)
        m["shaft_pair_rho"] = np.mean([spearmanr(pdist(Z[ix]), pdist(s.P[ix]))[0] for ix in s.shafts.values()])
        pr = s.pairs; d = Z[pr[:, 0]] - Z[pr[:, 1]]; md = d.mean(0); m["mirror_shift"] = 1 - ((d - md) ** 2).sum() / (d ** 2).sum()
        Zs = Z.copy(); Zs[pr[:, 0]] -= md / 2; Zs[pr[:, 1]] += md / 2; D2 = squareform(pdist(Zs)); np.fill_diagonal(D2, np.inf)
        m["partner_nn"] = np.mean(np.argmin(D2[pr[:, 0]], 1) == pr[:, 1])
        if ZW is not None:
            for kind in sorted(set(p["kind"] for p in probes)):
                ix = [j for j, p in enumerate(probes) if p["kind"] == kind]; W = ZW[ix] - ZW[ix].mean(0) + Z.mean(0); r = []; cm = []
                for j, w in zip(ix, W):
                    o = np.argsort(np.linalg.norm(Z - w, axis=1)); t = probes[j]["target"]; r.append(np.where(o == t)[0][0]); cm.append(s.DF[o[:5], t].mean())
                if kind == "alt":
                    sh = {i: ix for ix in s.shafts.values() for i in ix}; e = []; ch = []
                    for j, w in zip(ix, W):
                        t = probes[j]["target"]; c = sh[t]; k = c[np.argmin(np.linalg.norm(Z[c] - w, axis=1))]
                        e.append(np.linalg.norm(s.P[k] - s.P[t])); ch.append(np.mean([np.linalg.norm(s.P[q] - s.P[t]) for q in c]))
                    m["alt_inshaft_cm"] = np.mean(e); m["alt_inshaft_chance_cm"] = np.mean(ch)
                m[f"probe_{kind}_rank"] = float(np.median(r)); m[f"probe_{kind}_pct"] = float(np.median(r) / s.n * 100); m[f"probe_{kind}_cm"] = float(np.median(cm))
        return {k: round(float(v), 3) for k, v in m.items()}
from sm import spot_means
def embed(A, W, k=30, whiten=False):
    mu = A.mean(0); An = A - mu; nr = np.linalg.norm(An, axis=1, keepdims=True); An = An / nr
    Wn = W - mu; Wn = Wn / np.linalg.norm(Wn, axis=1, keepdims=True)
    p = PCA(k, whiten=whiten).fit(An); return p.transform(An), p.transform(Wn)
