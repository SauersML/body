"""Analyse body-map formation runs. python analyze_form.py -> form_summary.json + printed tables"""
import json, glob, numpy as np, sys
from scipy.stats import spearmanr
sys.path.insert(0, "$BODY_RUNS/body3/form")
R = "$BODY_RUNS{}{}"
runs = {}
for a, an in [("named", "named"), ("namel", "nameless")]:
    for s in range(3):
        import os
        f = glob.glob(R.format(a, s) + f"/form_Qwen3-8B_{an}_s{s}.json") if not os.environ.get("SMOKE") else glob.glob("$BODY_RUNS/form_*.json")
        if f: runs[(an, s)] = (json.load(open(f[0])), np.load(f[0].replace(".json", ".npz")))
print("runs", sorted(runs))
L = "18"
ARCS = np.linspace(.03, 1.15 - .03, 23)
geo = np.array([[abs(ai - aj) if a == b else ai + aj + .6 for b in range(2) for aj in ARCS] for a in range(2) for ai in ARCS])
iu = np.triu_indices(46, 1)
win = np.array([[a == b for b in range(2) for _ in ARCS] for a in range(2) for _ in ARCS])[iu]


def rdm(M):
    M = M.astype(float); M = M - M.mean(0); return np.linalg.norm(M[:, None] - M[None], axis=-1)


def cka(X, Y):
    X = X.astype(float) - X.mean(0); Y = Y.astype(float) - Y.mean(0)
    K, Lm = X @ X.T, Y @ Y.T
    return float((K * Lm).sum() / (np.linalg.norm(K) * np.linalg.norm(Lm)))


def ridge_w(X, y, al=1e2):
    X = X.astype(float); Z = X - X.mean(0); sd = Z.std(0) + 1e-6; Z = Z / sd
    return (Z.T @ np.linalg.solve(Z @ Z.T + al * np.eye(len(Z)), y - y.mean())) / sd


def curves(res):
    out = {}
    g = lambda f: [f(r) for r in res]
    out["step"] = g(lambda r: r["step"])
    for l in ["inject", "7", "12", "18", "27", "36"]: out["rsa_w_" + l] = g(lambda r: r["rsa_within"][l])
    out["rsa_all_18"] = g(lambda r: r["rsa"]["18"])
    out["arc_r2_18"] = g(lambda r: r["arc_probe_r2"]["18"])
    out["pose_18"] = g(lambda r: r["pose_r2"]["18"]); out["pose_inj"] = g(lambda r: r["pose_r2"]["inject"])
    out["x_18"] = g(lambda r: r["x_r2"]["18"]); out["x_inj"] = g(lambda r: r["x_r2"]["inject"])
    out["self_18"] = g(lambda r: r["self_probe"]["18"]); out["self_inj"] = g(lambda r: r["self_probe"]["inject"])
    out["self_verbal"] = g(lambda r: np.mean(list(r["eval"]["self"].values())))
    out["pos_mae"] = g(lambda r: np.mean(list(r["eval"]["pos"].values())))
    for k in ["reach", "find", "wall"]: out["cl_" + k] = g(lambda r: r["closed_loop"][k])
    out["task"] = list(np.mean([out["cl_reach"], out["cl_find"], out["cl_wall"]], 0))
    for l in ["12", "18", "27", "36"]: out["tt_" + l] = g(lambda r: r["text_transfer"][l]); out["ttrev_" + l] = g(lambda r: r["text_transfer"][l + "_rev"]); out["ttside_" + l] = g(lambda r: r["text_transfer"][l + "_side"])
    return out


def onset(step, v, frac, rising=True):
    v = np.array(v, float); d = v[-1] - v[0]
    if abs(d) < 1e-9: return None
    p = (v - v[0]) / d
    for s, x in zip(step, p):
        if x >= frac: return s
    return None


S = {}
for k, (js, nz) in runs.items():
    c = curves(js["res"]); S[k] = c
    # text alignment offline: RDM of 18 text centroids (9 parts x 2 sides) vs embodied centroids at nearest arcs
    tm = nz["tmeta"]; PARTS = js["PARTS"]; parc = [a for _, a in PARTS]
    near = [int(np.argmin(abs(ARCS - a))) for a in parc]
    c["text_rdm"], c["text_rdm_partial"], c["text_null95"] = {}, {}, {}
    for l in ["12", "18", "27", "36"]:
        TX = nz["text_" + l].astype(float)
        T = np.stack([TX[(tm[:, 0] == s) & (tm[:, 2] == p)].mean(0) for s in range(2) for p in range(9)])
        Dt = rdm(T); iu18 = np.triu_indices(18, 1)
        g18 = np.array([[abs(ai - aj) if a == b else ai + aj + .6 for b in range(2) for aj in parc] for a in range(2) for ai in parc])
        cent = nz["cent_" + l]
        vals, part = [], []
        for t in range(len(cent)):
            E = cent[t][[23 * s + i for s in range(2) for i in near]]; De = rdm(E)
            vals.append(float(spearmanr(De[iu18], Dt[iu18])[0]))
            # partial Spearman controlling body distance
            from scipy.stats import rankdata
            x, y, z = rankdata(De[iu18]), rankdata(Dt[iu18]), rankdata(g18[iu18])
            rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
            part.append(float(np.corrcoef(rx, ry)[0, 1]))
        c["text_rdm"][l] = vals; c["text_rdm_partial"][l] = part
        # null for text-axis transfer: shuffle part labels among the 9 parts (wordings kept), 40 shuffles, final + step0
        rng = np.random.default_rng(0); nulls = []
        for _ in range(40):
            perm = rng.permutation(9); y = np.array([parc[perm[int(p)]] for p in tm[:, 2]])
            w = ridge_w(TX, y); M = cent[-1].astype(float)
            nulls.append(np.mean([spearmanr(M[23 * a:23 * a + 23] @ w, ARCS)[0] for a in range(2)]))
        c["text_null95"][l] = float(np.percentile(np.abs(nulls), 95))
S2 = {}
# named vs nameless convergence (layer 18 centroids)
pairs = {"same_seed_named_vs_nameless": [(("named", s), ("nameless", s)) for s in range(3)],
         "cross_seed_named_vs_nameless": [(("named", s), ("nameless", t)) for s in range(3) for t in range(3) if s != t],
         "cross_seed_named": [(("named", s), ("named", t)) for s in range(3) for t in range(s + 1, 3)],
         "cross_seed_nameless": [(("nameless", s), ("nameless", t)) for s in range(3) for t in range(s + 1, 3)]}
for l in ["inject", "18", "36"]:
    for nm, pl in pairs.items():
        pl = [p for p in pl if p[0] in runs and p[1] in runs]
        if not pl: continue
        rs, ck, raw, rw = [], [], [], []
        for a, b in pl:
            A, Bc = runs[a][1]["cent_" + l], runs[b][1]["cent_" + l]; n = min(len(A), len(Bc))
            rs.append([spearmanr(rdm(A[t])[iu], rdm(Bc[t])[iu])[0] for t in range(n)])
            rw.append([np.mean([spearmanr(rdm(A[t][23*h:23*h+23])[np.triu_indices(23,1)], rdm(Bc[t][23*h:23*h+23])[np.triu_indices(23,1)])[0] for h in range(2)]) for t in range(n)])
            ck.append([cka(A[t], Bc[t]) for t in range(n)])
            raw.append([np.corrcoef((A[t] - A[t].mean(0)).astype(float).ravel(), (Bc[t] - Bc[t].mean(0)).astype(float).ravel())[0, 1] for t in range(n)])
        n = min(map(len, rs))
        S2[f"{nm}_{l}"] = dict(rdm_within_r=np.mean([r[:n] for r in rw], 0).round(3).tolist(), rdm_r=np.mean([r[:n] for r in rs], 0).round(3).tolist(), cka=np.mean([r[:n] for r in ck], 0).round(3).tolist(),
                               same_coords_r=np.mean([r[:n] for r in raw], 0).round(3).tolist(), n_pairs=len(pl))
# raw sensor RSA for the static set (no model)
import sim3 as sim
T = 10; srng = np.random.default_rng(42); POS = [sim.rand_q(srng) for _ in range(8)]
def static_S(q, a, arc):
    A_ = sim.landmark_point(q, a, arc); env = sim.Env(q, np.random.default_rng(0)); segs = sim.seg_pts(q); P, Q = segs[3 * a + min(sim.seg_of(arc), 2)]
    nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm); env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=.1, p=.8, t_end=T - 1)
    for t in range(T - 1): env.step(12)
    return env.sense()
Mraw = np.stack([np.mean([static_S(q, a, arc) for q in POS], 0) for a in range(2) for arc in ARCS])
Draw = rdm(Mraw); Dsk = rdm(Mraw[:, 22:])
S2["raw_rsa_within"] = float(spearmanr(Draw[iu][win], geo[iu][win])[0]); S2["raw_skin_rsa_within"] = float(spearmanr(Dsk[iu][win], geo[iu][win])[0])
out = dict(runs={f"{a}_s{s}": v for (a, s), v in S.items()}, pairs=S2)
json.dump(out, open("$BODY_WORK/body_instill/form_summary.json", "w"), default=float)
# print
st = next(iter(S.values()))["step"]; print("steps", st)
def mean_curve(key, arm, sub=None):
    v = [S[(arm, s)][key] if sub is None else S[(arm, s)][key][sub] for s in range(3) if (arm, s) in S]
    n = min(map(len, v)); return np.mean([x[:n] for x in v], 0), np.std([x[:n] for x in v], 0)
for arm in ["named", "nameless"]:
    print("==", arm)
    for key in ["rsa_w_inject", "rsa_w_12", "rsa_w_18", "rsa_w_36", "arc_r2_18", "pose_inj", "pose_18", "x_18", "self_18", "self_verbal", "pos_mae", "task", "cl_reach", "tt_18", "tt_27", "ttrev_18", "ttside_18"]:
        m, sd = mean_curve(key, arm); print(f"{key:13s}", " ".join(f"{x:5.2f}" for x in m))
    for l in ["18", "27"]:
        m, _ = mean_curve("text_rdm", arm, l); print(f"txtRDM_{l:6s}", " ".join(f"{x:5.2f}" for x in m))
        m, _ = mean_curve("text_rdm_partial", arm, l); print(f"txtRDMp_{l:5s}", " ".join(f"{x:5.2f}" for x in m))
    print("text null95", {l: np.mean([S[(arm, s)]["text_null95"][l] for s in range(3) if (arm, s) in S]) for l in ["12", "18", "27", "36"]})
    for key in ["rsa_w_18", "arc_r2_18", "pose_18", "x_18", "self_18", "self_verbal", "task", "cl_reach"]:
        print("onset", key, [(onset(S[(arm, s)]["step"], S[(arm, s)][key], .5), onset(S[(arm, s)]["step"], S[(arm, s)][key], .9)) for s in range(3) if (arm, s) in S])
for k, v in S2.items(): print(k, v)
