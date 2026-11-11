"""Final analysis: one model, readouts x layers. Spot states from reduced npz (spot means over 26 wordings, A=550 landmark spots, W=125 word-free probes)."""
import sys, json, numpy as np; from scipy.stats import spearmanr
import spots as SP; sys.path.insert(0, "../hires2"); import stim as S
OUT, NPZ, LAYERS = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
D = json.load(open(f"{OUT}/items.json")); I = D["items"]; keep = D["keep"]; K = 16; st = np.arange(1, K + 1); Z_ = np.load(NPZ)
def sp(a, b): return float(spearmanr(a, b).correlation)
def nrm(A): return A / np.linalg.norm(A, axis=-1, keepdims=True)
body = [n for n, it in enumerate(I) if it["kind"] == "body" and it["agent"] != "water_fast"]
def sel(**kw): return [n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())]
PR = S.probes(); armx = lambda i: (100 - abs(S.LOCS[i]["xyz"][0])) / 80
wf_arm = [(j, armx(p["target"])) for j, p in enumerate(PR) if p["kind"] == "wf" and S.LOCS[p["target"]]["region"][:3] in ("arm", "han")]
out = {}
for rd in ["reply", "step"]:
    R = np.load(f"{OUT}/{rd}.npy", mmap_mode="r")
    for L in LAYERS:
        X = np.asarray(R[keep.index(L)], np.float32)[:, 1:]
        ctl = {ag: X[sel(kind="nonbody", agent=ag)].mean(0) for ag in ["water", "ant", "feather", "finger"]}
        ctl = {k: v - v.mean(0) for k, v in ctl.items()}; ctl["water_fast"] = ctl["water"]
        Xs = np.stack([X[n] - ctl[it["agent"]] if it["kind"] != "static" else X[n] for n, it in enumerate(I)]); Xs -= Xs.mean(1, keepdims=True)
        mot = lambda Y, ns: float(np.mean([np.linalg.norm(Y[n], axis=1).mean() for n in ns]))
        nbh = []
        for n in sel(kind="nonbody"):
            o = [m for m in sel(kind="nonbody", agent=I[n]["agent"]) if I[m]["path"] != I[n]["path"]]; c = X[o].mean(0); y = X[n] - (c - c.mean(0)); nbh.append(np.linalg.norm(y - y.mean(0), axis=1).mean())
        Xc = X - X.mean(1, keepdims=True)
        r = dict(raw_body=mot(Xc, body), raw_nonbody=mot(Xc, sel(kind="nonbody")), raw_static=mot(Xc, sel(kind="static")), resid_body=mot(Xs, body), resid_nonbody_heldout=float(np.mean(nbh)))
        g = []
        for n in body:
            y = Xs[n]; d = np.diff(y, axis=0); s = np.linalg.norm(d, axis=1); ch = y[-1] - y[0]
            g.append([s.std() / s.mean(), np.linalg.norm(ch) / s.sum(), sp(st, y @ ch), np.mean(np.sum(nrm(d[:-1]) * nrm(d[1:]), 1))])
        g = np.array(g).mean(0); r.update(step_cv=g[0], straightness=g[1], monotone=g[2], turn_cos=g[3])
        # shared-body-curve removal -> path-specific part
        Y = Xs.copy(); share = {}
        for ag in ["water", "ant", "feather", "finger"]:
            gg = [n for n in body if I[n]["agent"] == ag]; share[ag] = Xs[gg].mean(0)
            for n in gg: Y[n] = Xs[n] - share[ag]
        r["frac_path_specific"] = mot(Y, body) / mot(Xs, body)
        for nm, V in [("full", Xs), ("pathspec", Y)]:
            fr, lr = [], []
            for n in body:
                it = I[n]
                if it["dir"] != "fwd": continue
                m = [q for q in body if I[q]["path"] == it["path"] and I[q]["agent"] == it["agent"] and I[q]["dir"] == "rev"][0]
                M = nrm(V[n]) @ nrm(V[m]).T; fr.append([np.trace(M) / K, np.trace(M[:, ::-1]) / K])
                if it["path"].endswith("_left"):
                    m2 = [q for q in body if I[q]["path"] == it["path"].replace("left", "right") and I[q]["agent"] == it["agent"] and I[q]["dir"] == "fwd"][0]
                    lr.append(np.trace(nrm(V[n]) @ nrm(V[m2]).T) / K)
            fr = np.array(fr); r[f"fr_same_step_cos_{nm}"] = fr[:, 0].mean(); r[f"fr_same_place_cos_{nm}"] = fr[:, 1].mean(); r[f"lr_cos_{nm}"] = float(np.mean(lr))
        # progress probe trained on water fwd arms
        tr = [n for n in body if I[n]["agent"] == "water" and I[n]["dir"] == "fwd" and I[n]["path"].startswith("arm")]
        A = Xs[tr].reshape(-1, X.shape[2]); yv = np.tile(st / K, len(tr)); mu = A.mean(0); Ac = A - mu
        lam = 0.1 * np.trace(Ac.T @ Ac) / len(Ac); w = Ac.T @ np.linalg.solve(Ac @ Ac.T + lam * np.eye(len(Ac)), yv - yv.mean())
        pr = lambda V: V @ w
        T = {}
        for ag in ["ant", "feather", "finger", "water_fast"]: T["arm_" + ag] = np.mean([sp(st, pr(Xs[n])) for n in sel(kind="body", agent=ag, dir="fwd") if I[n]["path"].startswith("arm")])
        T["arm_reverse"] = np.mean([sp(st, pr(Xs[n])) for n in body if I[n]["dir"] == "rev" and I[n]["path"].startswith("arm")])
        for p in ["leg", "head", "spine", "chest"]: T[p] = np.mean([sp(st, pr(Xs[n])) for n in body if I[n]["path"].startswith(p)])
        T["nonbody_rawcentered"] = np.mean([sp(st, pr(Xc[n])) for n in sel(kind="nonbody")])
        T["static_body"] = np.mean([sp(st, pr(Xc[n])) for n in sel(kind="static")])
        key = f"{'reply' if rd == 'reply' else 'final'}|{L}|all|A"
        if key in Z_.files:
            SA = Z_[key].astype(np.float32); WA = Z_[key[:-1] + "W"].astype(np.float32)
            sps, axis_f, axis_r = {}, {}, {}
            for path in ["arm_palm_left", "arm_palm_right", "arm_back_left", "arm_back_right", "leg_left", "leg_right", "head", "spine"]:
                ids, t = SP.path_spots(path); Sx = SA[ids] - SA[ids].mean(0); sps[path] = sp(t, pr(Sx))
                ax = Sx.T @ (t - t.mean())
                axis_f[path] = np.mean([sp(st, Y[n] @ ax) for n in body if I[n]["path"] == path and I[n]["dir"] == "fwd"])
                axis_r[path] = np.mean([sp(st, Y[n] @ ax) for n in body if I[n]["path"] == path and I[n]["dir"] == "rev"])
            T["landmark_spots_arms"] = np.mean([v for k, v in sps.items() if k.startswith("arm")]); T["landmark_spots_all"] = np.mean(list(sps.values()))
            j = [a for a, _ in wf_arm]; tt = [b for _, b in wf_arm]; Wx = WA[j] - WA[j].mean(0); T["wordfree_arm_probes"] = sp(tt, pr(Wx)); T["n_wordfree"] = len(j)
            r["spot_axis_fwd"] = float(np.mean(list(axis_f.values()))); r["spot_axis_rev"] = float(np.mean(list(axis_r.values())))
            r["spot_axis_detail"] = {k: [round(axis_f[k], 2), round(axis_r[k], 2)] for k in axis_f}
        r["probe"] = {k: float(v) for k, v in T.items()}
        out[f"{rd}|{L}"] = r
        print(rd, L, json.dumps(r, default=lambda x: round(float(x), 3)), flush=True)
json.dump(out, open(f"{OUT}/anaf.json", "w"), indent=1, default=float)
