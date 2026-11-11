import sys, json, numpy as np; from scipy.stats import spearmanr
import spots as SP
OUT = sys.argv[1]; SPOTDIR = sys.argv[2]; LAYERS = json.loads(sys.argv[3]); RES = {}
D = json.load(open(f"{OUT}/items.json")); I = D["items"]; keep = D["keep"]; K = 16
Rr = np.load(f"{OUT}/reply.npy", mmap_mode="r"); Rs = np.load(f"{OUT}/step.npy", mmap_mode="r")
SD = json.load(open(f"{SPOTDIR}/rows.json")); srows = SD["rows"]; skeep = SD["keep"]
def idx(**kw): return [n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())]
def sp(a, b): return float(spearmanr(a, b).correlation)
def ridge(X, y, lam):
    mu = X.mean(0); Xc = X - mu; w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y.mean())); return lambda Z: (Z - mu) @ w + y.mean()
steps = np.arange(1, K + 1)
for rd, R in [("reply", Rr), ("step", Rs)]:
    for L in LAYERS:
        li = keep.index(L); X = np.asarray(R[li], np.float32)[:, 1:]  # items x 16 x d (steps 1..16)
        # agent-matched non-body step curve
        Xs = X.copy(); ctl = {}
        for ag in ["water", "ant", "feather", "finger"]:
            c = X[idx(kind="nonbody", agent=ag)].mean(0); ctl[ag] = c - c.mean(0)
        for n, it in enumerate(I):
            ag = "water" if it["agent"] == "water_fast" else it["agent"]
            if it["kind"] != "static": Xs[n] = X[n] - ctl[ag]
        # held-out control residual: nonbody item minus control curve built without its object
        def resid_motion(n, Y):
            y = Y[n] - Y[n].mean(0); return float(np.linalg.norm(y, axis=1).mean())
        nb_res = []
        for n in idx(kind="nonbody"):
            it = I[n]; o = [m for m in idx(kind="nonbody", agent=it["agent"]) if I[m]["path"] != it["path"]]
            c = X[o].mean(0); y = X[n] - (c - c.mean(0)); y = y - y.mean(0); nb_res.append(np.linalg.norm(y, axis=1).mean())
        body = idx(kind="body"); raw_body = np.mean([resid_motion(n, X) for n in body]); raw_nb = np.mean([resid_motion(n, X) for n in idx(kind="nonbody")])
        res_body = np.mean([resid_motion(n, Xs) for n in body]); res_static = np.mean([resid_motion(n, X) for n in idx(kind="static")])
        # geometry per body trajectory (after control subtraction)
        geo = []
        for n in body:
            y = Xs[n]; dif = np.diff(y, axis=0); st = np.linalg.norm(dif, axis=1)
            chord = y[-1] - y[0]; proj = (y - y[0]) @ chord / np.linalg.norm(chord)
            cosv = [dif[j] @ dif[j + 1] / st[j] / st[j + 1] for j in range(len(st) - 1)]
            geo.append([st.std() / st.mean(), np.linalg.norm(chord) / st.sum(), sp(steps, proj), np.mean(cosv)])
        geo = np.array(geo)
        r = dict(motion_raw_body=raw_body, motion_raw_nonbody=raw_nb, motion_resid_body=res_body, motion_resid_nonbody_heldout=float(np.mean(nb_res)), motion_static=res_static,
                 step_cv=geo[:, 0].mean(), straightness=geo[:, 1].mean(), monotonic=geo[:, 2].mean(), turn_cos=geo[:, 3].mean())
        # fwd vs rev; left vs right (centered per trajectory)
        def C(a, b):
            A = Xs[a] - Xs[a].mean(0); B = Xs[b] - Xs[b].mean(0); A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True); return A @ B.T
        fr, lr, fra = [], [], []
        for it_n in body:
            it = I[it_n]
            if it["dir"] == "fwd" and it["agent"] != "water_fast":
                m = [q for q in body if I[q]["path"] == it["path"] and I[q]["agent"] == it["agent"] and I[q]["dir"] == "rev"][0]
                M = C(it_n, m); fr.append(sp(steps, M.argmax(1))); fra.append([np.trace(M) / K, np.trace(M[:, ::-1]) / K])
                if it["path"].endswith("_left"):
                    m2 = [q for q in body if I[q]["path"] == it["path"].replace("left", "right") and I[q]["agent"] == it["agent"] and I[q]["dir"] == "fwd"][0]
                    M2 = C(it_n, m2); lr.append([sp(steps, M2.argmax(1)), np.trace(M2) / K])
        fra = np.array(fra); lr = np.array(lr)
        r.update(fwdrev_argmax_rho=float(np.mean(fr)), fwdrev_diag_cos=float(fra[:, 0].mean()), fwdrev_antidiag_cos=float(fra[:, 1].mean()), lr_argmax_rho=float(lr[:, 0].mean()), lr_diag_cos=float(lr[:, 1].mean()))
        # progress probe: train on water fwd arms, test elsewhere (states centered per trajectory)
        Z = Xs - Xs.mean(1, keepdims=True); y = np.tile(steps / K, 1)
        tr = [n for n in body if I[n]["agent"] == "water" and I[n]["dir"] == "fwd" and I[n]["path"].startswith("arm")]
        pr = ridge(Z[tr].reshape(-1, Z.shape[2]), np.tile(steps / K, len(tr)), lam=float(np.trace(np.cov(Z[tr].reshape(-1, Z.shape[2]).T)) * 0.1))
        def tst(ns, sign=1): return float(np.mean([sp(steps, sign * pr(Z[n])) for n in ns])) if ns else None
        grp = {}
        for ag in ["ant", "feather", "finger", "water_fast"]: grp["arm_fwd_" + ag] = tst([n for n in body if I[n]["agent"] == ag and I[n]["dir"] == "fwd" and I[n]["path"].startswith("arm")])
        grp["arm_rev_all"] = tst([n for n in body if I[n]["dir"] == "rev" and I[n]["path"].startswith("arm")])
        for p in ["leg", "head", "spine", "chest"]: grp[p + "_fwd"] = tst([n for n in body if I[n]["dir"] == "fwd" and I[n]["path"].startswith(p)])
        grp["nonbody_fwd"] = tst([n for n in idx(kind="nonbody", dir="fwd")])
        grp["nonbody_raw_fwd"] = float(np.mean([sp(steps, pr(X[n] - X[n].mean(0))) for n in idx(kind="nonbody", dir="fwd")]))
        r["probe"] = grp
        # spots (same readout family: reply<->reply, step<->final), mean over 26 wordings, landmark words
        if L in skeep:
            sl = skeep.index(L); SA = np.load(f"{SPOTDIR}/{'reply' if rd == 'reply' else 'final'}.npy", mmap_mode="r")
            def spot_mean(li_): return np.asarray(SA[sl, li_ * 26:(li_ + 1) * 26], np.float32).mean(0)
            spr, nnr, nnr2 = {}, {}, {}
            for path in ["arm_palm_left", "arm_palm_right", "arm_back_left", "arm_back_right", "leg_left", "leg_right", "head", "spine"]:
                ids, t = SP.path_spots(path); Sx = np.stack([spot_mean(i) for i in ids]); Sx -= Sx.mean(0)
                p_ = pr(Sx); spr[path] = sp(t, p_)
                trs = [n for n in body if I[n]["path"] == path and I[n]["dir"] == "fwd" and I[n]["agent"] != "water_fast"]
                a, b = [], []
                for n in trs:
                    A = Z[n] / np.linalg.norm(Z[n], axis=1, keepdims=True); B = Sx / np.linalg.norm(Sx, axis=1, keepdims=True); M = A @ B.T
                    a.append(sp(steps, t[M.argmax(1)])); b.append(sp(t, M.argmax(0)))
                nnr[path] = float(np.mean(a)); nnr2[path] = float(np.mean(b))
            r["probe_on_spots"] = spr; r["nn_step_to_spot_rho"] = nnr; r["nn_spot_to_step_rho"] = nnr2
        RES[f"{rd}|{L}"] = r; print(rd, L, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: (round(b, 2) if b is not None else None) for a, b in v.items()}) for k, v in r.items()}), flush=True)
json.dump(RES, open(f"{OUT}/ana.json", "w"), indent=1)
