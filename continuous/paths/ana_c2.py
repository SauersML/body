"""Path-specific residual: remove the step curve shared by all body paths (per agent), then test fwd/rev, spots."""
import sys, json, numpy as np; from scipy.stats import spearmanr
import spots as SP
OUT, SPOTDIR, LAYERS = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
D = json.load(open(f"{OUT}/items.json")); I = D["items"]; keep = D["keep"]; K = 16; steps = np.arange(1, K + 1)
SD = json.load(open(f"{SPOTDIR}/rows.json")); skeep = SD["keep"]
def sp(a, b): return float(spearmanr(a, b).correlation)
body = [n for n, it in enumerate(I) if it["kind"] == "body" and it["agent"] != "water_fast"]
out = {}
for rd in ["reply", "step"]:
    R = np.load(f"{OUT}/{rd}.npy", mmap_mode="r")
    for L in LAYERS:
        X = np.asarray(R[keep.index(L)], np.float32)[:, 1:]; X = X - X.mean(1, keepdims=True)
        Y = X.copy(); share = {}
        for ag in ["water", "ant", "feather", "finger"]:
            g = [n for n in body if I[n]["agent"] == ag]; share[ag] = X[g].mean(0)
            for n in g: Y[n] = X[n] - share[ag]
        tot = np.mean([np.linalg.norm(X[n], axis=1).mean() for n in body]); rem = np.mean([np.linalg.norm(Y[n], axis=1).mean() for n in body])
        # path-specific part: is it a monotone drift? fwd vs rev anti-diagonal?
        mono, fr = [], []
        for n in body:
            y = Y[n]; ch = y[-1] - y[0]; mono.append(sp(steps, y @ ch))
        for n in body:
            it = I[n]
            if it["dir"] != "fwd": continue
            m = [q for q in body if I[q]["path"] == it["path"] and I[q]["agent"] == it["agent"] and I[q]["dir"] == "rev"][0]
            A = Y[n] / np.linalg.norm(Y[n], axis=1, keepdims=True); B = Y[m] / np.linalg.norm(Y[m], axis=1, keepdims=True); M = A @ B.T
            fr.append([np.trace(M) / K, np.trace(M[:, ::-1]) / K, sp(steps, M.argmax(1))])
        fr = np.array(fr)
        # same path, different agents: does path-specific trajectory agree across agents (diagonal)?
        ag_c = []
        for n in body:
            it = I[n]
            if it["agent"] != "water": continue
            for q in body:
                if I[q]["path"] == it["path"] and I[q]["dir"] == it["dir"] and I[q]["agent"] != "water":
                    A = Y[n] / np.linalg.norm(Y[n], axis=1, keepdims=True); B = Y[q] / np.linalg.norm(Y[q], axis=1, keepdims=True); ag_c.append(np.trace(A @ B.T) / K)
        r = dict(frac_path_specific=rem / tot, monotone=np.mean(mono), fr_diag=fr[:, 0].mean(), fr_anti=fr[:, 1].mean(), fr_argmax_rho=np.nanmean(fr[:, 2]), cross_agent_diag=np.mean(ag_c))
        # spots: path-specific trajectory vs spot sequence: regress spot states (centered) on arc t -> direction; project trajectory
        if L in skeep:
            SA = np.load(f"{SPOTDIR}/{'reply' if rd == 'reply' else 'final'}.npy", mmap_mode="r"); sl = skeep.index(L)
            spr = {}
            for path in ["arm_palm_left", "arm_palm_right", "arm_back_left", "arm_back_right", "leg_left", "leg_right", "head", "spine"]:
                ids, t = SP.path_spots(path); Sx = np.stack([np.asarray(SA[sl, i * 26:(i + 1) * 26], np.float32).mean(0) for i in ids]); Sx -= Sx.mean(0)
                w = Sx.T @ (t - t.mean())  # spot axis: fingertip->shoulder direction among landmark spots
                vals = []
                for n in body:
                    if I[n]["path"] == path:
                        vals.append(sp(steps, Y[n] @ w) * (1 if I[n]["dir"] == "fwd" else 1j))
                v = np.array(vals); spr[path + "_fwd"] = float(v[np.isreal(v)].real.mean()); spr[path + "_rev"] = float(v[~np.isreal(v)].imag.mean())
            r["traj_on_spot_axis"] = spr
            r["traj_on_spot_axis_mean"] = float(np.mean(list(spr.values())))
        out[f"{rd}|{L}"] = r; print(rd, L, {k: (round(v, 2) if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in r.items()}, flush=True)
json.dump(out, open(f"{OUT}/ana2.json", "w"), indent=1, default=float)
