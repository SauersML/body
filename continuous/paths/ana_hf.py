"""Hand/face paths: fwd vs rev (same step vs same place), arm-trained step probe transfer, landmark-spot axis, probe on landmark spots."""
import sys, json, numpy as np; from scipy.stats import spearmanr
import spots as SP
OUT, OUTHF, NPZ, LAYERS = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]); K = 16; st = np.arange(1, K + 1); Z_ = np.load(NPZ)
def sp(a, b): return float(spearmanr(a, b).correlation)
def nrm(A): return A / np.linalg.norm(A, axis=-1, keepdims=True)
def load(o, rd, L):
    D = json.load(open(f"{o}/items.json")); I = D["items"]; X = np.asarray(np.load(f"{o}/{rd}.npy", mmap_mode="r")[D["keep"].index(L)], np.float32)[:, 1:]
    sel = lambda **kw: [n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())]
    ctl = {ag: X[sel(kind="nonbody", agent=ag)].mean(0) for ag in ["water", "ant", "feather", "finger"]}; ctl = {k: v - v.mean(0) for k, v in ctl.items()}; ctl["water_fast"] = ctl["water"]
    Xs = np.stack([X[n] - ctl[it["agent"]] if it["kind"] != "static" else X[n] for n, it in enumerate(I)]); Xs -= Xs.mean(1, keepdims=True)
    return I, Xs
res = {}
for rd in ["reply", "step"]:
    for L in LAYERS:
        I0, X0 = load(OUT, rd, L); I, Xs = load(OUTHF, rd, L)
        tr = [n for n, it in enumerate(I0) if it["kind"] == "body" and it["agent"] == "water" and it["dir"] == "fwd" and it["path"].startswith("arm")]
        A = X0[tr].reshape(-1, X0.shape[2]); yv = np.tile(st / K, len(tr)); Ac = A - A.mean(0); lam = 0.1 * np.trace(Ac.T @ Ac) / len(Ac)
        w = Ac.T @ np.linalg.solve(Ac @ Ac.T + lam * np.eye(len(Ac)), yv - yv.mean())
        body = [n for n, it in enumerate(I) if it["kind"] == "body"]
        Y = Xs.copy()
        for ag in ["water", "ant", "feather", "finger"]:
            g = [n for n in body if I[n]["agent"] == ag]; m = Xs[g].mean(0)
            for n in g: Y[n] = Xs[n] - m
        key = f"{'reply' if rd == 'reply' else 'final'}|{L}|all|A"; SA = Z_[key].astype(np.float32) if key in Z_.files else None
        r = {}
        for path in SP.HFL:
            ns = [n for n in body if I[n]["path"] == path]; fr = []
            for n in ns:
                if I[n]["dir"] != "fwd": continue
                m = [q for q in ns if I[q]["agent"] == I[n]["agent"] and I[q]["dir"] == "rev"][0]; M = nrm(Xs[n]) @ nrm(Xs[m]).T; fr.append([np.trace(M) / K, np.trace(M[:, ::-1]) / K])
            fr = np.array(fr).mean(0); d = dict(same_step=fr[0], same_place=fr[1], probe=np.mean([sp(st, Xs[n] @ w) for n in ns]))
            if SA is not None:
                ids, t, _ = SP.path_spots_hf(path); Sx = SA[ids] - SA[ids].mean(0); ax = Sx.T @ (t - t.mean())
                d["probe_on_spots"] = sp(t, Sx @ w)
                d["axis_fwd"] = np.mean([sp(st, Y[n] @ ax) for n in ns if I[n]["dir"] == "fwd"]); d["axis_rev"] = np.mean([sp(st, Y[n] @ ax) for n in ns if I[n]["dir"] == "rev"])
            r[path] = {k: round(float(v), 3) for k, v in d.items()}
        res[f"{rd}|{L}"] = r; print(rd, L, json.dumps(r), flush=True)
json.dump(res, open(f"{OUTHF}/ana_hf.json", "w"), indent=1)
