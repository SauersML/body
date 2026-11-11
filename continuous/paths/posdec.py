"""Position decoder: label = place along path (fwd i/K, rev 1-i/K), trained on all arm trajectories (both directions); tested on other paths."""
import sys, json, numpy as np; from scipy.stats import spearmanr
K = 16; st = np.arange(1, K + 1)
def load(o, rd, L):
    D = json.load(open(f"{o}/items.json")); I = D["items"]; X = np.asarray(np.load(f"{o}/{rd}.npy", mmap_mode="r")[D["keep"].index(L)], np.float32)[:, 1:]
    sel = lambda **kw: [n for n, it in enumerate(I) if all(it[k] == v for k, v in kw.items())]
    ctl = {ag: X[sel(kind="nonbody", agent=ag)].mean(0) for ag in ["water", "ant", "feather", "finger"]}; ctl = {k: v - v.mean(0) for k, v in ctl.items()}; ctl["water_fast"] = ctl["water"]
    Xs = np.stack([X[n] - ctl[it["agent"]] if it["kind"] != "static" else X[n] for n, it in enumerate(I)]); Xs -= Xs.mean(1, keepdims=True); return I, Xs
for o, oh, Ls in [("out8b", "out8b_hf", [22, 25, 27]), ("out32", "out32_hf", [44, 48, 53])]:
    for rd in ["reply", "step"]:
        for L in Ls:
            I, X = load(o, rd, L); Ih, Xh = load(oh, rd, L)
            body = [(I, X, n) for n, it in enumerate(I) if it["kind"] == "body" and it["agent"] != "water_fast"] + [(Ih, Xh, n) for n, it in enumerate(Ih) if it["kind"] == "body"]
            tr = [b for b in body if b[0] is I and b[0][b[2]]["path"].startswith("arm")]; te = [b for b in body if not (b[0] is I and b[0][b[2]]["path"].startswith("arm"))]
            lab = lambda b: st / K if b[0][b[2]]["dir"] == "fwd" else 1 - st / K
            A = np.concatenate([b[1][b[2]] for b in tr]); y = np.concatenate([lab(b) for b in tr]); Ac = A - A.mean(0)
            lam = 0.1 * np.trace(Ac.T @ Ac) / len(Ac); w = Ac.T @ np.linalg.solve(Ac @ Ac.T + lam * np.eye(len(Ac)), y - y.mean())
            rf = np.mean([spearmanr(st, b[1][b[2]] @ w).correlation for b in te if b[0][b[2]]["dir"] == "fwd"]); rr = np.mean([spearmanr(1 - st / K, b[1][b[2]] @ w).correlation for b in te if b[0][b[2]]["dir"] == "rev"])
            ins = np.mean([spearmanr(lab(b), b[1][b[2]] @ w).correlation for b in tr])
            print(o, rd, L, "train(arms) rho", round(ins, 2), "| test place rho fwd", round(rf, 2), "rev", round(rr, 2), flush=True)
