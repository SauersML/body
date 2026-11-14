"""ana.py DIR... -> summary.json. Threshold fits, direct report, state probe."""
import sys, json, numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
rng = np.random.default_rng(0)
def sig_fit(h, p):
    f = lambda t: 1 / (1 + np.exp(-(t[0] - h) / abs(t[1]))) * (1 - 2 * t[2]) + t[2] - p   # threshold, width, lapse
    r = least_squares(f, [1.7, .1, .05], bounds=([0.3, .01, 0], [4, 2, .45])); return r.x
def thresh(rows, invert=False):
    """rows: (h, p_above, mass). returns threshold + bootstrap interval over heights' items, and slope sanity."""
    a = np.array(rows); h, p = a[:, 0], a[:, 1]
    hs = np.unique(h); pm = np.array([p[h == x].mean() for x in hs])
    t = sig_fit(hs, pm); bs = []
    idx = {x: np.where(h == x)[0] for x in hs}
    for _ in range(200):
        pb = np.array([p[rng.choice(idx[x], len(idx[x]))].mean() for x in hs]); bs.append(sig_fit(hs, pb)[0])
    rngp = pm.max() - pm.min(); ok = rngp > .3 and pm[0] > pm[-1]
    return dict(H=float(t[0]), lo=float(np.percentile(bs, 5)), hi=float(np.percentile(bs, 95)), width=float(abs(t[1])),
                range=float(rngp), ok=bool(ok), curve=[(float(x), float(y)) for x, y in zip(hs, pm)])
G = np.arange(100, 231)
out = {}
for D in (sys.argv[1:] if __name__ == "__main__" else []):
    R = json.load(open(f"{D}/res.json")); name = R["model"].split("/")[-1]; o = {"beh": {}}
    for c, r in R["beh"].items():
        e = {"wall": thresh(r["wall"]), "fence": thresh(r["fence"]), "obj": thresh(r["obj"])}
        d = np.array(r["direct"]); p = np.exp(d - d.max()); p /= p.sum()
        e["direct"] = dict(mean=float((p * G).sum()), mode=int(G[p.argmax()]), q=[int(G[np.searchsorted(p.cumsum(), x)]) for x in (.1, .5, .9)],
                           digit=r["digitmass"][0], text=r.get("direct_text"))
        o["beh"][c] = e
    X = np.load(f"{D}/states.npz")["X"].astype(np.float32); items = R["state_items"]; nT = len(R["tpl"])
    ent = R["ent"]; num = R["num"]; selfx = R["selfx"]; nE, nN = len(ent), len(num)
    Xe = X[:nE * nT]; ye = np.log(np.repeat([h for _, h in ent], nT)); grp = np.repeat(np.arange(nE), nT)
    Xn = X[nE * nT:(nE + nN) * nT]; yn = np.repeat([h for _, h in num], nT); Xs = X[(nE + nN) * nT:]
    best = None
    for k, L in enumerate(R["layers"]):
        A = Xe[:, k]; mu, sd = A.mean(0), A.std(0) + 1e-3; Z = (A - mu) / sd
        pred = np.zeros(len(ye))
        for g in range(nE):   # leave one entity out
            tr = grp != g; m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Z[tr], ye[tr]); pred[~tr] = m.predict(Z[~tr])
        r2 = 1 - ((pred - ye) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
        m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Z, ye)
        pn = np.exp(m.predict((Xn[:, k] - mu) / sd)); rn = np.corrcoef(pn, yn)[0, 1]
        ps = np.exp(m.predict((Xs[:, k] - mu) / sd)).reshape(len(selfx), nT)
        rec = dict(layer=L, r2=float(r2), num_r=float(rn), num_pred=[float(pn.reshape(nN, nT)[i].mean()) for i in range(nN)],
                   self={f"{x}|{c}": [float(ps[i].mean()), float(ps[i].min()), float(ps[i].max())] for i, (x, c) in enumerate(selfx)})
        o.setdefault("probe", []).append(rec)
        if best is None or r2 > best["r2"]: best = rec
    o["probe_best"] = best; out[name] = o
    b = o["beh"]
    print(f"{name:16s} wall {b['self']['wall']['H']:.2f}[{b['self']['wall']['lo']:.2f},{b['self']['wall']['hi']:.2f}] ok={b['self']['wall']['ok']} "
          f"obj {b['self']['obj']['H']:.2f} fence {b['self']['fence']['H']:.2f} ok={b['self']['fence']['ok']} direct {b['self']['direct']['mean']:.0f} "
          f"| person {b['person']['wall']['H']:.2f} w {b['woman']['wall']['H']:.2f} m {b['man']['wall']['H']:.2f} | s150 {b['self150']['wall']['H']:.2f} s200 {b['self200']['wall']['H']:.2f} "
          f"| probe L{best['layer']} r2 {best['r2']:.2f} numr {best['num_r']:.2f} I {best['self']['I|self'][0]:.2f} She {best['self']['She|self'][0]:.2f} I150 {best['self']['I|self150'][0]:.2f} I200 {best['self']['I|self200'][0]:.2f}")
if __name__ == "__main__": json.dump(out, open("summary.json", "w"), indent=1)
