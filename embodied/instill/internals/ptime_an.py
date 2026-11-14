import json, numpy as np, sys
R = json.load(open(sys.argv[1])); T = 10
def zc(x): x = np.asarray(x, float); return x - x.mean(1, keepdims=True)  # within-episode centring
def r(a, b): a, b = a.ravel(), b.ravel(); return float(np.corrcoef(a, b)[0, 1])
def boot(f, n, B=1000):
    rg = np.random.default_rng(0); v = [f(rg.integers(0, n, n)) for _ in range(B)]; return np.percentile(v, [2.5, 97.5]).round(2).tolist()
for tag, D in R.items():
    h = D['held']; pj = np.array(h['proj']); p = np.array(h['p']); S = np.array(h['S']); el = S[:, :, 22:38].sum(-1)
    rp = r(pj, p); rw = r(zc(pj), zc(p)); re_ = r(pj, el)
    ci = boot(lambda i: r(pj[i], p[i]), len(p))
    # contact-only steps: grade with pressure .3-1
    m = p > 0; rc = float(np.corrcoef(pj[m], p[m])[0, 1])
    n = D['normal']; Pj = np.array(n['proj']); Pn = np.array(n['p']); A = n['A']; BL = np.array([b + [np.nan] for b in n['blocked']], float)  # blocked[t]: action at t pushed into wall
    Pz, Jz = zc(Pn), zc(Pj)
    def lagc(X, Y, k):  # corr(proj_t, Y_{t+k}) within episode
        xs, ys = [], []
        for t in range(T):
            if 0 <= t + k < T and not np.isnan(Y[:, t + k]).any(): xs.append(X[:, t]); ys.append(Y[:, t + k])
        return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]) if xs else np.nan
    BLz = BL[:, :T-1] - np.nanmean(BL[:, :T-1], 1, keepdims=True); BLf = np.concatenate([BLz, np.full((len(BL), 1), np.nan)], 1)
    lp = {k: round(lagc(Jz, Pz, k), 2) for k in (-2, -1, 0, 1, 2)}; lb = {k: round(lagc(Jz, BLf, k), 2) for k in (-2, -1, 0, 1, 2)}
    # regression proj_t ~ p_t + press_t (upcoming) + press_{t-1}, steps 1..8, within-episode centred, standardized
    rows = [(Jz[:, t], Pz[:, t], BLf[:, t], BLf[:, t - 1]) for t in range(1, T - 1)]
    y, x1, x2, x3 = [np.concatenate(c) for c in zip(*rows)]; X = np.stack([x1, x2, x3], 1); X = (X - X.mean(0)) / X.std(0); yy = (y - y.mean()) / y.std()
    beta = np.linalg.lstsq(np.c_[X, np.ones(len(yy))], yy, rcond=None)[0][:3].round(2).tolist()
    print(f"{tag}: HELD r(proj,p)={rp:.2f} {ci} within-ep {rw:.2f} contact-only {rc:.2f} r(proj,electrodes)={re_:.2f} | NORMAL lag corr with pressure p_(t+k): {lp} ; with blocked press at t+k: {lb} ; beta[p_t, press_t(next), press_t-1]={beta}")
