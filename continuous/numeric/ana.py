"""Analysis: expected numeric self-report vs truth; walk tracking; intensity; per-layer probes; steering curves."""
import sys, json, numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
D = sys.argv[1]; res = {}
def L(name): z = np.load(f"{D}/{name}.npz"); j = json.load(open(f"{D}/{name}.json")); return j["items"], z, j
def dist(P1, P2, mx):
    """number distribution 0..mx (mx 10 or 39) from first-digit and second-digit probs (branches 1,2,3)."""
    n = len(P1); p = np.zeros((n, 40))
    for d in range(10):
        if 1 <= d <= 3: p[:, d] = P1[:, d] * (1 - P2[:, d - 1].sum(1)); p[:, 10 * d:10 * d + 10] += P1[:, d:d + 1] * P2[:, d - 1]
        else: p[:, d] = P1[:, d]
    p = p[:, :mx + 1]; mass = p.sum(1); p = p / p.sum(1, keepdims=True); x = np.arange(mx + 1)
    ev = p @ x; sd = np.sqrt(np.maximum(p @ x ** 2 - ev ** 2, 0)); return ev, sd, mass, p
def stats(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float); r = pearsonr(t, y)[0]; rho = spearmanr(t, y)[0]; sl, ic = np.polyfit(t, y, 1)
    rng = np.random.default_rng(0); sh = np.mean([abs(pearsonr(rng.permutation(t), y)[0]) for _ in range(200)])
    return dict(r=round(r, 3), rho=round(rho, 3), slope=round(sl, 3), icpt=round(ic, 2), mae=round(float(np.mean(abs(y - t))), 2), shuf_absr=round(sh, 3), n=len(t))

# 1. static touches
it, z, j = L("pos"); P1, P2 = z["P1"], z["P2"]
q = np.array([x["q"] for x in it]); kind = np.array([x["kind"] for x in it])
ev10, sd10, m10, _ = dist(P1, P2, 10); evcm, sdcm, mcm, _ = dist(P1, P2, 39)
ev = np.where(q == "fa_cm", evcm, ev10); sd = np.where(q == "fa_cm", sdcm, sd10); mass = np.where(q == "fa_cm", mcm, m10)
out1 = {}
for k in ("lm", "nl", "num", "none", "ruler"):
    for qq, tk in (("fa_scale", "s"), ("fa_cm", "cm"), ("ruler", "s")):
        m = (kind == k) & (q == qq)
        if m.sum() < 3 or k == "none": 
            if m.sum(): out1[f"{k}_{qq}"] = dict(ev=ev[m].round(2).tolist(), sd=sd[m].round(2).tolist())
            continue
        t = [it[i][tk] for i in np.where(m)[0]]; out1[f"{k}_{qq}"] = dict(**stats(t, ev[m]), mean_sd=round(float(sd[m].mean()), 2), mass=round(float(mass[m].mean()), 3))
res["static"] = out1
res["static_rows"] = [dict(stim=x["stim"], q=x["q"], true=x["s"] if x["q"] != "fa_cm" else x["cm"], ev=round(float(ev[i]), 2), sd=round(float(sd[i]), 2), top=j["top"][i]) for i, x in enumerate(it)]
# per-layer probe on stimulus-end states (lm+nl, scale question rows; same states for both questions), leave-one-location-out ridge
def cvprobe(H, t, groups, lam=None):
    from sklearn.linear_model import RidgeCV
    pred = np.zeros(len(t))
    for g in np.unique(groups):
        tr = groups != g; mu = H[tr].mean(0); sdv = H[tr].std(0) + 1e-3
        m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit((H[tr] - mu) / sdv, t[tr]); pred[~tr] = m.predict((H[~tr] - mu) / sdv)
    return pred
keep = j["keep"]; Hs = z["Hs"].astype(np.float32); Ha = z["Ha"].astype(np.float32)
m = np.isin(kind, ["lm", "nl"]) & (q == "fa_scale"); t = np.array([it[i]["s"] for i in np.where(m)[0]]); import stim as S0
def locof(s): return next(s[len(a) + 1:] for a in S0.AG if s.startswith(a))
grp = np.array([locof(it[i]["stim"]) for i in np.where(m)[0]])
prob = []
for li, Ly in enumerate(keep):
    p1 = cvprobe(Hs[li][m], t, grp); p2 = cvprobe(Ha[li][m], t, grp)
    prob.append(dict(layer=Ly, stim_r=round(pearsonr(t, p1)[0], 3), ans_r=round(pearsonr(t, p2)[0], 3)))
res["probe_layers"] = prob; res["selfreport_r_same_rows"] = round(pearsonr(t, ev[m])[0], 3)
rng = np.random.default_rng(1); best = max(prob, key=lambda d: d["stim_r"])["layer"]; li = keep.index(best)
res["probe_shuffled"] = round(pearsonr(t, cvprobe(Hs[li][m], rng.permutation(t), grp))[0], 3)

# 3. walk
it, z, j = L("walk"); ev10, sd10, _, _ = dist(z["P1"], z["P2"], 10); evcm, sdcm, _, _ = dist(z["P1"], z["P2"], 39)
wr = {}; curves = {}
for i, x in enumerate(it):
    key = f"a{x['agent']}_N{x['N']}_{'told' if x['told'] else 'untold'}{'_rev' if x['rev'] else ''}_{x['q']}"
    curves.setdefault(key, []).append((x["k"], x["s"] if x["q"] == "fa_scale" else x["cm"], float(evcm[i] if x["q"] == "fa_cm" else ev10[i]), float(sdcm[i] if x["q"] == "fa_cm" else sd10[i])))
def fits(k, y):
    k = np.asarray(k, float); y = np.asarray(y, float); lin = np.polyfit(k, y, 1); rl = 1 - np.sum((np.polyval(lin, k) - y) ** 2) / np.sum((y - y.mean()) ** 2)
    best = -1
    for c in k[1:]:  # single step function
        pr = np.where(k < c, y[k < c].mean(), y[k >= c].mean()); best = max(best, 1 - np.sum((pr - y) ** 2) / np.sum((y - y.mean()) ** 2))
    dy = np.diff(y); mono = float(np.mean(np.sign(dy) == np.sign(y[-1] - y[0]))) if len(dy) else 0
    return dict(r2_linear=round(rl, 3), r2_step=round(best, 3), monotone_frac=round(mono, 2), max_jump_frac=round(float(np.max(abs(dy)) / (abs(y[-1] - y[0]) + 1e-9)), 2))
for key, c in curves.items():
    c = sorted(c); k, tr, e, s = map(np.array, zip(*c))
    wr[key] = dict(**stats(tr, e), **fits(k, e), ev=np.round(e, 2).tolist(), sd=np.round(s, 2).tolist())
res["walk"] = wr
# walk probe per layer: train on states of told walks (agents 0,1), test on agent 2
Hs = z["Hs"].astype(np.float32); keep = j["keep"]; sc = np.array([x["q"] == "fa_scale" and not x["rev"] for x in it]); ag = np.array([x["agent"] for x in it]); t = np.array([x["s"] for x in it])
wp = []
for li, Ly in enumerate(keep):
    p = cvprobe(Hs[li][sc], t[sc], ag[sc]); wp.append(dict(layer=Ly, r_heldout_agent=round(pearsonr(t[sc], p)[0], 3)))
res["walk_probe"] = wp

# 4. intensity
it, z, j = L("int"); ev10, sd10, _, _ = dist(z["P1"], z["P2"], 10); keep = j["keep"]; Hs = z["Hs"].astype(np.float32)
ir = {}
for dim in ("press", "temp"):
    m = np.array([x["exp"] == dim for x in it]); rk = np.array([x["rank"] for x in it])[m]; pl = np.array([x["place"] for x in it])[m]; e = ev10[m]
    d = dict(**stats(rk, e), ev_by_rank=[round(float(e[rk == r].mean()), 2) for r in range(rk.max() + 1)])
    lay = []
    for li, Ly in enumerate(keep):  # leave-one-place-out difference-of-means direction; projection vs EV and rank
        H = Hs[li][m]; pr = np.zeros(m.sum())
        for p in range(3):
            tr = pl != p; mx = rk.max(); v = H[tr & (rk >= mx - 2)].mean(0) - H[tr & (rk <= 2)].mean(0); pr[~tr] = H[~tr] @ v / np.linalg.norm(v)
        lay.append(dict(layer=Ly, proj_vs_ev=round(pearsonr(pr, e)[0], 3), proj_vs_rank=round(spearmanr(pr, rk)[0], 3)))
    d["layers"] = lay; ir[dim] = d
res["intensity"] = ir
# 2. steering
try:
    st = json.load(open(f"{D}/steer.json")); sr = {}
    for x in st:
        sr.setdefault(f"L{x['L']}_{x['key']}_{x.get('mode','stim')}_g{x['g']}", []).append((x["alpha"], x["ev"]))
    out = {}
    for k, c in sr.items():
        a, e = map(np.array, zip(*sorted(c))); f = fits(a * 20, e)
        out[k] = dict(range=round(float(e.max() - e.min()), 2), ev=np.round(e, 2).tolist(), spearman=round(spearmanr(a, e)[0], 3), **f)
    res["steer"] = out; res["steer_dnorm"] = {f"L{x['L']}_{x['key']}": (round(x["dnorm"], 1), round(x["hnorm"], 1)) for x in st}
except FileNotFoundError: pass
json.dump(res, open(f"{D}/ana.json", "w"), indent=1)
for k, v in res["static"].items(): print(k, v)
print("probe", [(p["layer"], p["stim_r"], p["ans_r"]) for p in res["probe_layers"]], "self", res["selfreport_r_same_rows"], "shuf", res["probe_shuffled"])
for k, v in res["walk"].items(): print(k, {a: b for a, b in v.items() if a not in ("sd",)})
print("walkprobe", [(p["layer"], p["r_heldout_agent"]) for p in res["walk_probe"]])
for k, v in res["intensity"].items(): print(k, {a: b for a, b in v.items() if a != "layers"}, [(l["layer"], l["proj_vs_ev"], l["proj_vs_rank"]) for l in v["layers"]])
for k, v in res.get("steer", {}).items(): print(k, v["range"], v["spearman"], v["r2_linear"], v["r2_step"], v["ev"][::4])
