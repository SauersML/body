import json, os, sys, numpy as np, re
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import mstim as MS, stim as S
R = MS.probes(); KEEP = [22, 24, 25, 26, 27, 28, 30]; C = json.load(open(f"{D}/contexts.json"))
LAY = int(os.environ.get("LAY", 27)); li = KEEP.index(LAY)
loc = [i for i, r in enumerate(R) if r["kind"] == "loc"]; spots = MS.SPOTS
def grp(l):
    r = l["region"]
    if r == "torso": return "torsoB" if l["n"][2] < 0 else "torsoF"
    return r
G = np.array([grp(S.LOCS[s]) for s in spots]); GN = ["head", "neck", "torsoF", "torsoB", "armL", "armR", "handL", "handR", "legL", "legR", "footL", "footR"]
XYZ = np.array([S.LOCS[s]["xyz"] for s in spots])
newkeys = [p for _, p in MS.NEW]
def load(n):
    H = np.load(f"{D}/H/{n.replace(':', '__')}.npy")[li].astype(np.float32)
    X = np.zeros((len(spots), H.shape[1])); Nw = np.zeros((len(MS.NEW), H.shape[1]))
    for i, r in enumerate(R):
        if r["kind"] == "loc": X[spots.index(r["li"])] += H[i] / 2
        else: Nw[r["li"]] += H[i] / 2
    mu = X.mean(0); X = X - mu; Nw = Nw - mu; s = np.sqrt((X ** 2).sum(1).mean())
    return X / s, Nw / s  # centred and scaled by the context's own spread
cache = {}
def L(n):
    if n not in cache: cache[n] = load(n)
    return cache[n]
def regmask(regs): return np.isin(np.array([S.LOCS[s]["region"] for s in spots]), regs)
def spread(X, m):  # mean pairwise distance within region / mean pairwise distance over the body
    Y = X[m]; d = np.linalg.norm(Y[:, None] - Y[None], axis=-1); a = np.linalg.norm(X[:, None] - X[None], axis=-1)
    return d[np.triu_indices(len(Y), 1)].mean() / a[np.triu_indices(len(X), 1)].mean()
def cent(X): return np.array([X[G == g].mean(0) for g in GN])
def cdist(X):
    c = cent(X); d = np.linalg.norm(c[:, None] - c[None], axis=-1); return d / d[np.triu_indices(len(GN), 1)].mean()
def disp(a, b): return np.linalg.norm(L(a)[0] - L(b)[0], axis=1)  # per-spot displacement (units: body rms radius)
def delta(a, b): return (L(a)[0] - L(b)[0]).ravel()
def cos(u, v): return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
out = {}
ref = "neutral"
for m, (grpname, narr, act, sham, regs) in MS.MODS.items():
    o = {}; tn, ta, sh, pe = f"told_N:{m}", f"told_A:{m}", f"sham:{m}", f"persist:{m}"
    mk = regmask(regs) if regs else np.ones(len(spots), bool)
    for c in (tn, ta, sh, pe):
        r0 = "persist:neutral" if c == pe else ref
        d = disp(c, r0); o[c.split(":")[0]] = dict(disp=float(d.mean()), target=float(d[mk].mean() / d[~mk].mean()) if regs else None,
                                                    spread=float(spread(L(c)[0], mk) / spread(L(r0)[0], mk)) if regs else None)
    dN, dA, dS, dP = delta(tn, ref), delta(ta, ref), delta(sh, ref), delta(pe, "persist:neutral")
    o["cos_N_A"] = cos(dN, dA); o["cos_N_sham"] = cos(dN, dS); o["cos_N_persist"] = cos(dN, dP)
    o["persist_mag"] = float(np.linalg.norm(dP) / np.linalg.norm(dN))
    # beyond mention: told vs sham
    dts = disp(tn, sh); o["vs_sham_disp"] = float(dts.mean()); o["vs_sham_target"] = float(dts[mk].mean() / dts[~mk].mean()) if regs else None
    o["spot_disp_vs_sham"] = dts.tolist(); o["spot_disp_vs_neutral"] = disp(tn, ref).tolist()
    # centroid distance changes, told_N vs sham
    dd = cdist(L(tn)[0]) - cdist(L(sh)[0]); iu = np.triu_indices(len(GN), 1); order = np.argsort(dd[iu])
    o["closer"] = [(GN[iu[0][k]], GN[iu[1][k]], round(float(dd[iu][k]), 3)) for k in order[:4]]
    o["farther"] = [(GN[iu[0][k]], GN[iu[1][k]], round(float(dd[iu][k]), 3)) for k in order[::-1][:3]]
    o["cdist_told"] = cdist(L(tn)[0]).tolist(); o["cdist_sham"] = cdist(L(sh)[0]).tolist()
    # region spread for every group (told vs sham)
    o["spread_vs_sham"] = {g: float(spread(L(tn)[0], G == g) / spread(L(sh)[0], G == g)) for g in GN if (G == g).sum() > 2}
    # new-part probes: nearest body groups, in this context and in base
    X, Nw = L(tn); Xb, Nb = L(ref); cX, cB = cent(X), cent(Xb)
    o["new"] = {}
    for k, p in enumerate(newkeys):
        dm = np.linalg.norm(cX - Nw[k], axis=1); db = np.linalg.norm(cB - Nb[k], axis=1)
        nn = np.sort(np.linalg.norm(X - Nw[k], axis=1))[:5].mean(); nnb = np.sort(np.linalg.norm(Xb - Nb[k], axis=1))[:5].mean()
        o["new"][p] = dict(near=[GN[j] for j in np.argsort(dm)[:3]], near_base=[GN[j] for j in np.argsort(db)[:3]], nn=float(nn), nn_base=float(nnb))
    out[m] = o
# agency
ag = {}
told = {m: delta(f"told_N:{m}", ref) for m in MS.MODS}
for a, (tm, q) in MS.AGENCY.items():
    o = {}; ds = [delta(f"do:{a}:{s}", ref) for s in range(3)]
    o["cos_do_told"] = [cos(d, told[tm]) for d in ds]
    o["cos_do_told_actionfmt"] = [cos(d, delta(f"told_A:{tm}", ref)) for d in ds]
    o["cos_do_do"] = [cos(ds[i], ds[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    o["cos_do_sham"] = [cos(d, delta(f"sham:{tm}", ref)) for d in ds]
    o["rank_of_told"] = [int(sorted(told, key=lambda m: -cos(d, told[m])).index(tm)) + 1 for d in ds]
    o["best_told"] = [max(told, key=lambda m: cos(d, told[m])) for d in ds]
    o["mag_do_over_told"] = [float(np.linalg.norm(d) / np.linalg.norm(told[tm])) for d in ds]
    dp = delta(f"persist:{a}:0", "persist:neutral"); dpt = delta(f"persist:{tm}", "persist:neutral")
    o["persist_cos_do"] = cos(dp, ds[0]); o["persist_mag_do"] = float(np.linalg.norm(dp) / np.linalg.norm(ds[0]))
    o["persist_cos_told"] = cos(dpt, told[tm]); o["persist_mag_told"] = float(np.linalg.norm(dpt) / np.linalg.norm(told[tm]))
    o["persist_abs_do"] = float(np.linalg.norm(dp)); o["persist_abs_told"] = float(np.linalg.norm(dpt))
    o["abs_do"] = float(np.mean([np.linalg.norm(d) for d in ds])); o["abs_told"] = float(np.linalg.norm(told[tm]))
    # side test: displacement of left vs right limb for request / edited text
    side = (regmask(["armL", "handL"]), regmask(["armR", "handR"])) if "arm" in a or "hand" in a else (regmask(["legL", "footL"]), regmask(["legR", "footR"]))
    for c in [f"do:{a}:0", f"told_N:{tm}"] + ([f"edit:{a}"] if a in MS.EDIT else []):
        d = disp(c, ref); o["LR_" + c.split(":")[0]] = float(d[side[0]].mean() / d[side[1]].mean())
    ag[a] = o
# noise floor: two neutral-like contexts
out["_noise"] = dict(base_vs_neutral=float(disp("base", ref).mean()))
json.dump(dict(mods=out, agency=ag, groups=GN, xyz=XYZ.tolist(), G=G.tolist()), open(f"{D}/results_L{LAY}.json", "w"), indent=1)
print("noise base vs neutral disp", out["_noise"])
for m, o in out.items():
    if m.startswith("_"): continue
    f = lambda x: "-" if x is None else f"{x:.2f}"
    print(f"{m:15s} dispN {o['told_N']['disp']:.2f} sham {o['sham']['disp']:.2f} | targN {f(o['told_N']['target'])} targSham {f(o['sham']['target'])} vsSham {o['vs_sham_disp']:.2f} targ {f(o['vs_sham_target'])} | spreadN {f(o['told_N']['spread'])} sham {f(o['sham']['spread'])} | cosNA {o['cos_N_A']:.2f} cosNS {o['cos_N_sham']:.2f} | pers cos {o['cos_N_persist']:.2f} mag {o['persist_mag']:.2f} | closer {o['closer'][:2]}")
for a, o in ag.items(): print(a, {k: (np.round(v, 2).tolist() if isinstance(v, list) and not isinstance(v[0], str) else (round(v, 2) if isinstance(v, float) else v)) for k, v in o.items()})
