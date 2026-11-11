"""Metrics inside the body-map subspace (top K PCs of the neutral-context spot states)."""
import os, sys, numpy as np, json
os.environ.setdefault("LAY", "27")
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import analyze as A  # reuses loaders (runs analyze once; cheap enough)
K = int(os.environ.get("K", 10))
Xn = A.L("neutral")[0]; U, Sv, Vt = np.linalg.svd(Xn, full_matrices=False); V = Vt[:K].T
def P(n): X, Nw = A.L(n); return X @ V, Nw @ V
GN, G = A.GN, A.G
def spread(Y, m): d = np.linalg.norm(Y[m][:, None] - Y[m][None], axis=-1); a = np.linalg.norm(Y[:, None] - Y[None], axis=-1); return d[np.triu_indices(m.sum(), 1)].mean() / a[np.triu_indices(len(Y), 1)].mean()
def cd(Y):
    c = np.array([Y[G == g].mean(0) for g in GN]); d = np.linalg.norm(c[:, None] - c[None], axis=-1); return d / d[np.triu_indices(len(GN), 1)].mean()
def gi(g): return GN.index(g)
PAIRS = {"L_arm_up": [("handL", "head"), ("armL", "head"), ("handL", "legL")], "arms_crossed": [("handL", "torsoF"), ("handR", "torsoF"), ("handL", "handR")],
 "cross_legged": [("footL", "legR"), ("footL", "footR"), ("legL", "legR")], "on_side": [("head", "armR"), ("armR", "legR")],
 "R_hand_behind": [("handR", "torsoB"), ("handR", "torsoF")], "R_leg_lifted": [("legR", "torsoF"), ("footR", "legL"), ("footR", "footL")],
 "L_arm_long": [("handL", "footL"), ("handL", "legL")], "tree": [("footL", "footR"), ("armL", "armR"), ("legL", "legR")],
 "liquid": [("head", "footL"), ("handL", "footL")], "ribbon": [("head", "footL"), ("head", "handL")]}
res = {}
for m, (g, _, _, _, regs) in A.MS.MODS.items():
    Yt, Nt = P(f"told_N:{m}"); Ys, Ns = P(f"sham:{m}"); Yn, Nn = P("neutral"); Yp, _ = P(f"persist:{m}"); Ypn, _ = P("persist:neutral")
    o = {}
    groups = sorted(set(G[A.regmask(regs)])) if regs else []
    o["spread"] = {gg: (spread(Yt, G == gg) / spread(Ys, G == gg), spread(Yt, G == gg) / spread(Yn, G == gg), spread(Yp, G == gg) / spread(Ypn, G == gg)) for gg in GN if gg != "neck"}
    Ct, Cs, Cn, Cp, Cpn = cd(Yt), cd(Ys), cd(Yn), cd(Yp), cd(Ypn)
    o["C"] = dict(t=Ct.tolist(), s=Cs.tolist(), n=Cn.tolist(), p=Cp.tolist(), pn=Cpn.tolist())
    o["pairs"] = {f"{a}-{b}": (Ct[gi(a), gi(b)] / Cs[gi(a), gi(b)], Ct[gi(a), gi(b)] / Cn[gi(a), gi(b)], Cp[gi(a), gi(b)] / Cpn[gi(a), gi(b)]) for a, b in PAIRS.get(m, [])}
    # per-group displacement of centroid (told vs sham), relative to mean displacement
    dt = np.linalg.norm(Yt - Ys, axis=1); o["gdisp"] = {gg: float(dt[G == gg].mean() / dt.mean()) for gg in GN}
    dtn = np.linalg.norm(Yt - Yn, axis=1); o["gdisp_n"] = {gg: float(dtn[G == gg].mean() / dtn.mean()) for gg in GN}
    o["spot_disp"] = dt.tolist(); o["spot_disp_n"] = dtn.tolist()
    c = np.array([Yt[G == gg].mean(0) for gg in GN]); cb = np.array([Yn[G == gg].mean(0) for gg in GN])
    o["new"] = {p: dict(near=[GN[j] for j in np.argsort(np.linalg.norm(c - Nt[k], axis=1))[:3]], near_neutral=[GN[j] for j in np.argsort(np.linalg.norm(cb - Nn[k], axis=1))[:3]])
                for k, p in enumerate(A.newkeys)}
    res[m] = o
json.dump(res, open(f"{D}/sub_L{os.environ['LAY']}_K{K}.json", "w"))
if __name__ == "__main__":
    for m, o in res.items():
        print(m, "spread(t/sham,t/neu,persist)", {k: tuple(round(x, 2) for x in v) for k, v in o["spread"].items()}, "pairs", {k: tuple(round(x, 2) for x in v) for k, v in o["pairs"].items()})
        top = sorted(o["gdisp"].items(), key=lambda x: -x[1])[:3]; print("    most moved vs sham", [(k, round(v, 2)) for k, v in top])
