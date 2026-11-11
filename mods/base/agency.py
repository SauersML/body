"""Agency vs told, persistence, and side test with modification-specific deltas (common 'body talk' component removed)."""
import os, sys, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import mstim as MS, stim as S
lay = int(os.environ.get("LAY", 27)); KEEP = [22, 24, 25, 26, 27, 28, 30]; li = KEEP.index(lay)
R = MS.probes(); spots = MS.SPOTS; ri = {s: k for k, s in enumerate(spots)}
def load(n):
    H = np.load(f"{D}/H/{n.replace(':', '__')}.npy")[li].astype(np.float32); X = np.zeros((len(spots), H.shape[1]))
    for i, r in enumerate(R):
        if r["kind"] == "loc": X[ri[r["li"]]] += H[i] / 2
    X -= X.mean(0); return X / np.sqrt((X ** 2).sum(1).mean())
mods = list(MS.MODS); nX = load("neutral"); pX = load("persist:neutral")
T = {m: load(f"told_N:{m}") - nX for m in mods}; mu = np.mean([T[m] for m in mods], 0)
Tp = {m: load(f"persist:{m}") - pX for m in mods}; mup = np.mean([Tp[m] for m in mods], 0)
spec = lambda d, c: (d - c).ravel()
def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
reg = np.array([S.LOCS[s]["region"] for s in spots])
out = {}
# persistence: own vs other mods
own = [cos(spec(T[m], mu), spec(Tp[m], mup)) for m in mods]
oth = [np.mean([cos(spec(T[m], mu), spec(Tp[o], mup)) for o in mods if o != m]) for m in mods]
mag = [float(np.linalg.norm(spec(Tp[m], mup)) / np.linalg.norm(spec(T[m], mu))) for m in mods]
out["persist"] = {m: dict(cos=round(a, 3), cos_other=round(b, 3), mag=round(c, 2)) for m, a, b, c in zip(mods, own, oth, mag)}
# told action format vs narration (specific)
A = {m: load(f"told_A:{m}") - nX for m in mods}; muA = np.mean([A[m] for m in mods], 0)
Sh = {m: load(f"sham:{m}") - nX for m in mods}; muS = np.mean([Sh[m] for m in mods], 0)
out["format"] = {m: dict(N_A=round(cos(spec(T[m], mu), spec(A[m], muA)), 3), N_A_other=round(np.mean([cos(spec(T[m], mu), spec(A[o], muA)) for o in mods if o != m]), 3),
                         N_sham=round(cos(spec(T[m], mu), spec(Sh[m], muS)), 3)) for m in mods}
# agency
ag = {}
for a, (tm, q) in MS.AGENCY.items():
    Ds = [load(f"do:{a}:{s}") - nX for s in range(3)]; Dm = np.mean(Ds, 0)
    sims = {m: cos(spec(Dm, mu), spec(T[m], mu)) for m in mods}
    rank = sorted(sims, key=lambda m: -sims[m]).index(tm) + 1
    Dp = load(f"persist:{a}:0") - pX
    o = dict(cos_told=round(sims[tm], 3), cos_told_other=round(np.mean([v for m, v in sims.items() if m != tm]), 3), rank=rank, best=sorted(sims, key=lambda m: -sims[m])[:3],
             mag_do_over_told=round(float(np.linalg.norm(spec(Dm, mu)) / np.linalg.norm(spec(T[tm], mu))), 2),
             persist_cos_do=round(cos(spec(Ds[0], mu), spec(Dp, mup)), 3), persist_mag_do=round(float(np.linalg.norm(spec(Dp, mup)) / np.linalg.norm(spec(Ds[0], mu))), 2),
             persist_cos_told=out["persist"][tm]["cos"], persist_mag_told=out["persist"][tm]["mag"])
    if a in MS.EDIT:  # side test: per-spot displacement of the requested-side limb vs other-side limb (specific delta)
        side = "arm" if ("arm" in a or "hand" in a) else "leg"
        Lm = np.isin(reg, ["armL", "handL"] if side == "arm" else ["legL", "footL"]); Rm = np.isin(reg, ["armR", "handR"] if side == "arm" else ["legR", "footR"])
        E = load(f"edit:{a}") - nX
        f = lambda d: round(float(np.linalg.norm(d - mu, axis=1)[Lm].mean() / np.linalg.norm(d - mu, axis=1)[Rm].mean()), 3)
        o["LR_request"] = f(Ds[0]); o["LR_edit"] = f(E); o["cos_edit_do"] = round(cos(spec(E, mu), spec(Ds[0], mu)), 3)
        o["edit_minus_do_LR"] = f(E - Ds[0] + mu)  # where does the edit move things relative to the unedited reply
        o["LR_told"] = f(T[tm])
    ag[a] = o
out["agency"] = ag
json.dump(out, open(f"{D}/agency_L{lay}.json", "w"), indent=1)
for k in ("persist", "format"):
    print(k); [print("  ", m, v) for m, v in out[k].items()]
print("agency"); [print("  ", a, v) for a, v in ag.items()]
