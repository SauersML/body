"""ana.py OUTDIR -> summary.json + printed table. Behaviour: mean of the number distribution (cm). Internal: ridge rulers
trained on ruler items pooled over all contexts, each context's ruler mean subtracted; CV r = leave-objects-out (5 folds)."""
import sys, json, numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
O = sys.argv[1]; R = json.load(open(f"{O}/res.json")); ST = np.load(f"{O}/states.npz")
CTX = R["ctx"]; LAY = R["layers"]; P4 = ["sitting", "standing", "lying", "neutral"]
def grp(cid, s): return "long_self" if cid.startswith("long") else ("short_self" if s == "self" else "short_person")
def ci(v):
    v = np.asarray(v, float); m = v.mean()
    if len(v) < 2: return [m, m, m]
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [m, m - h, m + h]
out = {"beh": {}, "int": {}}
# behaviour
per = {}
for b in R["beh"]:
    p = np.array(b["prob"]); ev = (p * np.arange(300)).sum() / p.sum()
    per.setdefault((b["ctx"], b["kind"]), []).append((ev, b["digitmass"]))
cs = {c[0]: c for c in CTX}
for kind in ["head", "eyes", "tall"]:
    for g in ["long_self", "short_self", "short_person"]:
        for c in P4:
            v = [np.mean([e for e, _ in per[(cid, kind)]]) for cid, cc, s in CTX if cc == c and grp(cid, s) == g]
            dm = [np.mean([d for _, d in per[(cid, kind)]]) for cid, cc, s in CTX if cc == c and grp(cid, s) == g]
            out["beh"][f"{kind}|{g}|{c}"] = dict(cm=ci(v), digitmass=float(np.mean(dm)), n=len(v))
# internal
def ruler(kind, names, hts, logy, tsel):
    res = {}
    its = {s: R["items"][s] for s in R["items"]}
    for li, L in enumerate(LAY):
        Xtr, ytr, gtr, reads = [], [], [], []
        for cid, c, s in CTX:
            X = ST[cid][:, li].astype(np.float32); it = its[s]
            ri = [i for i, x in enumerate(it) if x["kind"] == kind and x["X"] in names]; mu = X[ri].mean(0)
            Xtr.append(X[ri] - mu); ytr += [hts[it[i]["X"]] for i in ri]; gtr += [it[i]["X"] for i in ri]
            for t in tsel(s):
                ti = [i for i, x in enumerate(it) if x["kind"] == kind and x["X"] == t[0]]
                reads.append((cid, c, s, t[1], X[ti] - mu))
        Xtr = np.concatenate(Xtr); y = np.array(ytr); y = np.log(y) if logy else y
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3; Z = (Xtr - mu) / sd
        g = np.unique(gtr, return_inverse=True)[1]; pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(Z, y, g):
            pred[te] = RidgeCV(alphas=np.logspace(1, 6, 11)).fit(Z[tr], y[tr]).predict(Z[te])
        cvr = float(np.corrcoef(pred, y)[0, 1]); m = RidgeCV(alphas=np.logspace(1, 6, 11)).fit(Z, y)
        f = (lambda v: np.exp(v)) if logy else (lambda v: v)
        tab = {}
        for cid, c, s, lab, Xt in reads:
            tab.setdefault((grp(cid, s), lab, c), []).append(float(f(m.predict((Xt - mu) / sd).mean())))
        res[L] = dict(cv_r=cvr, read={f"{g}|{lab}|{c}": ci(v) for (g, lab, c), v in tab.items()})
    return res
EH = {x: h for x, h in R["elev"]}; SH = {x: h for x, h in R["ent"]}
def etarg(s):
    if s == "self": return [("my head", "head"), ("my eyes", "eyes"), ("my feet", "feet"), ("my knees", "knees")]
    n = "the man" if s == "man" else "the woman"; return [(f"{n}'s head", "head"), (f"{n}'s eyes", "eyes"), (f"{n}'s feet", "feet"), (f"{n}'s knees", "knees")]
def starg(s): return [("I", "stature")] if s == "self" else [(("The man" if s == "man" else "The woman"), "stature")]
out["int"]["elev"] = ruler("elev", set(EH), EH, False, etarg)
out["int"]["stat"] = ruler("stat", set(SH), SH, True, starg)
json.dump(out, open(f"{O}/summary.json", "w"), indent=1, default=float)
f = lambda v: f"{v[0]:.0f} [{v[1]:.0f},{v[2]:.0f}]"
for kind in ["head", "eyes", "tall"]:
    for g in ["long_self", "short_self", "short_person"]:
        print("BEH", kind, g, " | ".join(f"{c} {f(out['beh'][f'{kind}|{g}|{c}']['cm'])} dm{out['beh'][f'{kind}|{g}|{c}']['digitmass']:.2f}" for c in P4))
for k in ["elev", "stat"]:
    for L, r in out["int"][k].items():
        print(k.upper(), "L", L, "cv_r %.2f" % r["cv_r"])
        for g in ["long_self", "short_self", "short_person"]:
            for lab in (["head", "eyes", "feet", "knees"] if k == "elev" else ["stature"]):
                print("   ", g, lab, " | ".join(f"{c} {100*np.array(r['read'][f'{g}|{lab}|{c}'])[0]:.0f} [{100*r['read'][f'{g}|{lab}|{c}'][1]:.0f},{100*r['read'][f'{g}|{lab}|{c}'][2]:.0f}]" for c in P4))
