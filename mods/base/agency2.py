"""Fair told-vs-do comparison: both sets of deltas use the mean over the same 6 posture tasks as the common component."""
import os, sys, json, numpy as np
sys.path.insert(0, '.'); os.environ.setdefault("LAY", "27")
import agency as AG  # prints; fine
MS = AG.MS; load = AG.load; nX, pX = AG.nX, AG.pX
tasks = list(MS.AGENCY); tm = {a: MS.AGENCY[a][0] for a in tasks}
Dt = {a: AG.T[tm[a]] for a in tasks}; Dd = {a: np.mean([load(f"do:{a}:{s}") - nX for s in range(3)], 0) for a in tasks}
Dd0 = {a: load(f"do:{a}:0") - nX for a in tasks}
Pt = {a: AG.Tp[tm[a]] for a in tasks}; Pd = {a: load(f"persist:{a}:0") - pX for a in tasks}
mt, md, mpt, mpd, md0 = [np.mean(list(d.values()), 0) for d in (Dt, Dd, Pt, Pd, Dd0)]
cos = AG.cos; out = {}
for a in tasks:
    t, d, d0, pt, pd = (Dt[a] - mt).ravel(), (Dd[a] - md).ravel(), (Dd0[a] - md0).ravel(), (Pt[a] - mpt).ravel(), (Pd[a] - mpd).ravel()
    sims = {b: cos(d, (Dt[b] - mt).ravel()) for b in tasks}
    out[a] = dict(cos_do_told=round(sims[a], 3), cos_do_other_told=round(np.mean([v for b, v in sims.items() if b != a]), 3), rank=sorted(sims, key=lambda b: -sims[b]).index(a) + 1,
                  size_do=round(float(np.linalg.norm(d0)), 2), size_told=round(float(np.linalg.norm(t)), 2),
                  kept_do=round(cos(d0, pd) * np.linalg.norm(pd) / np.linalg.norm(d0), 3), kept_told=round(cos(t, pt) * np.linalg.norm(pt) / np.linalg.norm(t), 3))
json.dump(out, open("agency_fair_L27.json", "w"), indent=1)
print("FAIR"); [print(a, v) for a, v in out.items()]
