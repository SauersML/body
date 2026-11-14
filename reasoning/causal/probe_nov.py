"""Probes on last-token states (out32n/states.pt). Folds by scenario parity (fit on one half of scenarios, test on the other).
P1 swap-vs-normal: fit on bodies, test bodies + objects (transfer); and fit on objects.
P2 strength of the 'rearranged' signal: projection of (swap - twin) onto the shared swap axis (fit on the other fold), body vs object, in units of the normal-item spread.
P3 does that per-item strength predict the model's swap margin, within each side?"""
import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
I = json.load(open("items_nov.json")); ST = torch.load("out32n/states.pt").float().numpy()
m = np.array(json.load(open("out32n/margins.json"))["base"], float)
task = np.array([i["task"] for i in I]); fold = np.array([i["scen"] % 2 for i in I]); twin = np.array([i["twin"] for i in I])
out = {}
for L in [8, 16, 24, 32, 40, 44, 48, 52, 56, 60, 64]:
    X = ST[:, L]; X = (X - X.mean(0)) / (X.std(0) + 1e-6); r = {}
    for side in ["body", "obj"]:
        other = "obj" if side == "body" else "body"; accs = []
        for f in [0, 1]:
            tr = np.isin(task, [f"{side}_swap", f"{side}_norm"]) & (fold != f)
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], task[tr] == f"{side}_swap")
            te_s = np.isin(task, [f"{side}_swap", f"{side}_norm"]) & (fold == f); te_o = np.isin(task, [f"{other}_swap", f"{other}_norm"]) & (fold == f)
            accs.append((clf.score(X[te_s], task[te_s] == f"{side}_swap"), clf.score(X[te_o], task[te_o] == f"{other}_swap")))
        r[f"P1_{side}"] = np.mean(accs, 0).round(3).tolist()
    # P2/P3 shared swap axis from the other fold
    strength = np.full(len(I), np.nan)
    for f in [0, 1]:
        d = []
        for side in ["body", "obj"]:
            sw = (task == f"{side}_swap") & (fold != f); d.append((ST[sw, L] - ST[twin[sw], L]).mean(0))
        u = d[0] / np.linalg.norm(d[0]) + d[1] / np.linalg.norm(d[1]); u /= np.linalg.norm(u)
        sw = np.isin(task, ["body_swap", "obj_swap"]) & (fold == f); strength[sw] = (ST[sw, L] - ST[twin[sw], L]) @ u
    nsd = np.std(ST[np.isin(task, ["body_norm", "obj_norm"]), L] @ u)
    for side in ["body", "obj"]:
        s = task == f"{side}_swap"; r[f"P2_{side}"] = round(float(np.nanmean(strength[s]) / nsd), 2)
        r[f"P3_{side}_r"] = round(float(np.corrcoef(strength[s], m[s])[0, 1]), 2)
    out[L] = r; print(L, r, flush=True)
json.dump(out, open("out32n/probes.json", "w"))
