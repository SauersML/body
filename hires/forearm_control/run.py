import json, numpy as np, fana as A, fstim as S, sys
from collections import defaultdict
D0 = sys.argv[1]; R = json.load(open(f"{D0}/rows.json")); rows = R["rows"]; KEEP = R["keep"]
X = np.load(f"{D0}/reply.npy", mmap_mode="r"); E = np.load(f"{D0}/emb.npz")
conds = ["orig", "para", "syn", "mix"]
by = defaultdict(list); phr = defaultdict(set)
for i, r in enumerate(rows): by[(r["cond"], r["si"])].append(i); phr[(r["cond"], r["si"])].add(r["text"])
def spotmat(M, c): return np.stack([np.asarray(M[by[(c, s)]], np.float32).mean(0) for s in range(80)])
out = {"keep": KEEP}; arms = {a: np.where(A.ARM == a)[0] for a in ["left", "right"]}
def allm(V, ctrl=None, do_prsa=False, c=None):
    res = defaultdict(list); pr = []
    for a, idx in arms.items():
        D = A.dmat(V[idx])
        for k, v in A.metrics(D, idx).items(): res[k].append(v)
        if do_prsa: pr.append(A.prsa(D, idx, [m[np.ix_(idx, idx)] for m in ctrl], nperm=300))
    o = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
    if do_prsa: o["prsa"] = pr
    return o
out["chance"] = {k: round(v, 3) for k, v in A.chance(arms["left"]).items()}
for c in conds:
    Ep = spotmat(E["emb_phrase"], c); Es = spotmat(E["emb_sent"], c); B = A.bow([sorted(phr[(c, s)]) for s in range(80)])
    ctrl = [A.dmat(Ep), A.dmat(Es), A.dmat(B, "cosine")]
    o = {"emb_phrase": allm(Ep), "emb_sent": allm(Es), "bow": allm(B)}
    for li, L in enumerate(KEEP):
        V = spotmat(X[li], c); o[f"L{L}"] = allm(V, ctrl, do_prsa=(L in (44, 53, 60)) , c=c)
    out[c] = o; print(c, json.dumps({k: v for k, v in o.items() if not k.startswith("L") or k == "L53"}), flush=True)
json.dump(out, open("fctl_results.json", "w"), indent=1)
