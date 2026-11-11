import numpy as np, json, sys
from ana2 import *
S = load_stim(sys.argv[3] if len(sys.argv) > 3 else "stim.py"); b = Body(S); probes = S.probes()
S1 = load_stim("../hires/stim.py"); b1 = Body(S1); p2 = [l["phrase"] for l in S.LOCS]; IDX = [p2.index(l["phrase"]) for l in S1.LOCS]
pr1 = [dict(p, kind="wf") for p in S1.wordfree()]; NWF = len(pr1)
R = np.load(sys.argv[1]); keep = list(R["keep"]); out = {}
def get(rd, k, sub="all"): return R[f"{rd}|{k}|{sub}|A"].astype(np.float32), R[f"{rd}|{k}|{sub}|W"].astype(np.float32)
def run(name, A, W, **kw):
    Z, ZW = embed(A, W, **kw); out[name] = b.metrics(Z, ZW, probes)
    Z1, ZW1 = embed(A[IDX], W[:NWF], **kw); out[name].update({"v1set_" + k: v for k, v in b1.metrics(Z1, ZW1, pr1).items()}); return Z
if "lex|0|all|A" in R.files: run("lex", *get("lex", 0))
for rd in ["reply", "final", "body"]:
    for k in keep:
        if f"{rd}|{k}|all|A" in R.files: run(f"{rd}|L{k}", *get(rd, k))
mid = keep[3:12]
if "lex|0|all|A" in R.files: keep = []; mid = []  # 22..30 in 8B terms
def cat(rds, ks):
    As, Ws = [], []
    for rd in rds:
        for k in ks:
            A, W = get(rd, k); mu = A.mean(0); sc = np.linalg.norm(A - mu, axis=1).mean(); As.append((A - mu) / sc); Ws.append((W - mu) / sc)
    return np.concatenate(As, 1), np.concatenate(Ws, 1)
for rd in ([] if not mid else ["reply", "final", "body"]): run(f"{rd}|cat", *cat([rd], mid)); run(f"{rd}|cat|white", *cat([rd], mid), whiten=True)
if mid: run("all3|cat", *cat(["reply", "final", "body"], mid))
if mid: run("all3|cat|white", *cat(["reply", "final", "body"], mid), whiten=True)
for rd in ["reply", "final", "body"]:
    for k in ([keep[5], keep[8], keep[11]] if keep else []):
        for sub in [s.split("|")[2] for s in R.files if s.startswith(f"{rd}|{k}|") and s.endswith("|A")]:
            if sub != "all": run(f"{rd}|L{k}|{sub}", *get(rd, k, sub))
json.dump(out, open(sys.argv[2], "w"), indent=0)
COLS = ["v1set_nn5_cm", "v1set_chain_rho", "v1set_partner_nn", "v1set_probe_wf_rank", "nn5_cm", "chain_rho", "shaft_pair_rho", "partner_nn", "probe_wf_rank", "probe_alt_rank", "alt_inshaft_cm"]
print("name".ljust(28), " ".join(c.replace("v1set_", "1:")[:9].rjust(9) for c in COLS))
for k, v in out.items(): print(k.ljust(28), " ".join(f"{v.get(c, np.nan):9.3g}" for c in COLS))
