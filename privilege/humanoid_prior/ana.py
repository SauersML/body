"""state analyses (run where X.npy is): matched-graph maps, reference frame, centrality, fill-in states, drift states."""
import sys, json, numpy as np, os
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, H + "/../nonhuman")
import b16
D = sys.argv[1]; li = 1
X = np.load(f"{D}/X.npy")[:, li].astype(np.float32); rows = json.load(open(f"{D}/rows.json"))
rc = np.array([r["chat"] for r in rows]); rl = np.array([r["loc"] for r in rows]); rt = np.array([r["tpl"] for r in rows])
out = {}
def C(n, keys=None, tpls=range(8)):
    ks = keys or list(dict.fromkeys(rl[rc == n]))
    return ks, np.stack([X[(rc == n) & (rl == k) & np.isin(rt, list(tpls))].mean(0) for k in ks])
def rdm(M): M = M - M.mean(0); U = M / np.linalg.norm(M, axis=1, keepdims=True); return 1 - U @ U.T
iu = np.triu_indices(16, 1)
def rsa(M, G): return spearmanr(rdm(M)[iu], G[iu])[0]
def pcorr(M, G, Z):
    from scipy.stats import rankdata
    a, b, c = rankdata(rdm(M)[iu]), rankdata(G[iu]), rankdata(Z[iu])
    r = lambda x, y: np.corrcoef(x, y)[0, 1]
    rab, rac, rbc = r(a, b), r(a, c), r(b, c); return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
K16 = [str(i) for i in range(16)]; rng = np.random.default_rng(0)
CAN = b16.G16[np.ix_(b16.PI, b16.PI)]  # distance between the canonical places of the names
m = {}
for b in b16.BODIES16:
    n = "16" + b; G = b16.GRAD if b == "trell" else b16.G16
    _, M = C(n, K16); boots = [rsa(C(n, K16, rng.choice(8, 8))[1], G) for _ in range(200)]
    m[b] = dict(rsa=rsa(M, G), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                split=(rsa(C(n, K16, range(0, 8, 2))[1], G) + rsa(C(n, K16, range(1, 8, 2))[1], G)) / 2)
    if b.endswith("_re"): m[b].update(rsa_names_canonical=rsa(M, CAN), partial_layout=pcorr(M, G, CAN), partial_names=pcorr(M, CAN, G))
out["map16"] = m
# reference frame: nearest-centroid decoding across bodies with node correspondence (single-wording states)
def dec(A, B, labelsB=None):
    _, CA = C(A, K16); CA = CA - CA.mean(0); CA /= np.linalg.norm(CA, axis=1, keepdims=True)
    sel = (rc == B) & np.isin(rl, K16); XB = X[sel]; XB = XB - XB.mean(0); lab = np.array([int(k) for k in rl[sel]])
    pred = np.argmax(XB @ CA.T, 1); return pred, lab
fr = {}
for A in ["human", "dog", "velk"]:
    for B in ["human", "dog", "velk"]:
        if A == B: continue
        p, l = dec("16" + A, "16" + B); fr[f"{A}->{B}"] = float((p == l).mean())
for b in ["human", "dog"]:
    p, l = dec("16" + b, "16" + b + "_re")
    fr[f"{b}_re by {b}: layout"] = float((p == l).mean()); fr[f"{b}_re by {b}: name"] = float((p == b16.PI[l]).mean())
out["frame"] = fr
# centrality: variance of target parts explained by each reference body's top-5 part PCs
def basis(M, k=5): M = M - M.mean(0); return np.linalg.svd(M, full_matrices=False)[2][:k]
def ev(M, V): M = M - M.mean(0); return float((np.linalg.norm(M @ V.T) ** 2) / np.linalg.norm(M) ** 2)
chats = [c for c in dict.fromkeys(rc) if not c.startswith("f")]
cent = {c: C(c)[1] for c in chats}
refs = ["16human", "16dog", "16velk", "16trell", "dog", "octopus", "bird"]
hk = list(dict.fromkeys(rl[rc == "human54"])); hsub = [rng.choice(len(hk), 16, replace=False) for _ in range(30)]
evt = {}
for r in refs + ["human54 (16 random places)"]:
    row = {}
    for t in chats:
        if t == r or (r.startswith("human54") and t == "human54"): continue
        row[t] = float(np.mean([ev(cent[t], basis(cent["human54"][s])) for s in hsub])) if r.startswith("human54") else ev(cent[t], basis(cent[r]))
    evt[r] = row
out["centrality"] = evt
# fill-in: front-minus-back limb-end vector vs references
def dv(n, a, b): return C(n, [a])[1][0] - C(n, [b])[1][0]
refv = {"human hand-foot (16human)": dv("16human", "8", "13"), "dog front-hind paw (16dog)": dv("16dog", "8", "13"),
        "velk frond-pad (16velk)": dv("16velk", "8", "13"), "human palm-sole (human54)": dv("human54", "palmL", "soleL")}
cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
out["fill"] = {f: {k: cs(dv(f, "front", "back"), v) for k, v in refv.items()} for f in dict.fromkeys(rc) if f.startswith("f")}
# drift: XS (turns, traj, layers, d); move along ref0 - own0, as fraction of the distance
dr = {}
if os.path.exists(f"{D}/XS.npy"):
  XS = np.load(f"{D}/XS.npy")[:, :, li].astype(np.float32); order = json.load(open(f"{D}/t3.json"))["order"]
  bod = np.array([o[0] for o in order]); B = list(dict.fromkeys(bod))
  mu0 = {b: XS[0, bod == b].mean(0) for b in B}
  for b in B:
    for r in B:
        if r == b: continue
        u = mu0[r] - mu0[b]; dr[f"{b} toward {r}"] = [float(((XS[t, bod == b] - mu0[b]) @ u).mean() / (u @ u)) for t in range(XS.shape[0])]
out["drift"] = dr
json.dump(out, open(f"{D}/ana.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["map16", "frame", "fill"]}, indent=0)[:4000])
for r, row in evt.items(): print(r, round(np.mean(list(row.values())), 3), {t: round(v, 2) for t, v in row.items() if t in ("16human", "16dog", "jellyfish", "octopus", "robot", "human54")})
for k, v in dr.items(): print(k, [round(x, 2) for x in v[::3]])
