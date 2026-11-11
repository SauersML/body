import os
import sys, os, json, numpy as np, networkx as nx
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from scipy.stats import spearmanr, pearsonr
SO = os.path.expandvars("$BODY_REPO/map/somato/out2_8b")
H = np.load(f"{SO}/H_rs.npy", mmap_mode="r"); rows = json.load(open(f"{SO}/rows.json"))
nloc = len(S.LOCS); idx = np.zeros((nloc, 8), int)
for k, r in enumerate(rows):
    if r["cond"] == "self": idx[r["li"], r["tpl"]] = k
keys = [l["key"] for l in S.LOCS]
tk = list(dict.fromkeys(l["type"] for l in S.LOCS)); Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tk}
rep = [Tidx[t][0] for t in tk]; it = np.triu_indices(len(tk), 1)
# --- body distances ---
Gsteps = S.model_rdms()["body path distance"][np.ix_(rep, rep)]
P = dict(crown=(0,170,0), forehead=(0,160,9), cheekL=(5,150,7), earL=(7.5,153,0), lips=(0,146,9), tongue=(0,146,5),
 neckF=(0,135,6), neckB=(0,138,-5), chest=(0,122,11), belly=(0,100,11), upback=(0,125,-11), lowback=(0,100,-10),
 shoulderL=(19,142,0), upperarmL=(21,128,0), elbowL=(22,108,-2), forearmL=(22,96,1), wristL=(22,84,2), palmL=(22,77,4),
 backhandL=(22,74,0), thumbL=(25,73,4), indexL=(23,66,3), middleL=(21.5,65,3), ringL=(20,66,3), littleL=(18.5,68,3),
 hipL=(14,90,0), thighL=(11,72,3), kneeL=(10,50,4), shinL=(10,35,5), ankleL=(10,8,0), heelL=(10,2,-5), soleL=(10,0,5), toesL=(10,1,18))
for k in list(P):
    if k.endswith("L"): x, y, z = P[k]; P[k[:-1] + "R"] = (-x, y, z)
X = np.array([P[keys[i]] for i in rep], float)
Gcm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
# along-the-body cm: same graph as steps, edges weighted by straight cm between joined parts
g0 = S.geodesic.__code__  # rebuild graph with cm weights
import types
G = nx.Graph()
src = open(os.path.expandvars("$BODY_REPO/map/somato/stimuli.py")).read()
# reuse stimuli edges by monkeypatching networkx Graph.add_edge to record
edges = []
class Rec(nx.Graph):
    def add_edge(self, a, b, **kw): edges.append((a, b)); super().add_edge(a, b, **kw)
_g = nx.Graph; nx.Graph = Rec; S.geodesic(); nx.Graph = _g
for a, b in edges: G.add_edge(a, b, weight=float(np.linalg.norm(np.subtract(P[a], P[b]))))
Dd = dict(nx.all_pairs_dijkstra_path_length(G))
Gpath = np.array([[Dd[keys[i]][keys[j]] for j in rep] for i in rep])
BODY = {"skeleton-graph steps": Gsteps, "straight-line cm": Gcm, "cm along the body": Gpath}
# --- representations ---
def typemeans(L, side="pooled"):
    a = np.asarray(H[idx, L], np.float64)          # nloc x 8 x d
    if side == "pooled": return np.stack([a[Tidx[t]].reshape(-1, a.shape[-1]).mean(0) for t in tk])
    return np.stack([a[Tidx[t][0]].mean(0) for t in tk])
def cos_d(M, center=True):
    if center: M = M - M.mean(0)
    n = M / np.linalg.norm(M, axis=1, keepdims=True); return 1 - n @ n.T
def corr_d(M, center=True):
    if center: M = M - M.mean(0)
    M = M - M.mean(1, keepdims=True); n = M / np.linalg.norm(M, axis=1, keepdims=True); return 1 - n @ n.T
def euc(M): M = M - M.mean(0); return np.sqrt(((M[:, None] - M[None]) ** 2).sum(-1))
def cveuc(L):
    a = np.asarray(H[idx, L], np.float32); a = a - a.mean(0, keepdims=True)
    at = np.stack([a[Tidx[t]].mean(0) for t in tk]); A, B = at[:, 0::2].mean(1), at[:, 1::2].mean(1)
    return np.einsum("ijd,ijd->ij", A[:, None] - A[None], B[:, None] - B[None])
r = lambda D, B: spearmanr(D[it], B[it])[0]
W = np.load(f"{SO}/wordemb.npy").astype(np.float64); Wt = np.stack([W[Tidx[t]].mean(0) for t in tk])
out = {}
print("word emb: cos-centered", {k: round(r(cos_d(Wt), B), 3) for k, B in BODY.items()},
      "cos-raw", round(r(cos_d(Wt, False), Gsteps), 3), "euc", round(r(euc(Wt), Gsteps), 3))
out["word"] = {k: r(cos_d(Wt), B) for k, B in BODY.items()}
NL = H.shape[1]; tab = {}
for L in range(1, NL):
    M = typemeans(L)
    tab[L] = dict(cos_c=r(cos_d(M), Gsteps), cos_raw=r(cos_d(M, False), Gsteps), corr=r(corr_d(M), Gsteps), euc=r(euc(M), Gsteps),
                  cveuc=r(cveuc(L), Gsteps), cos_c_cm=r(cos_d(M), Gcm), cos_c_path=r(cos_d(M), Gpath),
                  cos_c_left=r(cos_d(typemeans(L, "left")), Gsteps))
print("L   " + " ".join(f"{k:>10}" for k in tab[1]))
for L in tab: print(f"{L:<3} " + " ".join(f"{v:10.3f}" for v in tab[L].values()))
json.dump(dict(tab=tab, word=out["word"]), open("anal.json", "w"), indent=1)
np.save("bodyD.npy", np.stack([Gsteps, Gcm, Gpath])); json.dump(tk, open("tk.json", "w"))
