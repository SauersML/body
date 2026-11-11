import os
import sys, json; sys.path.insert(0, os.path.expandvars("$BODY_REPO/figures/src"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sdf import REGIONS, sdf_np
Z = states()
def rho(key, ix):
    iu = np.triu_indices(len(ix), 1); return float(spearmanr(squareform(pdist(Z[key][ix]))[iu], squareform(pdist(P[ix]))[iu])[0])
def pca2(X):
    X = X - X.mean(0); u, s, vt = np.linalg.svd(X, full_matrices=False); c = u[:, :2] * s[:2]; return c / c.std(0)
def procrustes(A, B):  # rotate/reflect A onto B
    u, _, vt = np.linalg.svd(A.T @ B); return A @ (u @ vt)
def oklab(c, rot):
    r = np.linalg.norm(c, axis=1); th = np.arctan2(c[:, 1], c[:, 0]) + rot
    rs = np.sqrt(1 - np.exp(-r**2 / 2))
    return np.stack([0.76 + 0*rs, 0.165 * rs * np.cos(th), 0.165 * rs * np.sin(th)], 1)
def snap(reg, p):
    p = p.copy()
    for _ in range(30):
        e = 1e-3; d = sdf_np(reg, p)
        g = np.stack([(sdf_np(reg, p + e*np.eye(3)[k]) - sdf_np(reg, p - e*np.eye(3)[k])) / (2*e) for k in range(3)], 1)
        p = p - d[:, None] * g / (np.linalg.norm(g, axis=1, keepdims=True)**2 + 1e-9)
    return p
REGN = {"face": "face", "hand": "hand", "foot": "foot", "forearm": "forearm"}
out = {}
for reg in ["face", "hand", "foot", "forearm"]:
    ix = REG[reg]; c32 = pca2(Z["32B"][ix]); cw = procrustes(pca2(Z["words"][ix]), c32)
    ps = snap(reg, P[ix].astype(float)); moved = np.linalg.norm(ps - P[ix], axis=1)
    nn = np.sort(squareform(pdist(ps)), 1)[:, 1]
    D = squareform(pdist(ps)); same = (N[ix] @ N[ix].T) > 0.3; D2 = np.where(same, D, np.inf); np.fill_diagonal(D2, np.inf)
    sig_i = np.clip(0.8 * np.sort(D2, 1)[:, 1], 0.9 * np.median(nn), 1.7 * np.median(nn))
    out[reg] = dict(sigi=sig_i.round(3).tolist(), pos=ps.round(4).tolist(), nrm=N[ix].tolist(), c32=oklab(c32, 0).round(4).tolist(), cw=oklab(cw, 0).round(4).tolist(),
                    sigma=float(np.median(nn)), r32=rho("32B", ix), rw=rho("words", ix), phr=[L[i]["phrase"] for i in ix])
    print(reg, len(ix), "moved max %.2f med %.2f" % (moved.max(), np.median(moved)), "sigma %.2f" % out[reg]["sigma"], "rho %.2f %.2f" % (out[reg]["r32"], out[reg]["rw"]))
json.dump(out, open("spots.json", "w"))
