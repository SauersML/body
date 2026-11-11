import os
import sys, os, numpy as np, json
sys.path.insert(0, os.path.expandvars("$BODY_REPO/hires/v2")); os.chdir(os.path.expandvars("$BODY_REPO/hires/v2"))
import ana3 as A
from sklearn.utils.extmath import randomized_svd
out = {}
for L in [18, 20, 22, 23, 24, 25, 26, 27, 28]:
    X, rows = A.load("out8b_long_person", "reply", L)
    kind = np.array([r["kind"] for r in rows]); li = np.array([r["li"] for r in rows]); tp = np.array([r["tpl"] for r in rows]); T = tp.max() + 1
    X = X.copy()
    for t in range(T): X[tp == t] -= X[(tp == t) & (kind == "loc")].mean(0)
    Y = np.zeros((A.n, T, X.shape[1]), np.float32); Y[li[kind == "loc"], tp[kind == "loc"]] = X[kind == "loc"]; del X
    Y -= Y.mean((0, 1)); Vt = randomized_svd(Y.reshape(-1, Y.shape[2]), 800, random_state=0)[2]
    Yr = Y @ Vt.T; V, ev, kh = A.reliable(Yr, A.splits(T))
    C = np.zeros((800, 800)); sp = A.splits(T)
    for p in sp: a, c = Yr[:, p[:T//2]].mean(1), Yr[:, p[T//2:]].mean(1); C += (a.T @ c + c.T @ a) / 2
    ev, E = np.linalg.eigh(C); E = E[:, ::-1][:, :150]
    U = (Vt.T @ E).astype(np.float32); U, _ = np.linalg.qr(U)
    var = np.linalg.norm(Y.mean(1) @ U) ** 2 / np.linalg.norm(Y.mean(1)) ** 2
    print(L, "kh", kh, "spot-mean var captured by 150:", round(float(var), 3), flush=True)
    out[f"U{L}"] = U.astype(np.float16); out[f"kh{L}"] = kh
np.savez(os.path.expandvars("$BODY_REPO/egodeath/bodyU.npz"), **out); print("done")
