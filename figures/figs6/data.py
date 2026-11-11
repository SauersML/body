import os, sys, numpy as np
os.chdir(os.path.expandvars('$BODY_REPO/mods/base')); sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base'))
import mstim as MS, stim as S
li=[22,24,25,26,27,28,30].index(27); R=MS.probes(); spots=MS.SPOTS; ri={q:k for k,q in enumerate(spots)}
def raw(n):
    H=np.load(f"H/{n.replace(':','__')}.npy")[li].astype(np.float32); X=np.zeros((len(spots),H.shape[1]))
    for i,r in enumerate(R):
        if r["kind"]=="loc": X[ri[r["li"]]]+=H[i]/2
    return X
def load(n):
    X=raw(n); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())
REG=np.array([("torso" if S.LOCS[q]["region"] in ("torso","neck") else S.LOCS[q]["region"]) for q in spots])
XYZ=np.array([S.LOCS[q]["xyz"] for q in spots])
GR=["head","torso","armL","armR","handL","handR","legL","legR","footL","footR"]
def reg(d): return {g: round(float(d[REG==g].mean()/d.mean()),2) for g in GR}
