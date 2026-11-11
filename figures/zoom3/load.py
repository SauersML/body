import os
import numpy as np, collections
def load(path=os.path.expandvars("$BODY_REPO/figures/zoom3/base.obj")):
    V=[]; G=collections.defaultdict(list); g=None
    for ln in open(path):
        if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("g "): g=ln.split()[1]
        elif ln.startswith("f "): G[g].append([int(t.split("/")[0])-1 for t in ln.split()[1:]])
    return np.array(V), G
def tri(F):
    T=[]
    for f in F:
        for k in range(1,len(f)-1): T.append([f[0],f[k],f[k+1]])
    return np.array(T)
def joints(V,G):
    return {k[6:]:V[np.unique(np.concatenate(G[k]))].mean(0) for k in G if k.startswith("joint-")}
