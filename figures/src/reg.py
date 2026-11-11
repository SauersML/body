import os
import sys; sys.path.insert(0,os.path.expandvars('$BODY_REPO/hires/v2'))
import numpy as np
from ana2 import *
S=load_stim(os.path.expandvars("$BODY_REPO/hires/v2/stim.py")); L=S.LOCS
P=np.array([l["xyz"] for l in L]); N=np.array([l["n"] for l in L])
REG={"face":[i for i,l in enumerate(L) if l["region"]=="head" and l["n"][2]>0.4 and 150<l["xyz"][1]<173],
 "hand":[i for i,l in enumerate(L) if l["region"]=="handL" and "wrist" not in l["phrase"]],
 "forearm":[i for i,l in enumerate(L) if "left forearm" in l["phrase"]],
 "foot":[i for i,l in enumerate(L) if l["region"]=="footL"],
 "lower leg":[i for i,l in enumerate(L) if l["region"]=="legL" and any(w in l["phrase"] for w in ["shin","calf","lower leg"])]}
def states():
    d=os.path.expandvars("$BODY_REPO/hires/v2/"); out={}
    for nm,f,k in [("8B","red8b_short.npz","reply|27|all"),("32B","red32_short.npz","reply|53|all"),("words","red8b_lex.npz","lex|0|all")]:
        R=np.load(d+f); A=R[k+"|A"].astype(np.float32); W=R[k+"|W"].astype(np.float32); out[nm]=embed(A,W)[0]
    return out
