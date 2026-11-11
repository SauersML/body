import numpy as np, os
from ana7 import *
LS=[25,26,27,28,30]; pos=os.environ.get("POS","dp")
def tcent(L):
    X=M[MAP][cond=="self"][:,READ.index(L)].astype(np.float32); lc=np.array([tkey(k) for k in loc[cond=="self"]])
    C=np.stack([X[lc==t].mean(0) for t in TYPES]); return C-C.mean(0)
def prof(c,q,b,lam):
    out=[]
    for L in LS:
        C=tcent(L); d=Hs(c,q,pos,L).mean(0)-Hs(c,b,pos,L).mean(0)
        G=C@C.T; w=np.linalg.solve(G+lam*np.trace(G)/len(G)*np.eye(len(G)),C@d); out.append(w/np.abs(w).max())
    return np.mean(out,0)
for lam in [0.01,0.1,1.0]:
    print("lam",lam)
    for nm,(c,q,b) in {"own":("normal","self_toe","self_burn"),"normal":("normal","woman_toe","friend_burn"),"me":("me","woman_toe","friend_burn"),"random":("random","woman_toe","friend_burn")}.items():
        p=prof(c,q,b,lam); o=np.argsort(-p); print(" ",nm,[(TYPES[i],round(float(p[i]),2)) for i in o[:6]],"bottom",[TYPES[i] for i in o[-4:]])
