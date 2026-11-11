import numpy as np, json
from ana7 import *
import os; LS=[25,26,27,28,30]; pos=os.environ.get("POS","reply")
def prof(c,q,b,pos=pos):
    out=[]
    for L in LS:
        T=tdirs(L); d=Hs(c,q,pos,L).mean(0)-Hs(c,b,pos,L).mean(0); p=T@d/np.linalg.norm(d); out.append((p-p.mean())/p.std())
    return np.mean(out,0)
def fh(p): return np.mean([p[TYPES.index(t)] for t in FOOTT])-np.mean([p[TYPES.index(t)] for t in HANDT])
R={"own":prof("normal","self_toe","self_burn")}
for c in ["normal","me","random","leftright"]: R[c]=prof(c,"woman_toe","friend_burn"); R[c+"_man"]=prof(c,"man_toe","friend_burn")
# per-reply spread for foot-hand score
S={}
for c in ["normal","me","random","leftright"]:
    ss=[]
    for L in LS:
        T=tdirs(L); a=Hs(c,"woman_toe",pos,L); b=Hs(c,"friend_burn",pos,L).mean(0); ss.append(score(a-b,T))
    S[c]=np.mean(ss,0)
for k,v in R.items(): print(k, "foot-hand %+.2f"%fh(v), "r(own) %.2f"%np.corrcoef(v,R["own"])[0,1], "top", [TYPES[i] for i in np.argsort(-v)[:5]])
for c,v in S.items(): print(c, "per-reply foot-hand mean %.2f sd %.2f"%(v.mean(),v.std()))
json.dump({"types":TYPES,"prof":{k:v.tolist() for k,v in R.items()},"fh":{k:float(fh(v)) for k,v in R.items()},"perreply":{k:v.tolist() for k,v in S.items()}},open(f"out2/figdata_{pos}.json","w"),indent=1)
