import numpy as np, os, json
import ana2
from ana7 import *
from ana2 import VE
ST3=np.load("out3/st.npz")
def HH(c,q,pos,L):
    src=ST3 if f"{c}|{q}|{pos}" in ST3 else ST
    h=src[f"{c}|{q}|{pos}"].astype(np.float32); return h[None,READ.index(L)] if pos=="dp" else h[:,READ.index(L)]
LS=[25,26,27,28,30]
def prof(c,q,b,pos):
    out=[]
    for L in LS:
        T=tdirs(L); d=HH(c,q,pos,L).mean(0)-HH(c,b,pos,L).mean(0); p=T@d/np.linalg.norm(d); out.append((p-p.mean())/p.std())
    return np.mean(out,0)
def fh(p): return np.mean([p[TYPES.index(t)] for t in FOOTT])-np.mean([p[TYPES.index(t)] for t in HANDT])
res={}
for pos in ["dp","reply12","reply"]:
    R={"own":prof("normal","self_toe2","self_burn2",pos)}
    for c in ["normal","me","random","leftright"]:
        R[c]=(prof(c,"woman_toe","woman_burn",pos)+prof(c,"man_toe","man_burn",pos))/2
    for k,v in R.items(): print(MAP,pos,k,"foot-hand %+.2f"%fh(v),"r(own) %.2f"%np.corrcoef(v,R["own"])[0,1],[ (TYPES[i],round(float(v[i]),1)) for i in np.argsort(-v)[:5]])
    # owner axis (toe and burn prompts, woman+man), minus added vector, 0 = her (normal), 1 = own (normal)
    vals={c:[] for c in ["normal","me","random","leftright"]}
    for L in LS:
        X=M[MAP][:,READ.index(L)].astype(np.float32); u=X[cond=="self"].mean(0)-X[cond=="other"].mean(0); u/=np.linalg.norm(u)
        for q,s_ in [("woman_toe","self_toe2"),("man_toe","self_toe2"),("woman_burn","self_burn2"),("man_burn","self_burn2")]:
            w=HH("normal",q,pos,L).mean(0)@u; s=HH("normal",s_,pos,L).mean(0)@u
            for c in vals: vals[c].append(((HH(c,q,pos,L).mean(0)-(VE[c] if c!="normal" else 0))@u-w)/(s-w))
    own={c:round(float(np.mean(v)),2) for c,v in vals.items()}; print(MAP,pos,"owner axis",own, {c:round(float(np.std(v)),2) for c,v in vals.items()})
    res[pos]={"types":TYPES,"prof":{k:v.tolist() for k,v in R.items()},"fh":{k:float(fh(v)) for k,v in R.items()},"owner":own,"owner_all":{c:[float(x) for x in v] for c,v in vals.items()}}
json.dump(res,open(f"out3/figdata_{MAP}.json","w"),indent=1)
