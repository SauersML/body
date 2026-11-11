import numpy as np
from ana7 import *
from ana2 import VE
for pos in ["dp","reply"]:
  for L in [25,27,30,32]:
    X=M[MAP][:,READ.index(L)].astype(np.float32); u=X[cond=="self"].mean(0)-X[cond=="other"].mean(0); u/=np.linalg.norm(u)
    w=Hs("normal","woman_toe",pos,L).mean(0)@u; s=Hs("normal","self_toe",pos,L).mean(0)@u
    line=f"{pos} L{L} owner axis (0=woman's toe normal, 1=own toe normal):"
    for c in ["me","random","leftright"]:
        h=Hs(c,"woman_toe",pos,L).mean(0); line+=f" {c[:2]} {(h@u-w)/(s-w):+.2f} (minus added vector {((h-VE[c])@u-w)/(s-w):+.2f})"
    print(line)
for pos in ["dp","reply"]:
    vals={c:[] for c in ["me","random","leftright"]}
    for L in [25,26,27,28,30]:
        X=M[MAP][:,READ.index(L)].astype(np.float32); u=X[cond=="self"].mean(0)-X[cond=="other"].mean(0); u/=np.linalg.norm(u)
        for q in ["woman_toe","man_toe"]:
            w=Hs("normal",q,pos,L).mean(0)@u; s=Hs("normal","self_toe",pos,L).mean(0)@u
            for c in vals: vals[c].append(((Hs(c,q,pos,L).mean(0)-VE[c])@u-w)/(s-w))
    print(pos,"L25-30 avg, woman+man, minus vector:",{c:round(float(np.mean(v)),2) for c,v in vals.items()})
