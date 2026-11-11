import numpy as np
from ana2 import ST, M, READ, KEYS, placedirs, MAP, cond, loc
TYPES=list(dict.fromkeys(k[:-1] if k[-1] in "LR" and k not in ("neckF","neckB") else k for k in KEYS))
def tkey(k): return k[:-1] if k[-1] in "LR" and k not in ("neckF","neckB") else k
FOOTT=["toes","sole","heel","ankle"]; HANDT=["thumb","index","middle","ring","little","palm","backhand"]
def tdirs(L):
    Pk,mu=placedirs(L); T=np.stack([Pk[[i for i,k in enumerate(KEYS) if tkey(k)==t]].mean(0) for t in TYPES]); T-=T.mean(0); return T/np.linalg.norm(T,axis=1,keepdims=True)
def Hs(c,q,pos,L):
    h=ST[f"{c}|{q}|{pos}"].astype(np.float32); return h[None,READ.index(L)] if pos=="dp" else h[:,READ.index(L)]
def score(d,T):
    p=(d@T.T)/np.linalg.norm(d,axis=-1,keepdims=True); zz=(p-p.mean(-1,keepdims=True))/p.std(-1,keepdims=True)
    return zz[...,[TYPES.index(t) for t in FOOTT]].mean(-1)-zz[...,[TYPES.index(t) for t in HANDT]].mean(-1)
# map sanity: held-out template decoding of types for self touches at L
for pos in ["dp","reply12","reply"]:
  for L in [25,27,30,32]:
    T=tdirs(L); line=f"{MAP} {pos:7s} L{L}"
    own=Hs("normal","self_toe",pos,L).mean(0)-Hs("normal","self_burn",pos,L).mean(0); line+=f" OWN {float(score(own,T)):+.2f} |"
    for c in ["normal","me","random","leftright"]:
        a=Hs(c,"woman_toe",pos,L); b=Hs(c,"friend_burn",pos,L).mean(0); s=score(a-b,T)
        a2=Hs(c,"man_toe",pos,L); s2=score(a2-b,T)
        line+=f" {c[:2]} {s.mean():+.2f}±{s.std()/np.sqrt(len(s)):.2f} man {s2.mean():+.2f} |"
    print(line)
