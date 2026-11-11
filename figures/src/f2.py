import sys; sys.path.insert(0,'.')
import numpy as np
from ana2 import *
S=load_stim("stim.py"); L=S.LOCS; n=len(L)
face=lambda l: ("inside" if "the inside" in l["phrase"] else "back" if "the back" in l["phrase"] else "thumb" if "thumb side" in l["phrase"] else "little" if "little-finger" in l["phrase"] else None)
def facen(l):
    nv=np.array(l["n"]); return ["inside","back","thumb","little"][[np.dot(nv,v) for v in ([0,0,1],[0,0,-1],[0,1,0],[0,-1,0])].index(max(np.dot(nv,v) for v in ([0,0,1],[0,0,-1],[0,1,0],[0,-1,0])))]
P=np.array([l["xyz"] for l in L])
for arm in ["left","right"]:
  fa=[i for i,l in enumerate(L) if f"{arm} forearm" in l["phrase"]]
  print(arm,len(fa), all(face(L[i])==facen(L[i]) for i in fa))
emb={}
for nm,f,k in [("8B","red8b_short.npz","reply|27|all"),("32B","red32_short.npz","reply|53|all"),("words","red8b_lex.npz","lex|0|all")]:
    R=np.load(f); A=R[k+"|A"].astype(np.float32); W=R[k+"|W"].astype(np.float32); emb[nm]=embed(A,W)[0]
e3=np.load("emb3_b53.npz"); emb["32B rel"]=e3["ZR"]; emb["8B rel"]=np.load("emb3_s27.npz")["ZR"]
for nm,Z in emb.items():
  for arm in ["left","right"]:
    fa=np.array([i for i,l in enumerate(L) if f"{arm} forearm" in l["phrase"]]); F=np.array([facen(L[i]) for i in fa])
    pos=np.array([P[i,0] for i in fa])
    D=squareform(pdist(Z[fa])); np.fill_diagonal(D,np.inf); nn=np.argsort(D,1)[:,:5]
    same=np.mean(F[nn]==F[:,None]); same1=np.mean(F[nn[:,0]]==F)
    chance=np.mean([(np.sum(F==F[i])-1)/(len(fa)-1) for i in range(len(fa))])
    # along-arm distance of nn (cm)
    dal=np.mean(np.abs(pos[nn]-pos[:,None])); dal_r=np.mean(np.abs(pos[:,None]-pos[None]) [~np.eye(len(fa),dtype=bool)])
    # whole-body nn: share that are forearm same arm & same face
    DG=squareform(pdist(Z)); np.fill_diagonal(DG,np.inf); nng=np.argsort(DG[fa],1)[:,:5]
    inarm=np.isin(nng,fa).mean()
    fg=np.array([[facen(L[j]) if "forearm" in L[j]["phrase"] else "x" for j in r] for r in nng]); sameg=np.mean(fg==F[:,None])
    print(f"{nm:8s} {arm:5s} within-forearm 5NN same side {same:.2f} (1NN {same1:.2f}) chance {chance:.2f} | along-arm gap nn {dal:.1f}cm vs random {dal_r:.1f} | whole-body 5NN: on this forearm {inarm:.2f}, forearm same side {sameg:.2f}")
