import numpy as np
from chains import *; from colors import pcs
from scipy.linalg import orthogonal_procrustes
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
LAYERS=[int(l) for l in R['keep']]
side=np.array([l['side'] for l in L])
Fd=np.c_[abs(SP[:,0]),SP[:,1],SP[:,2]]; mu=Fd.mean(0); F0=Fd-mu
def fold_view(Lr,k=3):
    Z=R[f'reply|{Lr}|all|A'].astype(np.float32); Y,_=pcs(Z,k); Y=Y-Y.mean(0)
    Rm,_=orthogonal_procrustes(Y,F0); V=Y@Rm; V*=np.sqrt((F0**2).sum()/(V**2).sum()); return V+mu
def sym_view(Lr):
    V=fold_view(Lr); a0=V[side==0,0].mean()
    X=np.where(side==-1,-(V[:,0]-a0),V[:,0]-a0)
    return np.c_[X,V[:,1]]
KEY={L_:sym_view(L_) for L_ in LAYERS}
