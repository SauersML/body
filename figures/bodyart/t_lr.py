import numpy as np
from chains import *; from colors import pcs
from scipy.linalg import orthogonal_procrustes
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
side=np.array([l['side'] for l in L]); print(len(SP), np.bincount(side+1))
T=SP-SP.mean(0)
for Lr in [28,41,53,57]:
    Z=R[f'reply|{Lr}|all|A'].astype(np.float64); Z=Z-Z.mean(0)
    for k in [3,5,10]:
        u,s,vt=np.linalg.svd(Z,full_matrices=False); Y=u[:,:k]*s[:k]
        # fit linear map k->3 orthogonal: procrustes from Y (k) to T padded
        Tp=np.c_[T,np.zeros((len(T),k-3))]; Rm,_=orthogonal_procrustes(Y,Tp); V=(Y@Rm)[:,:3]; V*=np.sqrt((T**2).sum()/(V**2).sum())
        r=[np.corrcoef(V[:,j],T[:,j])[0,1] for j in range(3)]
        print(Lr,k,np.round(r,2))
