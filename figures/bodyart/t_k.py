import numpy as np
from chains import *
from scipy.linalg import orthogonal_procrustes
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
T=SP-SP.mean(0); rng=np.random.default_rng(0)
for Lr in [53,57]:
    Z=R[f'reply|{Lr}|all|A'].astype(np.float64); Z=Z-Z.mean(0); u,s,vt=np.linalg.svd(Z,full_matrices=False)
    for k in [6,8,12,16,24]:
        Y=u[:,:k]*s[:k]
        for nm,TT in [('real',T),('shuf',T[rng.permutation(len(T))])]:
            Tp=np.c_[TT,np.zeros((len(T),k-3))]; Rm,_=orthogonal_procrustes(Y,Tp); V=(Y@Rm)[:,:3]
            print(Lr,k,nm,np.round([np.corrcoef(V[:,j],TT[:,j])[0,1] for j in range(3)],2),flush=True)
