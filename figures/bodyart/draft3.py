import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from chains import *; from colors import *
from scipy.linalg import orthogonal_procrustes
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
Fd=np.c_[abs(SP[:,0]),SP[:,1],SP[:,2]]; Fd=Fd-Fd.mean(0)
def view(Z,k=3):
    Y,_=pcs(Z,k); Y=Y-Y.mean(0); 
    if k>3:
        u,s,vt=np.linalg.svd(Y.T@Fd,full_matrices=False); Rm=u@vt  # k x 3
    else: Rm,_=orthogonal_procrustes(Y,Fd)
    V=Y@Rm; sc=(V*Fd).sum()/(V*V).sum(); return V*sc
sets={'ZR32':data()['q32'],'raw53':R['reply|53|all|A'].astype(np.float32),'ZR8':data()['q8'],'rnd':data()['rnd']}
fig,ax=plt.subplots(2,4,figsize=(20,12))
for c,(k,Z) in enumerate(sets.items()):
  for r,kk in enumerate([3,6]):
    V=view(Z,kk); a=ax[r,c]; a.scatter(V[:,0],V[:,1],s=4,c='0.75')
    for side,col in [('left','C3'),('right','C0')]:
        for nm,ch in chains(side).items():
            P=np.array([V[g].mean(0) for g in ch]); a.plot(P[:,0],P[:,1],'-',c=col)
    P=np.array([V[g].mean(0) for g in spine()]); a.plot(P[:,0],P[:,1],'k-'); a.plot(P[0,0],P[0,1],'ko')
    a.set_aspect('equal'); a.set_title(f'{k} k={kk} r={np.corrcoef(V.ravel(),Fd.ravel())[0,1]:.2f}')
plt.savefig('draft3.png',dpi=55)
