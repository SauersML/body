import os
import numpy as np, sys
from scipy.spatial import cKDTree
B2=os.path.expandvars('$BODY_REPO/figures/bodyvid2/')
d=np.load(B2+'data.npz',allow_pickle=True); XM=d['XM']; Yu=d['Yu']; SIDE=d['side']
V=np.fromfile(B2+'web/pos.bin',np.float32).reshape(-1,3).astype(float); F=np.fromfile(B2+'web/idx.bin',np.uint32).reshape(-1,3)
D=Yu-XM
def sidew(p): return np.where(SIDE[None]==0,1.0,1/(1+np.exp(-SIDE[None]*p[:,0:1]/2.5)))+1e-12
def blend(P,vals,sig,norm=True):
    out=np.zeros((len(P),vals.shape[1]))
    for i in range(0,len(P),20000):
        p=P[i:i+20000]; w=np.exp(-((p[:,None,:]-XM[None])**2).sum(-1)/(2*sig**2))*sidew(p)+(1e-300 if norm else 0)
        out[i:i+20000]=(w@vals)/(w.sum(1,keepdims=True) if norm else 1)
    return out
_,vi=cKDTree(V).query(XM)
E=np.unique(np.sort(np.r_[F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]],1),axis=0)
L0=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1)
def stats(Dv,name):
    P=V+Dv; L=np.linalg.norm(P[E[:,0]]-P[E[:,1]],axis=1)/L0
    print(name,'miss med %.1f p90 %.1f max %.1f'%tuple(np.percentile(np.linalg.norm(Dv[vi]-D,axis=1),[50,90,100])),'stretch p99 %.1f max %.1f'%(np.percentile(L,99),L.max()))
for SIG,lam in []:
    A=blend(V[vi],np.eye(407),SIG); C=np.linalg.solve(A.T@A+lam*np.eye(407),A.T@D); Dv=blend(V,C,SIG); stats(Dv,f'sig{SIG} lam{lam}')
    if SIG==6 and lam==0.1: base=Dv
    if SIG==4 and lam==0.001: best=Dv

SIG,lam=float(sys.argv[1]),float(sys.argv[2])
A=blend(V[vi],np.eye(407),SIG); C=np.linalg.solve(A.T@A+lam*np.eye(407),A.T@D); Dv=blend(V,C,SIG); stats(Dv,'chosen')
Dv.astype(np.float32).tofile('web/dv2.bin')
