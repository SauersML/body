import os
# Data for the real-mesh body videos: spots on the neutral MakeHuman mesh, the Qwen3-32B L53 layout, warp field.
import sys, numpy as np
sys.path.insert(0,os.path.expandvars('$BODY_REPO/figures/bodyvid')); import os; os.chdir(os.path.expandvars('$BODY_REPO/figures/bodyvid'))
from geo import *            # Z (407 x 5120 model states, centered), PC, procrustes, XYZ, SIDE, CAT, PART, P1
from rend import colours
from scipy.spatial.distance import cdist
os.chdir(os.path.expandvars('$BODY_REPO/figures/bodyvid2'))
NB=os.path.expandvars('$BODY_REPO/figures/nb4/')
d=np.load('../src/nn2.npz'); ph=list(d['ph']); inn=np.array([ph.index(p) for p in P1])
XM=np.load(NB+'spot_pos_neutral.npy')[inn]                     # spots on the real mesh (cm)
V=np.fromfile(NB+'geo_pos.bin',np.float32).reshape(-1,3).astype(float); Fc=np.fromfile(NB+'geo_idx.bin',np.uint32).reshape(-1,3)
g0=np.load('../bodyvid/geo.npz'); sd=g0['sd']
# same layout recipe as v1 (Procrustes of PCs 1-4 onto the folded body, one uniform scale), fitted to the mesh spots
Fo=np.c_[np.abs(XM[:,0]),XM[:,1],XM[:,2]]; mu=Fo.mean(0)
R,c=procrustes(PC[:,:4],Fo-mu); Y=PC[:,:4]@R
Y=Y*np.sqrt(((Fo-mu)**2).sum()/(Y**2).sum())+mu
DZ=cdist(Z,Z); np.fill_diagonal(DZ,np.inf); DF=cdist(Fo,Fo); nn=np.argsort(DZ,1)
err=DF[np.arange(407)[:,None],nn[:,:5]].mean(1)
E=sorted({(min(i,j),max(i,j)) for i in range(407) for j in nn[i,:2]})
u0=np.median(Y[SIDE==0,0]); u=Y[:,0]-u0
Yu=np.c_[np.where(SIDE==0,u,sd*u),Y[:,1],Y[:,2]]               # unfolded with the model's own left/right sign
# outward normal per spot (nearest vertex normal)
from scipy.spatial import cKDTree
import trimesh
m=trimesh.Trimesh(V,Fc,process=False); VN=m.vertex_normals; tr=cKDTree(V); _,vi=tr.query(XM); NS=VN[vi]
# warp field: displacement of every mesh vertex, smooth (Gaussian) blend of the spots' displacements; left/right kept apart
D=Yu-XM; SIG=6.0; LAM=0.1
def sidew(p): return np.where(SIDE[None]==0,1.0,1/(1+np.exp(-SIDE[None]*p[:,0:1]/2.5)))+1e-12
def blend(P,vals,sig):
    out=np.zeros((len(P),vals.shape[1]))
    for i in range(0,len(P),20000):
        p=P[i:i+20000]; w=np.exp(-((p[:,None,:]-XM[None])**2).sum(-1)/(2*sig**2))*sidew(p)+1e-300
        out[i:i+20000]=(w@vals)/w.sum(1,keepdims=True)
    return out
# exact at the spots: solve for weights so the blend hits each spot's own displacement (regularized RBF on top of the blend)
A=blend(XM,np.eye(407),SIG)
for lam in (0.02,0.1,0.3):
    C=np.linalg.solve(A.T@A+lam*np.eye(407),A.T@D); Ds=A@C
    print(lam,'miss %.1f'%np.median(np.linalg.norm(Ds-D,axis=1)),'coef/move %.2f'%(np.abs(C).max()/np.abs(D).max()))
lam=LAM; C=np.linalg.solve(A.T@A+lam*np.eye(407),A.T@D); Dv=blend(V,C,SIG); Ds=A@C
print('spot miss median %.1f cm vs move %.1f cm'%(np.median(np.linalg.norm(Ds-D,axis=1)),np.median(np.linalg.norm(D,axis=1))))
cols=colours(CAT,SIDE); Vcol=blend(V,cols,3.5)
np.savez('data.npz',XM=XM,NS=NS,Y=Y,Yu=Yu,sd=sd,err=err,E=np.array(E),side=SIDE,cat=CAT,part=PART,cols=cols,d32=d['d_32B'][inn])
os.makedirs('web',exist_ok=True)
for k,a,t in [('pos',V,np.float32),('idx',Fc,np.uint32),('nrm',VN,np.float32),('dv',Dv,np.float32),('vcol',Vcol,np.float32)]: a.astype(t).tofile(f'web/{k}.bin')
import shutil; shutil.copy(NB+'col_32B.bin','web/col32.bin'); shutil.copy(NB+'three.module.js','web/'); shutil.copytree(NB+'jsm','web/jsm',dirs_exist_ok=True)
print('Y range',Y.min(0).round(),Y.max(0).round(),'XM',XM.min(0).round(),XM.max(0).round(), 'sd acc',(sd[SIDE!=0]==SIDE[SIDE!=0]).mean())
