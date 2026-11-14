import numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
from mesh import load
u=np.load('unposed.npz'); U=u['U']; part=u['part']; SP=u['SP']; SR=u['SR']
P,T=load()
grp={0:['torso','head','neck','armL','armR','legL','legR'],1:['armL','handL','torso'],2:['armR','handR','torso'],3:['legL','footL','torso'],4:['legR','footR','torso']}
E=np.vstack([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]])
A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(len(P),)*2).tocsr(); A=((A+A.T)>0).astype(float)
Dg=np.asarray(A.sum(1)).ravel(); Lw=sp.diags(1/Dg)@A
def paint(vals, sigma=3.0, k=10, smooth=12):
    out=np.zeros((len(P),vals.shape[1]))
    for p,regs in grp.items():
        mm=part==p; ok=np.isin(SR,regs); tr=cKDTree(SP[ok]); dd,ii=tr.query(U[mm],k=k)
        w=np.exp(-(dd-dd[:,:1])**2/(2*sigma**2)); w/=w.sum(1,keepdims=True)
        out[mm]=(w[...,None]*vals[ok][ii]).sum(1)
    for _ in range(smooth): out=0.5*out+0.5*(Lw@out)
    return out
