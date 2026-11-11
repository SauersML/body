import os
# build a soft body surface (metaballs on a skeleton fitted to the 407 spots), and its smoothed warp into model space
import sys, numpy as np, torch
sys.path.insert(0,os.path.expandvars("$BODY_REPO/hires/v1"))
import skel
from scipy.spatial import cKDTree
g=np.load('geo.npz'); X=g['body']; side=g['side']; cat=g['cat']; NR=g['nrm']
G={k:np.array(v) for k,v in skel.groups().items()}
C={k:X[v].mean(0) for k,v in G.items()}
def rad(k): return float(np.clip(np.median(np.linalg.norm(X[G[k]]-C[k],axis=1)),1.2,12))
segs=[]   # (a, b, ra, rb)
for s in "LR":
    arm=["shoulder","ua1","ua2","ua3","ua4","ua5","elbow","fa1","fa2","fa3","fa4","fa5","fa6","wrist"]
    leg=["hip","th1","th2","th3","th4","th5","knee","sh1","sh2","sh3","sh4","sh5","ankle"]
    for chain in (arm,leg):
        for a,b in zip(chain,chain[1:]): segs.append((C[a+s],C[b+s],rad(a+s)*0.95,rad(b+s)*0.95))
    pc=C['palm'+s]; segs.append((C['wrist'+s],pc,rad('wrist'+s)*0.9,3.8))
    for f in ["thumb","index","middle","ring","little"]:
        P=X[G[f+s]]; tip=P[np.argmax(np.linalg.norm(P-pc,axis=1))]; base=pc+(C[f+s]-pc)*0.35
        segs.append((base,tip,1.3,1.0))
    segs.append((C['ankle'+s],C['foot'+s],3.6,3.2)); segs.append((C['foot'+s],C['toes'+s],3.2,2.2))
# torso from spine slices: width from flank spots, depth from front/back spots
T=X[cat=='torso']; ys=np.linspace(T[:,1].min()+2,T[:,1].max()-2,7)
for y0,y1 in zip(ys,ys[1:]):
    for y in (y0,):
        S=T[np.abs(T[:,1]-y)<10]; hw=np.clip(np.percentile(np.abs(S[:,0]),90),10,20); zc=(S[:,2].max()+S[:,2].min())/2; hd=max((S[:,2].max()-S[:,2].min())/2,8)
    for x in np.linspace(-hw+hd*0.8,hw-hd*0.8,3):
        segs.append((np.array([x,y0,zc]),np.array([x,y1,zc]),hd,hd))
# shoulders/hips joined to torso
for s,sg in (("L",1),("R",-1)):
    segs.append((C['shoulder'+s],np.array([sg*8,C['shoulder'+s][1]-6,C['shoulder'+s][2]]),rad('shoulder'+s),8))
    segs.append((C['hip'+s],np.array([sg*6,C['hip'+s][1]+4,C['hip'+s][2]]),rad('hip'+s)*1.1,9))
# head and neck
H=X[cat=='head']; hc=np.array([0,H[:,1].mean()+1,H[:,2].mean()-1]); hr=np.ptp(H,0)/2
segs.append((C['neck'],np.array([0,C['neck'][1]+8,C['neck'][2]-1]),5.5,5.5))
for dy in (-3,3): segs.append((hc+[0,dy,0],hc+[0,dy,0],min(hr[0],9.5),min(hr[0],9.5)))
segs.append((hc+[0,-5,4],hc+[0,-9,5],5.5,4.5))
A=np.array([s[0] for s in segs]); B=np.array([s[1] for s in segs]); RA=np.array([s[2] for s in segs]); RB=np.array([s[3] for s in segs])
def field(P):
    P=torch.tensor(P,dtype=torch.float32); a=torch.tensor(A,dtype=torch.float32); b=torch.tensor(B,dtype=torch.float32)
    ab=b-a; L2=(ab*ab).sum(1).clamp(min=1e-6); out=torch.zeros(len(P))
    for i in range(0,len(P),200000):
        p=P[i:i+200000,None,:]; t=(((p-a)*ab).sum(-1)/L2).clamp(0,1)
        d=((p-(a+t[...,None]*ab))**2).sum(-1).sqrt(); r=torch.tensor(RA)[None]*(1-t)+torch.tensor(RB)[None]*t
        out[i:i+200000]=torch.exp(-(d/r.float())**2*1.2).sum(1)  # smooth union
    return out.numpy()
TH=np.exp(-1.2)
h=0.45; lo=X.min(0)-14; hi=X.max(0)+14
gx,gy,gz=[np.arange(l,u,h) for l,u in zip(lo,hi)]
GR=np.stack(np.meshgrid(gx,gy,gz,indexing='ij'),-1)
F=field(GR.reshape(-1,3)).reshape(GR.shape[:3])-TH
pts=[];
for ax in range(3):
    f0=np.take(F,range(F.shape[ax]-1),axis=ax); f1=np.take(F,range(1,F.shape[ax]),axis=ax)
    m=(f0>0)!=(f1>0); idx=np.argwhere(m); a=f0[m]; b=f1[m]; t=a/(a-b)
    p=GR[tuple(idx.T)].copy(); p[:,ax]+=t*h; pts.append(p)
S=np.concatenate(pts)
e=0.2; Ng=np.stack([(field(S+e*np.eye(3)[k])-field(S-e*np.eye(3)[k]))/(2*e) for k in range(3)],1)
Ns=-Ng/np.linalg.norm(Ng,axis=1,keepdims=True)
print('surface points',len(S))
np.savez('surf.npz',S=S.astype(np.float32),N=Ns.astype(np.float32))
