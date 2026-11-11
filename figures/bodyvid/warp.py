import numpy as np, sys
from rend import colours, smooth
g=np.load('geo.npz'); X=g['body']; Y=g['Y']; side=g['side']; sd=g['sd']; cat=g['cat']
u0=np.median(Y[side==0,0]); u=Y[:,0]-u0
Yu=np.c_[np.where(side==0,u,sd*u),Y[:,1],Y[:,2]]     # model map, unfolded with the model's own left/right sign
D=Yu-X
s=np.load('surf.npz'); S=s['S'].astype(float); N=s['N'].astype(float)
SIG=7.0
def sidew(p):  # smooth: spots on the far side of the midline fade out over a few cm
    return np.where(side[None]==0,1.0,1/(1+np.exp(-side[None]*p[:,0:1]/2.5)))+1e-12
def disp(P):
    out=np.zeros_like(P)
    for i in range(0,len(P),20000):
        p=P[i:i+20000]; d2=((p[:,None,:]-X[None])**2).sum(-1); w=np.exp(-d2/(2*SIG**2))
        w*=sidew(p)
        out[i:i+20000]=(w@D)/w.sum(1,keepdims=True)
    return out
def jac(P,e=0.3): return np.stack([(disp(P+e*np.eye(3)[k])-disp(P-e*np.eye(3)[k]))/(2*e) for k in range(3)],2)
if __name__=="__main__":
    from scipy.spatial import cKDTree
    for it in range(2):   # fill in where the warp stretches the skin, so the splats stay closed
        J=jac(S); sv=np.linalg.svd(np.eye(3)[None]+J,compute_uv=False)[:,0]
        thr=max(2.2,np.percentile(sv,75)) if it==0 else thr; hot=np.where(sv>thr)[0]; print('sv pct',np.percentile(sv,[25,50,75,90,99]).round(2)); tr=cKDTree(S); _,nb=tr.query(S[hot],k=5)
        new=[];nn=[]
        for fr in ((0.5,) if it else (1/3,2/3)):
            Pm=S[hot,None,:]*(1-fr)+S[nb[:,1:]]*fr; Nm=N[hot,None,:]*(1-fr)+N[nb[:,1:]]*fr
            new.append(Pm.reshape(-1,3)); nn.append(Nm.reshape(-1,3))
        S=np.concatenate([S]+new); N=np.concatenate([N]+nn); N/=np.linalg.norm(N,axis=1,keepdims=True)
        S,ui=np.unique(np.round(S,2),axis=0,return_index=True); N=N[ui]
        print('pass',it,'hot',len(hot),'points',len(S))
    np.savez('surf_dense.npz',S=S.astype(np.float32),N=N.astype(np.float32))
    Dv=disp(S); e=0.3
    J=np.stack([(disp(S+e*np.eye(3)[k])-disp(S-e*np.eye(3)[k]))/(2*e) for k in range(3)],2)  # dD/dx
    # surface colours: kernel blend of spot colours
    cols=colours(cat,side); out=np.zeros((len(S),3))
    for i in range(0,len(S),20000):
        p=S[i:i+20000]; d2=((p[:,None,:]-X[None])**2).sum(-1); w=np.exp(-d2/(2*3.5**2))+1e-30
        w*=sidew(p)
        out[i:i+20000]=(w@cols)/w.sum(1,keepdims=True)
    np.savez('warp.npz',Dv=Dv.astype(np.float32),J=J.astype(np.float32),col=out.astype(np.float32),Yu=Yu)
    # honesty check: how far the smoothed field is from each spot's own model position
    Ds=disp(X); print('spot fit: median miss %.1f cm vs median move %.1f cm'%(np.median(np.linalg.norm(Ds-D,axis=1)),np.median(np.linalg.norm(D,axis=1))))
