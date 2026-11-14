import numpy as np, importlib.util
from scipy.ndimage import gaussian_filter1d
C=np.load('C.npy'); g=np.load('geo.npz'); arm=g['arm']; d=g['d']
x,y,z=C.T
sp=importlib.util.spec_from_file_location('s','$BODY_WORK/body/hires2/stim.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
L=m.LOCS; SP=np.array([l['xyz'] for l in L]); SR=np.array([l['region'] for l in L])
U=np.full_like(C,np.nan)
part=np.zeros(len(C),int)  # 0 trunk,1 armL,2 armR,3 legL,4 legR
leg=(~arm)&(y<86)&(np.abs(x)>0.5)
part[arm&(x>0)]=1; part[arm&(x<0)]=2; part[leg&(x>0)]=3; part[leg&(x<0)]=4
# ---- arms
def nrm(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
WY=np.array([0,1.,0])
for s,p in [(1,1),(-1,2)]:
    A=np.where(part==p)[0]; dA=d[A]; dmax=np.percentile(dA,99.5)
    bins=np.arange(0,dmax+1,1.0); cen=np.array([C[A[(dA>=b-2)&(dA<b+2)]].mean(0) for b in bins]); bad=np.isnan(cen[:,0]); cen[bad]=cen[~bad][np.searchsorted(bins[~bad],bins[bad]).clip(0,(~bad).sum()-1)]
    rad=np.array([np.linalg.norm(C[A[(dA>=b-1.5)&(dA<b+1.5)]]-cen[i],axis=1).mean() for i,b in enumerate(bins)])
    rad=np.nan_to_num(rad,nan=1e9); iw=np.argmin(np.where((bins>dmax*0.55)&(bins<dmax*0.8),rad,1e9)); dw=bins[iw]
    cs=gaussian_filter1d(cen,2,axis=0); tan=nrm(np.gradient(cs,axis=0))
    up0=nrm(WY-(tan@WY)[:,None]*tan); fr0=s*np.cross(tan,up0)
    # palm normal from hand
    H=A[(dA>dw+3)&(dA<dw+10)]; Hc=C[H]-C[H].mean(0); n=np.linalg.svd(Hc,full_matrices=False)[2][2]
    tips=C[A[dA>dmax-4]].mean(0); n=n if (tips-C[H].mean(0))@n>0 else -n
    tw=tan[iw]; n=nrm(n-(n@tw)*tw); upw=s*np.cross(n,tw)
    phi=np.arctan2(upw@fr0[iw],upw@up0[iw])
    de=dw*30/60.
    f=np.clip((bins-de)/(dw-de),0,1); ang=f*phi
    up=np.cos(ang)[:,None]*up0+np.sin(ang)[:,None]*fr0; fr=s*np.cross(tan,up)
    print('arm',s,'dw',dw,'dmax',dmax,'twist deg',np.degrees(phi).round(1),'wrist r',rad[iw].round(2))
    for v in A:
        dv=d[v]
        if dv<=dw:
            i=int(round(dv)); X=16+dv/dw*60; q=C[v]-cs[i]
            rs={True:5.,False:4.}[X<50]*(1 if X<70 else .8); sc=rs/max(rad[i],1.5)
            U[v]=[s*X,143+sc*(q@up[i]),sc*(q@fr[i])]
        else:
            X=76+(dv-dw)/(dmax-dw)*24; q=C[v]-cs[iw]
            U[v]=[s*min(X,100),143+1.1*(q@up[iw]),np.clip(q@fr[iw],-3,3)]
# ---- trunk & legs: per-height bbox map
def bbox_map(mask_mesh, mask_stim, ys):
    M=C[mask_mesh]; S=SP[mask_stim]; out=np.zeros_like(M)
    def prof(Q,yy,win):
        r=[]
        for yv in yy:
            s=Q[np.abs(Q[:,1]-yv)<win]
            if len(s)<2: r.append([np.nan]*4); continue
            r.append([np.percentile(s[:,0],2),np.percentile(s[:,0],98),np.percentile(s[:,2],2),np.percentile(s[:,2],98)])
        r=np.array(r)
        for k in range(4):
            bad=np.isnan(r[:,k]); r[bad,k]=np.interp(yy[bad],yy[~bad],r[~bad,k])
        return gaussian_filter1d(r,3,axis=0)
    pm=prof(M,ys,1.5); ps=prof(S,ys,7)
    for i,(px,py,pz) in enumerate(M):
        j=np.clip(np.searchsorted(ys,py),0,len(ys)-1); a=pm[j]; b=ps[j]
        X=b[0]+(px-a[0])/max(a[1]-a[0],1e-3)*(b[1]-b[0]); Z=b[2]+(pz-a[2])/max(a[3]-a[3*0+2],1e-3)*(b[3]-b[2])
        out[i]=[X,py,Z]
    return out
ys=np.arange(0,180,1.)
ym=y.copy()
for p,sr in [(3,['legL','footL']),(4,['legR','footR'])]:
    mm=part==p; U[mm]=bbox_map(mm,np.isin(SR,sr),ys)
# leg: mesh crotch ~86 -> stim leg top 98: stretch leg heights
for p in (3,4):
    mm=part==p; U[mm,1]=U[mm,1]*98/86.
mm=part==0; Ut=bbox_map(mm,np.isin(SR,['torso','head','neck']),ys); U[mm]=Ut
mm2=mm&(y<98); U[mm2,1]=98+(U[mm2,1]-86)*(8/12.)  # pelvis band
np.savez('unposed.npz',U=U,part=part,SP=SP,SR=SR)
print('nan',np.isnan(U).any(1).sum(), [ (part==k).sum() for k in range(5)])
