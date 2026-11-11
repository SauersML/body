import os
# Register all 407 spots of nn2.npz onto the MakeHuman base mesh (same topology as neutral mesh).
import numpy as np, sys, pickle
exec(open(os.path.expandvars('$BODY_REPO/figures/zoom3/register.py')).read())   # face/hand/foot/left forearm (writes ./reg.pkl)
d=np.load(os.path.expandvars('$BODY_REPO/figures/src/nn2.npz')); PP=d['P'].astype(float); NN=nz(d['N'].astype(float)); PH=list(d['ph']); RG=d['reg']; n=len(PP)
pos=np.full((n,3),np.nan); snap=np.full(n,np.nan); how=['']*n
for reg in res:
    for ph,h,ds in zip(res[reg]['phr'],res[reg]['pos'],res[reg]['dist']):
        if ph not in PH: continue
        i=PH.index(ph); pos[i]=h; snap[i]=ds; how[i]=reg
def rotmin(a,b):
    a=nz(a); b=nz(b); v=np.cross(a,b); c=a@b; vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx/(1+c)
def first_hit(o,dirn,maxd=40):
    h=cast(o[None],dirn[None])[0]
    if np.isnan(h[0]) or np.linalg.norm(h-o)>maxd: return None
    return h
S_=J['l-shoulder']; E_=J['l-elbow']; HIP=J['l-upper-leg']; KN=J['l-knee']; AN=J['l-ankle']
segs={'upper':(np.array([19,143,0.]),np.array([50,143,0.]),S_,E_),
      'thigh':(np.array([10.5,92,0.]),np.array([12,50,0.]),HIP,KN),'shin':(np.array([12,50,0.]),np.array([13,8,0.]),KN,AN)}
def seg_map(p,n,key):
    a0,a1,b0,b1=segs[key]; t=np.clip((p-a0)@(a1-a0)/((a1-a0)@(a1-a0)),0,1); R=rotmin(a1-a0,b1-b0)
    ax=a0+t*(a1-a0); off=p-ax; q=b0+t*(b1-b0)+R@off*0.0; return q, nz(R@n), np.linalg.norm(off)
sim_only=sim  # face similarity (stim head -> mesh head)
for i in range(n):
    if not np.isnan(pos[i,0]): continue
    p,nv=PP[i].copy(),NN[i].copy(); mir=p[0]<-0.01
    # right side: register the mirror point, then mirror back
    if mir: p[0]*=-1; nv[0]*=-1
    ph=PH[i].replace('right','left') if mir else PH[i]
    if mir and ph in PH and not np.isnan(pos[PH.index(ph),0]) and how[PH.index(ph)]:
        j=PH.index(ph); q=pos[j]*[-1,1,1]; h,_,_=trimesh.proximity.closest_point(body,q[None]); pos[i]=h[0]; snap[i]=snap[j]; how[i]='mirror:'+how[j]; continue
    r=RG[i]; start=None
    if r=='handL': q,dv,rr=forearm_map(np.r_[min(p[0],76.0),p[1:]],nv); rr=np.linalg.norm(p-[p[0],143,0]); kind='wrist'
    elif r=='armL' and 'armpit' in ph: q=S_-[0,2,0]; dv=np.array([0,-1.,0]); rr=5.0; kind='armpit'
    elif r=='armL' and 'forearm' not in ph and p[0]<50.5: q,dv,rr=seg_map(p,nv,'upper'); kind='upper arm'
    elif r=='armL': q,dv,rr=forearm_map(p,nv); rr=np.linalg.norm(p-[p[0],143,0]); kind='forearm'
    elif r=='legL':
        for dy in np.arange(0,12,0.5):
            pp=p-[0,dy,0]; q,dv,rr=seg_map(pp,nv,'thigh' if pp[1]>=50 else 'shin'); h=first_hit(q,dv,15)
            if h is not None and h[0]>0: break
        kind='leg'
    elif r=='head' and p[2]>2:
        q=fmap(p[None])[0]; dv=jac(fmap,p[None],nv[None])[0]; q=q-3*dv; rr=3.0; kind='head (face side)'
    elif r=='head':
        c0=np.array([0,164.,0]); q=sim_only(c0[None])[0]; dv=nz(sim_only(p[None])[0]-q); rr=np.linalg.norm(sim_only(p[None])[0]-q); kind='head'
    else:  # torso, neck: centre line of the mesh trunk at this height; ray from the centre towards the spot
        yy=p[1]+(2 if r=='neck' else 0)
        f=first_hit(np.array([0,yy,5.]),np.array([0,0,1.]),30); b_=first_hit(np.array([0,yy,5.]),np.array([0,0,-1.]),30); zc=(f[2]+b_[2])/2
        ps=np.array([p[0],yy,p[2]+zc]); q=np.array([0,yy,zc]); dv=nz(ps-q+1e-6*nv); rr=np.linalg.norm(ps-q); kind=r
    h=first_hit(q,dv)
    tgt=q+rr*dv
    if h is None: h=trimesh.proximity.closest_point(body,tgt[None])[0][0]; kind+='(fallback)'
    pos[i]=h; snap[i]=np.linalg.norm(h-tgt); how[i]=kind
    if mir: pass
# mirror right-side spots registered in this loop
for i in range(n):
    if PP[i,0]<-0.01 and not how[i].startswith('mirror') and how[i] not in ('face','hand','foot','forearm'):
        pos[i]=trimesh.proximity.closest_point(body,(pos[i]*[-1,1,1])[None])[0][0]; how[i]='mirror:'+how[i]
_,dd,tid=trimesh.proximity.closest_point(body,pos); 
B=trimesh.triangles.points_to_barycentric(body.triangles[tid],pos)
import collections
for k in sorted(set(h.split(':')[-1] for h in how)):
    m=np.array([h.split(':')[-1]==k for h in how]); print(f"{k:22s} n={m.sum():3d} snap cm median {np.median(snap[m]):.2f}  max {snap[m].max():.2f}  worst: {PH[np.where(m)[0][np.argmax(snap[m])]]}")
print("ALL median %.2f  90pct %.2f  max %.2f"%(np.median(snap),np.percentile(snap,90),snap.max()))
np.savez("reg_all.npz",pos=pos,tid=tid,bary=B,snap=snap,how=np.array(how))
