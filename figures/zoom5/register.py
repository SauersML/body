import os
import numpy as np, trimesh, pickle, json, sys
sys.path.insert(0,os.path.expandvars("$BODY_REPO/figures/src"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from scipy.interpolate import RBFInterpolator
M=np.load("mesh.npz"); body=trimesh.Trimesh(M['v'],M['f'],process=False); J=pickle.load(open("joints.pkl","rb"))
VN=body.vertex_normals; RI=trimesh.ray.ray_triangle.RayMeshIntersector(body)
def nz(v): v=np.asarray(v,float); return v/np.linalg.norm(v,axis=-1,keepdims=True)
def cast(o,d):
    """first hit of rays o+t d, t>0"""
    loc,ir,it=RI.intersects_location(o,d,multiple_hits=True); out=np.full((len(o),3),np.nan)
    for i in range(len(o)):
        m=ir==i
        if m.any(): L=loc[m]; t=((L-o[i])*d[i]).sum(1); L=L[t>0]; t=t[t>0]
        if m.any() and len(t): out[i]=L[np.argmin(t)]
    return out
def snap_ray(p,n,back=3.0):
    """line through p along n: take the surface crossing nearest to p (within back cm); fallback nearest point"""
    loc,ir,it=RI.intersects_location(p-back*n,n,multiple_hits=True); h=np.full((len(p),3),np.nan)
    for i in range(len(p)):
        m=ir==i
        if m.any():
            d=np.linalg.norm(loc[m]-p[i],axis=1); 
            if d.min()<back: h[i]=loc[m][np.argmin(d)]
    bad=np.isnan(h[:,0])
    if bad.any():
        q,_,_=trimesh.proximity.closest_point(body,p[bad]); h[bad]=q
    return h
def umeyama(A,B):
    ma,mb=A.mean(0),B.mean(0); a,b=A-ma,B-mb; U,S,Vt=np.linalg.svd(b.T@a); D=np.eye(3); D[2,2]=np.sign(np.linalg.det(U@Vt))
    R=U@D@Vt; s=(S*np.diag(D)).sum()/(a**2).sum(); return lambda x: s*(np.asarray(x)-ma)@R.T+mb, R
Bv=body.vertices
def pick(mask,score): i=np.where(mask)[0]; return Bv[i[np.argmax(score[i])]]
out={}
# ---------- face: landmark similarity + TPS, then ray along mapped normal
x,y,z=Bv.T
nose=pick((abs(x)<.3)&(y>158)&(y<165),z)
fore=pick((abs(x)<.3)&(abs(y-172.5)<.4),z)
chin=pick((abs(x)<.3)&(abs(y-153.6)<.3),z)
ulip=pick((abs(x)<.3)&(abs(y-158.6)<.3),z); llip=pick((abs(x)<.3)&(abs(y-156.9)<.3),z)
eyeL=(J['l-upperlid']+J['l-lowerlid'])/2; eyeR=(J['r-upperlid']+J['r-lowerlid'])/2
mc=(abs(y-157.6)<.3)&(z>13); mcL=pick(mc&(x>0)&(x<3.2),z*0+x*(x<3.0)); mcR=mcL*[-1,1,1]
A=np.array([[0,161,12.5],[0,172,9.5],[0,152.5,9],[0,158,10.5],[0,156,10.3],[3.5,166,9.5],[-3.5,166,9.5]]); B=np.array([nose,fore,chin,ulip,llip,eyeL,eyeR])
print("face landmarks mesh",B.round(1))
sim,R=umeyama(A,B); tps=RBFInterpolator(A,B-sim(A),kernel='thin_plate_spline')
fmap=lambda p: sim(p)+tps(p)
# ---------- foot
fo=(y<12)&(x>15); 
heel=pick(fo&(y>2)&(y<5),-z); btoe=pick(fo&(x<23),z); ltoe=pick(fo&(x>27.3),z)
ank_in=pick(fo&(abs(y-8)<.5)&(abs(z)<1.5),-x); ank_out=pick(fo&(abs(y-8)<.5)&(abs(z)<1.5),x)
ball=pick(fo&(abs(z-13)<1)&(x<23),-y)
A2=np.array([[13,3,-5],[10,2,21],[16.8,1.5,17],[9,8,0],[17,8,0],[12,0,14]]); B2=np.array([heel,btoe,ltoe,ank_in,ank_out,ball])
print("foot landmarks mesh",B2.round(1))
sim2,_=umeyama(A2,B2); tps2=RBFInterpolator(A2,B2-sim2(A2),kernel='thin_plate_spline'); pmap=lambda p: sim2(p)+tps2(p)
def jac(f,p,n):
    e=1e-2; Jm=np.stack([(f(p+e*np.eye(3)[k])-f(p-e*np.eye(3)[k]))/(2*e) for k in range(3)],2)  # (N,3,3)
    return nz(np.einsum('nij,nj->ni',np.linalg.inv(Jm).transpose(0,2,1),n))
# ---------- hand / forearm frames
W=J['l-hand']; mcp={F:J[f'l-finger-{k}-1'] for F,k in [('index',2),('middle',3),('ring',4),('little',5)]}
ex=nz(mcp['middle']-W); side=mcp['index']-mcp['little']; ey=nz(side-ex*(side@ex)); ez=np.cross(ex,ey)  # ez = palm normal (palm forward in our frame)
Rh=np.stack([ex,ey,ez],1)   # our-frame dir -> mesh dir
E=J['l-elbow']; ax=nz(W-E); fy=nz(ey-ax*(ey@ax)); fz=np.cross(ax,fy); Rf=np.stack([ax,fy,fz],1)
print("palm normal",ez.round(2),"forearm axis",ax.round(2))
def hand_map(p,n):
    """palm points: affine in-plane (wrist, index MCP, little MCP), start at mid-plane, cast along mapped normal"""
    p=np.array(p,float); n=np.array(n,float)
    A3=np.array([[76,143],[91,146],[91,135.5],[91,142.5]]); B3=np.array([W,mcp['index'],mcp['little'],mcp['middle']])
    X=np.c_[A3,np.ones(4)]; C=np.linalg.lstsq(X,B3,rcond=None)[0]; q=np.r_[p[:2],1]@C
    d=nz(Rh@n); return q-1.5*d, d, (abs(p[2]) if abs(n[2])>.5 else 0.0)+1.5
FING={'index':(2,146,8),'middle':(3,142.5,9),'ring':(4,139,8.5),'little':(5,135.5,6.5)}
def finger_map(F,p,n):
    k,yy,Lf=FING[F]; j=[J[f'l-finger-{k}-{i}'] for i in range(1,5)]; tip=j[2]+0.8*(j[3]-j[2])
    knots=[0,.45,.75,1.0]; pts=[j[0],j[1],j[2],tip]; u=np.clip((p[0]-91)/Lf,0,1)
    s=np.searchsorted(knots,u,side='right')-1; s=min(s,2); w=(u-knots[s])/(knots[s+1]-knots[s]); q=pts[s]*(1-w)+pts[s+1]*w
    d=nz(pts[s+1]-pts[s]); sy=nz(ey-d*(ey@d)); sz=np.cross(d,sy); R=np.stack([d,sy,sz],1)
    return q, nz(R@n), np.linalg.norm([p[1]-yy,p[2]])+max(0,(p[0]-91-Lf))
def thumb_map(p,n):
    j=[J[f'l-finger-1-{i}'] for i in range(1,5)]; tip=j[2]+0.8*(j[3]-j[2])
    a0=np.array([83.5,150.5]); a1=np.array([87,157.8]); dd=a1-a0; u=np.clip(((np.array(p[:2])-a0)@dd)/(dd@dd),0,1)
    knots=[0,.3,.65,1.0]; pts=[j[1],j[2],j[3],tip]; s=min(np.searchsorted(knots,u,side='right')-1,2); w=(u-knots[s])/(knots[s+1]-knots[s]); q=pts[s]*(1-w)+pts[s+1]*w
    d=nz(pts[s+1]-pts[s]); ours_d=np.r_[nz(dd),0]; ours_side=np.array([1,0,0.])-ours_d*ours_d[0]; ours_side=nz(ours_side); ours_z=np.cross(ours_d,ours_side)
    # mesh thumb frame: axis d, palm-ish normal = ez projected
    mz=nz(ez-d*(ez@d)); ms=np.cross(mz,d)
    nn=np.array([n@ours_d,n@ours_side,n@ours_z]); ax0=np.r_[a0+u*dd,0]; pp=np.array(p,float); pp[2]=pp[2]; r=np.linalg.norm(pp-ax0 - ours_d*((pp-ax0)@ours_d)) if u<1 else np.linalg.norm(pp-ax0)
    return q, nz(nn[0]*d+nn[1]*ms+nn[2]*mz), r
def forearm_map(p,n):
    t=(p[0]-50)/26; q=E+t*(W-E); return q, nz(Rf@np.array(n,float)), 4.0
res={}
for reg in ["face","hand","foot","forearm"]:
    ix=REG[reg]; P0=P[ix].astype(float); N0=nz(N[ix])
    if reg=="face":
        Q=fmap(P0); Nq=jac(fmap,P0,N0); H=snap_ray(Q,Nq,3.0); start=Q
    elif reg=="foot":
        Q=pmap(P0); Nq=jac(pmap,P0,N0); H=snap_ray(Q,Nq,3.0); start=Q
    else:
        Q=[];Nq=[];Rr=[]
        for i in ix:
            ph=L[i]["phrase"]; p=np.array(L[i]["xyz"],float); n=nz(L[i]["n"])
            if reg=="forearm": q,d,r=forearm_map(p,n)
            elif "thumb" in ph and "base of" not in ph and "web" not in ph and "facing the thumb" not in ph: q,d,r=thumb_map(p,n)
            else:
                F=[f for f in FING if f"{f} finger" in ph or f"{f} fingertip" in ph]
                if F and "skin between" not in ph and "edge" not in ph: q,d,r=finger_map(F[0],p,n)
                else: q,d,r=hand_map(p,n)
            Q.append(q);Nq.append(d);Rr.append(r)
        Q=np.array(Q);Nq=np.array(Nq);Rr=np.array(Rr); Qs=Q+Rr[:,None]*Nq; H=cast(Q,Nq)
        bad=np.isnan(H[:,0])|(np.linalg.norm(H-Qs,axis=1)>4)
        if bad.any(): print('fallback',[res_ph for res_ph,b in zip([L[i]['phrase'] for i in ix],bad) if b]); H[bad]=trimesh.proximity.closest_point(body,Qs[bad])[0]
        Q=Qs
    # snap distance: from the registered point to the surface it lands on (for axis-ray regions: the gap between our idealised skin radius and the mesh skin)
    if reg in("face","foot"): dist=np.linalg.norm(H-Q,axis=1)
    else:
        r_ours=np.array([np.linalg.norm(np.array(L[i]["xyz"])-[L[i]["xyz"][0],143,0]) if reg=="forearm" else 0 for i in ix])
        dist=np.linalg.norm(H-Q,axis=1)
    _,_,tid=trimesh.proximity.closest_point(body,H); nrm=body.face_normals[tid]
    res[reg]=dict(pos=H,nrm=nrm,dist=dist,phr=[L[i]["phrase"] for i in ix])
    print(f"{reg:8s} n={len(ix)} snap cm: median {np.median(dist):.2f} max {dist.max():.2f}", "worst:",res[reg]['phr'][int(np.argmax(dist))].replace('{p}','the'))
pickle.dump(dict(res=res,frames=dict(Rh=Rh,Rf=Rf,W=W,E=E,ez=ez,ey=ey,ex=ex)),open("reg.pkl","wb"))
if __name__=="__main__" and len(sys.argv)>1:
    for ph in ['the side of {p} left index finger facing the thumb','the little-finger edge of {p} left hand']:
        i=[k for k in REG['hand'] if L[k]['phrase']==ph][0]; p=np.array(L[i]['xyz'],float); n=nz(L[i]['n'])
        q,d,r=finger_map('index',p,n) if 'index' in ph else hand_map(p,n)
        print(ph,q.round(2),d.round(2),r, trimesh.proximity.signed_distance(body,[q]))
        loc,ir,it=RI.intersects_location([q],[d],multiple_hits=True); print(loc.round(2), np.linalg.norm(loc-q,axis=1).round(2))
