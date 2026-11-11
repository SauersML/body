import os
# Build neutral.obj: a genderless, faceless version of the CC0 MakeHuman base mesh (base.obj).
# Vertex count and order identical to base.obj (19158 v; body uses 0..13379). Only group 'body' is kept.
# Head: re-laid onto a smooth egg + neck (mean-value Tutte map in a stereographic chart, so no folds);
#       the face/ear/mouth/eye-socket faces are kept but their vertices now lie on the smooth egg.
# Chest and groin: biharmonic fill (smooth, C1 with the surrounding skin).
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, collections
H=os.path.expandvars('$BODY_REPO/mesh/')
lines=open(H+'base.obj').read().split('\n')
V=[];F=[];g=None
for l in lines:
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V); Fq=np.array(F); nb=Fq.max()+1; P=V[:nb].copy()
T=np.vstack([Fq[:,[0,1,2]],Fq[:,[0,2,3]]])
E=np.vstack([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]])
A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(nb,nb)); A=((A+A.T)>0).astype(float).tocsr()
x,y,z=P.T
# ---------------- head
YC=6.02
cand=y>YC
# connected component of cand containing the crown
top=np.argmax(y); seen=np.zeros(nb,bool); st=[top]; seen[top]=True
Ac=A.tolil().rows
while st:
    i=st.pop()
    for j in Ac[i]:
        if cand[j] and not seen[j]: seen[j]=True; st.append(j)
free=seen
ring=np.zeros(nb,bool)
for i in np.where(free)[0]:
    for j in Ac[i]:
        if not free[j]: ring[j]=True
print('head free',free.sum(),'ring',ring.sum(), 'ring y',P[ring,1].min(),P[ring,1].max())
C=np.array([0.0,7.32,0.52])
# implicit egg + neck
EGG=dict(ax=0.80,ay_up=1.17,ay_dn=1.10,az=0.96)
rp=P[ring]; NC=np.array([0.0,0,(rp[:,2].max()+rp[:,2].min())/2]); NR=np.array([(rp[:,0].max()-rp[:,0].min())/2,(rp[:,2].max()-rp[:,2].min())/2])
print('neck',NC,NR)
def f_egg(p):
    u=p-C; ay=np.where(u[:,1]>0,EGG['ay_up'],EGG['ay_dn'])
    t=np.clip(-u[:,1]/EGG['ay_dn'],0,1); tap=1-0.22*t*t   # narrower toward the chin
    q=np.sqrt((u[:,0]/(EGG['ax']*tap))**2+(u[:,1]/ay)**2+(u[:,2]/(EGG['az']*(1-0.12*t*t)))**2)
    return (q-1)*0.8
def f_neck(p):
    # neck leans forward slightly: centre z follows height
    zc=NC[2]+0.10*(p[:,1]-YC)
    q=np.sqrt((p[:,0]/NR[0])**2+((p[:,2]-zc)/NR[1])**2)
    return np.maximum((q-1)*min(NR), p[:,1]-(C[1]-0.2))
def f(p,k=0.22):
    a,b=f_egg(p),f_neck(p); h=np.clip(0.5+0.5*(b-a)/k,0,1)
    return b*(1-h)+a*h-k*h*(1-h)
# chart: stereographic from the south pole (-y) about C
def to_chart(p):
    d=p-C; d/=np.linalg.norm(d,axis=1)[:,None]
    th=np.arccos(np.clip(d[:,1],-1,1)); ph=np.arctan2(d[:,2],d[:,0]); r=np.tan(th/2)
    return np.c_[r*np.cos(ph),r*np.sin(ph)]
def from_chart(q):
    r=np.linalg.norm(q,axis=1); th=2*np.arctan(r); ph=np.arctan2(q[:,1],q[:,0])
    return np.c_[np.sin(th)*np.cos(ph),np.cos(th),np.sin(th)*np.sin(ph)]
# mean value weights from the original 3-D geometry
ii=[];jj=[];ww=[]
for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
    i,j,k=T[:,a],T[:,b],T[:,c]
    e1=P[j]-P[i]; e2=P[k]-P[i]; l1=np.linalg.norm(e1,axis=1); l2=np.linalg.norm(e2,axis=1)
    ang=np.arccos(np.clip((e1*e2).sum(1)/(l1*l2+1e-12),-1,1)); t=np.tan(ang/2)
    ii+= [i,i]; jj+=[j,k]; ww+=[t/(l1+1e-12),t/(l2+1e-12)]
Wm=sp.coo_matrix((np.concatenate(ww),(np.concatenate(ii),np.concatenate(jj))),shape=(nb,nb)).tocsr()
fi=np.where(free)[0]; bi=np.where(~free)[0]
L=sp.diags(np.asarray(Wm.sum(1)).ravel())-Wm
Q=to_chart(P)
Qf=sla.spsolve(L[fi][:,fi].tocsc(), -(L[fi][:,bi]@Q[bi]))
D=from_chart(Qf)
lo=np.zeros(len(fi)); hi=np.full(len(fi),3.0)
for _ in range(50):
    mid=(lo+hi)/2; inside=f(C+mid[:,None]*D)<0; lo=np.where(inside,mid,lo); hi=np.where(inside,hi,mid)
P[fi]=C+lo[:,None]*D
# soften the seam just below the ring (a few Taubin passes on the lower neck band)
Wn=sp.diags(1/np.asarray(A.sum(1)).ravel())@A
def ss(v,a,b): t=np.clip((v-a)/(b-a),0,1); return t*t*(3-2*t)
band=(ss(y,5.5,5.85)*(1-ss(y,6.3,6.6)))[:,None]
for _ in range(40): P=P+0.5*band*(Wn@P-P); P=P-0.52*band*(Wn@P-P)
# ---------------- chest and groin: biharmonic fill (smooth, C1 with the surrounding skin)
Lu=sp.diags(np.asarray(A.sum(1)).ravel())-A; L2=(Lu@Lu).tocsr()
def fill(region):
    f_=np.where(region)[0]; b_=np.where(~region)[0]
    P[f_]=sla.spsolve(L2[f_][:,f_].tocsc(), -(L2[f_][:,b_]@P[b_])); return region.sum()
from scipy.interpolate import RBFInterpolator
x,y,z=P.T
rc=np.sqrt((x/1.75)**2+((y-3.55)/1.05)**2)
ringc=(((rc>1.0)&(rc<1.35))|(abs(x)<0.12))&(z>0.45)&(y<4.7)&(y>2.0)
fit=RBFInterpolator(P[ringc][:,:2],P[ringc][:,2],kernel='thin_plate_spline',smoothing=0.02)
# unfold the breast overhang: re-lay the chest patch in (x,y) with mean-value weights (fold-free, boundary = ellipse),
# then set z from the spline fitted to the surrounding ribcage, sternum and belly
R=(rc<1.0)&(z>0.3)
ri=np.where(R)[0]; ro=np.where(~R)[0]
XY=sla.spsolve(L[ri][:,ri].tocsc(), -(L[ri][:,ro]@P[ro][:,:2]))
P[ri,:2]=XY
P[ri,2]=fit(XY)
cb=R.astype(float); idx=ri
print('chest',len(idx),ringc.sum())
x,y,z=P.T
groin=(np.sqrt((x/0.8)**2+((y+0.15)/0.6)**2)<1)&(z>0.3)
print('groin',fill(groin))
gb=groin.astype(float)
# mild androgynous hips: narrow the pelvis a little
x,y,z=P.T
hip=ss(y,-3.0,-1.8)*(1-ss(y,0.4,1.4))*(1-ss(abs(x),2.1,2.6))
P[:,0]=x*(1-0.05*hip)
# flatter seat: pull the buttocks in a little
bt=ss(y,-2.6,-1.6)*(1-ss(y,-0.2,0.6))*(1-ss(abs(x),1.6,2.4))*(1-ss(z,-0.1,0.3))
P[:,2]=P[:,2]+(0.3-P[:,2])*0.22*bt
# relax seams of the edits (Taubin, volume-preserving)
x,y,z=P.T
sm=np.clip(bt*(1-bt)*4+cb*(1-cb)*4+0.25*cb,0,1)[:,None]
wb=(ss(y,1.9,2.3)*(1-ss(y,3.1,3.5))*(1-ss(abs(x),1.9,2.4)))
sm=np.maximum(sm,wb[:,None])
for _ in range(30): P=P+0.5*sm*(Wn@P-P); P=P-0.52*sm*(Wn@P-P)
V2=V.copy(); V2[:nb]=P
np.save(H+'work/moved.npy',np.linalg.norm(V2-V,axis=1))
out=[]; vi=0; g=None
for l in lines:
    if l.startswith('v '): out.append('v %.4f %.4f %.4f'%tuple(V2[vi])); vi+=1
    elif l.startswith('g '):
        g=l.split()[1]
        if g=='body': out.append(l)
    elif l.startswith(('f ','usemtl','s ')):
        if g=='body': out.append(l)
    else: out.append(l)
open(H+'neutral.obj','w').write('\n'.join(out))
