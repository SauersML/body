import os
# Build neutral.obj: a genderless, faceless version of the CC0 MakeHuman base mesh (base.obj).
# Vertex count and order identical to base.obj (19158 v; body uses 0..13379). Only group 'body' is kept.
# Head: base.obj head kept (skull, ears, jaw, chin, cheekbones, brow, neck). Only a front patch over eyes, nose and lips
#       (plus the mouth/eye/nostril cavities behind it) is re-laid fold-free in the (x,y) plane (uniform Tutte map) and
#       lifted onto the base face's own depth map, morphologically closed/opened and blurred: a mannequin face form.
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
# ---------------- face: remove eyes, nose and lips only; skull, ears, jaw, chin, cheekbones, brow, neck stay as base.obj
Wn=sp.diags(1/np.asarray(A.sum(1)).ravel())@A
def ss(v,a,b): t=np.clip((v-a)/(b-a),0,1); return t*t*(3-2*t)
# mean value weights from the original 3-D geometry
ii=[];jj=[];ww=[]
for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
    i,j,k=T[:,a],T[:,b],T[:,c]
    e1=P[j]-P[i]; e2=P[k]-P[i]; l1=np.linalg.norm(e1,axis=1); l2=np.linalg.norm(e2,axis=1)
    ang=np.arccos(np.clip((e1*e2).sum(1)/(l1*l2+1e-12),-1,1)); t=np.tan(ang/2)
    ii+= [i,i]; jj+=[j,k]; ww+=[t/(l1+1e-12),t/(l2+1e-12)]
Wm=sp.coo_matrix((np.concatenate(ww),(np.concatenate(ii),np.concatenate(jj))),shape=(nb,nb)).tocsr()
L=sp.diags(np.asarray(Wm.sum(1)).ravel())-Wm
Ac=A.tolil().rows; Lu=sp.diags(np.asarray(A.sum(1)).ravel())-A; CLOSE=0.16; OPEN_Y=0.34; BLUR=0.07
x,y,z=P.T
# feature mask on the front face surface: a rounded patch from just under the brow to just under the lower lip
def mask_d(x,y):
    yc,ry=6.86,0.56
    t=(y-yc)/ry
    hw=0.40+0.17*ss(y,6.75,7.02)          # wider at the eyes than at the mouth
    return (np.abs(x)/hw)**4+np.abs(t)**4    # superellipse <1 inside
outer=(mask_d(x,y)<1)&(z>1.0)
# region = everything cut off from the crown by the outer patch (adds mouth and eye cavities behind it)
blocked=outer.copy(); top=np.argmax(y); seen=np.zeros(nb,bool); st=[top]; seen[top]=True
while st:
    i=st.pop()
    for j in Ac[i]:
        if not blocked[j] and not seen[j]: seen[j]=True; st.append(j)
R=~seen
print('face region',R.sum(),'outer',outer.sum(),'cavity',(R&~outer).sum())
ri=np.where(R)[0]; ro=np.where(~R)[0]; np.save(H+'work/face_region.npy',R)
XY=sla.spsolve(Lu[ri][:,ri].tocsc(), -(Lu[ri][:,ro]@P[ro][:,:2]))   # uniform Tutte: spreads the collapsed eye/mouth rings evenly
P[ri,:2]=XY
def mvw(Q):
    ii=[];jj=[];ww=[]
    for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
        i,j,k=T[:,a],T[:,b],T[:,c]
        e1=Q[j]-Q[i]; e2=Q[k]-Q[i]; l1=np.linalg.norm(e1,axis=1)+1e-9; l2=np.linalg.norm(e2,axis=1)+1e-9
        t=np.tan(np.arccos(np.clip((e1*e2).sum(1)/(l1*l2),-1,1))/2)
        ii+=[i,i]; jj+=[j,k]; ww+=[t/l1,t/l2]
    W=sp.coo_matrix((np.concatenate(ww),(np.concatenate(ii),np.concatenate(jj))),shape=(nb,nb)).tocsr()
    return sp.diags(1/np.asarray(W.sum(1)).ravel())@W
# height over the re-laid patch: the base face's own front depth map, closed (fills the eye and mouth slits) and
# blurred, so the face keeps its forms (brow, cheeks, muzzle, a soft nose) but loses eyes, nostrils and lips.
# A harmonic correction (geometric weights of the flat layout) makes it meet the untouched skin exactly.
import trimesh, scipy.ndimage as ndi
Wmn=mvw(P[:,:2]); Lg=(sp.eye(nb)-Wmn).tocsr()
mesh=trimesh.Trimesh(V[:nb],T,process=False)
h=0.01; gx=np.arange(-0.9,0.9+h,h); gy=np.arange(6.0,7.9+h,h); GX,GY=np.meshgrid(gx,gy)
org=np.c_[GX.ravel(),GY.ravel(),np.full(GX.size,5.0)]
loc,ray,_=mesh.ray.intersects_location(org,np.tile([0,0,-1.0],(len(org),1)),multiple_hits=True)
D=np.full(GX.size,np.nan); np.fmax.at(D,ray,loc[:,2]); D=D.reshape(GX.shape)
D=np.where(np.isnan(D),np.nanmin(D),D)
D=ndi.grey_closing(D,size=(int(CLOSE/h),int(CLOSE/h)))         # fills eye slits, nostrils, mouth line
D0=ndi.gaussian_filter(D,0.02/h)
D=ndi.grey_opening(D,size=(int(OPEN_Y/h),int(0.04/h)))           # removes bumps shorter than OPEN_Y in y (lips), keeps the nose ridge
D1=ndi.gaussian_filter(D,BLUR/h)
from scipy.interpolate import RegularGridInterpolator
d0=RegularGridInterpolator((gy,gx),D0,bounds_error=False,fill_value=None,method="cubic")
d1=RegularGridInterpolator((gy,gx),D1,bounds_error=False,fill_value=None,method="cubic")
# full blur in the middle of the patch, none at its edge, so it meets the untouched skin smoothly
wb=ss(1-mask_d(XY[:,0],XY[:,1]),0.0,0.6)
Zb=d0(XY[:,::-1])*(1-wb)+d1(XY[:,::-1])*wb
res=P[:,2]-d0(P[:,1::-1])
Z=Zb+sla.spsolve(Lg[ri][:,ri].tocsc(), -(Lg[ri][:,ro]@res[ro]))
P[ri,2]=Z
# a light Taubin pass across the seam
seam=np.zeros(nb); seam[ri]=1
for _ in range(5): seam=Wn@seam
seam=np.clip(seam*(1-seam)*6,0,1)[:,None]
for _ in range(25): P=P+0.5*seam*(Wn@P-P); P=P-0.52*seam*(Wn@P-P)
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
    if l.startswith('v '): out.append('v %.6f %.6f %.6f'%tuple(V2[vi])); vi+=1
    elif l.startswith('g '):
        g=l.split()[1]
        if g=='body': out.append(l)
    elif l.startswith(('f ','usemtl','s ')):
        if g=='body': out.append(l)
    else: out.append(l)
open(H+'neutral.obj','w').write('\n'.join(out))
