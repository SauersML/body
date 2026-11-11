import os
import sys,numpy as np, trimesh
sys.path.insert(0,os.path.expandvars("$BODY_REPO/figures/zoom3"))
from load import *
K=10.74
V,G=load(os.path.expandvars("$BODY_REPO/mesh/neutral.obj")); V=(V-[0,-8.1676,0])*K
F=tri(G['body']); used=np.unique(F); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used))
m=trimesh.Trimesh(V[used],remap[F],process=True)
v,f=trimesh.remesh.subdivide_loop(m.vertices,m.faces,iterations=2)
b=trimesh.Trimesh(v,f,process=True)
old=np.load("../nb4/neutral_mesh.npz"); print(b.vertices.shape, old['v'].shape, (b.faces==old['f']).all())
dv=np.linalg.norm(b.vertices-old['v'],axis=1); print("moved verts",(dv>0.01).sum(), dv.max())
R=np.load("../nb4/reg_all.npz"); d=np.load('../src/nn2.npz')
pos=(b.vertices[b.faces[R['tid']]]*R['bary'][:,:,None]).sum(1)
old_sp=np.load("../nb4/spot_pos_neutral.npy")
RI=trimesh.ray.ray_triangle.RayMeshIntersector(b); C0=np.array([0,164.0,3.0])
for i in np.where(d['reg']=='head')[0]:
    dd=R['pos'][i]-C0; dd/=np.linalg.norm(dd); loc,_,_=RI.intersects_location([C0],[dd]); t=(loc-C0)@dd; pos[i]=loc[np.argmin(np.where(t>0,t,1e9))]
print("head spot shift vs bary-on-base: median",np.median(np.linalg.norm(pos-R['pos'],axis=1)[d['reg']=='head']))
np.save("spot_pos.npy",pos)
b.vertices.astype(np.float32).tofile("geo_pos.bin"); b.vertex_normals.astype(np.float32).tofile("geo_nrm.bin"); b.faces.astype(np.uint32).tofile("geo_idx.bin")
SKIN=np.array([0.90,0.885,0.87]); (np.tile(SKIN,(len(b.vertices),1))*255).round().astype(np.uint8).tofile("col_plain.bin")
# visibility from front (+z) and back (-z): ray toward camera
_,_,fi=trimesh.proximity.closest_point(b,pos); nz=b.face_normals[fi][:,2]
vis={}
for s in (1,-1):
    o=pos+np.array([0,0,s*0.3]); hit=RI.intersects_any(o,np.tile([0,0,s],(len(pos),1)))
    vis[s]=~hit
np.savez("vis.npz",nz=nz,front=vis[1],back=vis[-1])
print("front vis",vis[1].sum(),"back vis",vis[-1].sum(),"neither",(~vis[1]&~vis[-1]).sum())
print(d['ph'][~vis[1]&~vis[-1]])
