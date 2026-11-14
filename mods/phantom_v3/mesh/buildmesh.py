import os
import sys, numpy as np, trimesh, pickle
sys.path.insert(0,os.path.expandvars('$BODY_REPO/figures/zoom3'))
from load import load, tri, joints
K=10.74
V,G=load(os.path.expandvars('$BODY_REPO/mesh/base.obj')); V=(V-[0,-8.1676,0])*K; J=joints(V,G)
Vn,Gn=load(os.path.expandvars('$BODY_REPO/mesh/neutral.obj')); Vn=(Vn-[0,-8.1676,0])*K
F=tri(Gn['body']); used=np.unique(F); remap=-np.ones(len(Vn),int); remap[used]=np.arange(len(used))
m=trimesh.Trimesh(Vn[used],remap[F],process=True)
v,f=trimesh.remesh.subdivide_loop(m.vertices,m.faces,iterations=2)
b=trimesh.Trimesh(v,f,process=True); print(b.vertices.shape, b.vertices.min(0), b.vertices.max(0))
np.savez('mesh.npz',v=b.vertices,f=b.faces); pickle.dump(J,open('joints.pkl','wb'))
