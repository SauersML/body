import os
import sys,numpy as np, trimesh
sys.path.insert(0,os.path.expandvars("$BODY_REPO/figures/zoom3"))
from load import *
K=10.74
out={}
for nm,path in [("base",os.path.expandvars("$BODY_REPO/figures/zoom3/base.obj")),("neutral",os.path.expandvars("$BODY_REPO/mesh/neutral.obj"))]:
    V,G=load(path); V=(V-[0,-8.1676,0])*K
    F=tri(G['body']); used=np.unique(F); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used))
    m=trimesh.Trimesh(V[used],remap[F],process=True)
    v,f=trimesh.remesh.subdivide_loop(m.vertices,m.faces,iterations=2)
    b=trimesh.Trimesh(v,f,process=True); out[nm]=b; print(nm,b.vertices.shape,b.faces.shape)
z=np.load(os.path.expandvars("$BODY_REPO/figures/zoom3/mesh.npz"))
print("base matches zoom3 mesh:",np.allclose(out['base'].vertices,z['v']),(out['base'].faces==z['f']).all())
print("same faces:",(out['base'].faces==out['neutral'].faces).all())
np.savez("neutral_mesh.npz",v=out['neutral'].vertices,f=out['neutral'].faces)
