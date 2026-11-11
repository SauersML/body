import numpy as np, trimesh
from load import *
K=10.74
def build():
    V,G=load(); V=(V-[0,-8.1676,0])*K; J=joints(V,G)
    F=tri(G['body']); used=np.unique(F); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used))
    m=trimesh.Trimesh(V[used],remap[F],process=True)
    v,f=trimesh.remesh.subdivide_loop(m.vertices,m.faces,iterations=2)
    body=trimesh.Trimesh(v,f,process=True)
    eyes=[]
    for s in 'lr':
        Fe=tri(G[f'helper-{s}-eye']); u=np.unique(Fe); rm=-np.ones(len(V),int); rm[u]=np.arange(len(u))
        e=trimesh.Trimesh(V[u],rm[Fe]); v2,f2=trimesh.remesh.subdivide_loop(e.vertices,e.faces,2); eyes.append(trimesh.Trimesh(v2,f2))
    return body,eyes,J
if __name__=="__main__":
    b,e,J=build(); print(b.vertices.shape,b.faces.shape, b.is_watertight, [x.vertices.shape for x in e])
    np.savez("mesh.npz",v=b.vertices,f=b.faces,ev0=e[0].vertices,ef0=e[0].faces,ev1=e[1].vertices,ef1=e[1].faces)
    import pickle; pickle.dump(J,open("joints.pkl","wb"))
