import sys, numpy as np, mujoco, trimesh
sys.path.insert(0,'$BODY_WORK/body3d')
import os; os.chdir('$BODY_WORK/body3d')
import hum as H, env as E
os.chdir('$BODY_WORK/fig/bodyart')
d=mujoco.MjData(H.M); d.qpos[H.JQ]=E.Q0; mujoco.mj_forward(H.M,d)
TW,_=H.taxel_world(d)
meshes=[]
for g in H.BODY_GEOMS:
    t=H.M.geom_type[g]; sz=H.M.geom_size[g]
    if t==mujoco.mjtGeom.mjGEOM_CAPSULE: m=trimesh.creation.capsule(height=2*sz[1],radius=sz[0],count=[48,48])
    elif t==mujoco.mjtGeom.mjGEOM_SPHERE: m=trimesh.creation.icosphere(4,radius=sz[0])
    elif t==mujoco.mjtGeom.mjGEOM_ELLIPSOID: m=trimesh.creation.icosphere(4,radius=1.0); m.vertices*=sz
    elif t==mujoco.mjtGeom.mjGEOM_BOX: m=trimesh.creation.box(extents=2*sz); m=m.subdivide().subdivide().subdivide()
    else: print('type',t); continue
    R=d.geom_xmat[g].reshape(3,3); v=m.vertices@R.T+d.geom_xpos[g]
    meshes.append((g,v,m.faces))
np.savez('robot_mesh.npz',TW=TW,TG=H.TG,n=len(meshes))
import pickle; pickle.dump(meshes,open('robot_mesh.pkl','wb'))
allv=np.vstack([v for _,v,_ in meshes]); print(len(meshes),allv.min(0).round(2),allv.max(0).round(2), TW.min(0).round(2),TW.max(0).round(2))
print([H.M.geom(g).name for g in H.BODY_GEOMS])
