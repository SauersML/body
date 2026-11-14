import os,sys,pickle,numpy as np
os.environ.setdefault("MUJOCO_GL","cgl")
sys.path.insert(0,'.')
import mujoco, hum as H
from PIL import Image
d0=pickle.load(open('../x8abs/steer_confirm_Qwen3-8B_named.pkl','rb'));r=d0['runs']
m=H.M; d=mujoco.MjData(m)
tgt=sys.argv[1]; i=int(sys.argv[2]); az=float(sys.argv[3]); W=int(sys.argv[4]); el=float(sys.argv[5]) if len(sys.argv)>5 else -12
p=H.PARTS.index(tgt); MR=0.06
m.vis.global_.offwidth=2400; m.vis.global_.offheight=2400
R=mujoco.Renderer(m,W,W); S=mujoco.Renderer(m,W,W); S.enable_segmentation_rendering()
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:]=[0.0,0,0.86]; cam.distance=2.3; cam.elevation=el; cam.azimuth=az
R.update_scene(d,cam); R.render(); S.update_scene(d,cam); S.render()
for cond in ['none',tgt]:
    e=r[cond][i]; d.qpos[:]=e['qpos_final']; mujoco.mj_forward(m,d)
    P,N=H.taxel_world(d); c=P[H.TPART==p].mean(0)
    d.mocap_pos[:]=[0,0,-50]; mujoco.mj_forward(m,d)
    imgs=[]
    for X,mk in ((R,1),(R,0),(S,0)):
        X.update_scene(d,cam); sc=X.scene; sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        if mk:
            gg=sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([MR,0,0]),c,np.eye(3).flatten(),np.array([0.82,0.26,0.18,0.55],np.float32)); sc.ngeom+=1
        imgs.append(X.render().copy())
    rgb,rgb0,seg=imgs; bg=(seg[...,0]<=0)&(np.abs(rgb.astype(int)-rgb0).sum(-1)<30)
    rgb[bg]=255
    tips=[d.site(f"tip_{h}").xpos for h in ("left","right")]
    dist=min(np.linalg.norm(P[H.TPART==p]-t,axis=1).min() for t in tips)
    hit=any(p in x[0] for x in e['tc'][-6:])
    Image.fromarray(rgb).save(f'$BODY_WORK/fig/b3h_{cond.replace(" ","_")}_{i}_{int(az)}.png')
    print(cond,i,az,round(dist*100,1),hit)
