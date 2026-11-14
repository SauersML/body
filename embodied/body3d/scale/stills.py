import sys,os; sys.path.insert(0,'..'); os.environ.update(X2_CANON='2', X2_GSMOOTH='1', X2_GCHI='0.07', X2_GCLO='0.04')
import numpy as np, mujoco, env as E, hum as H, episodes2 as Q, rb3d as R
from PIL import Image
def go(kind, seed, name, t_snap=None, **kw):
    sim,info,pol=Q.make_scene(kind,np.random.default_rng(seed),True); snaps={}
    for t in range(20):
        sim.step(sim.grav_ctrl(pol.act(sim,t)),t)
        snaps[t]=(sim.d.qpos.copy(), sim.d.mocap_pos.copy(), sim.d.mocap_quat.copy(), sim.last_skin.copy())
    out={}
    for t in (t_snap or [19]):
        q,mp,mq,sk=snaps[t]
        def prep(d): d.mocap_pos[:]=mp; d.mocap_quat[:]=mq
        cols=[(1,0.15,0.1,1) if s>0.05 else None for s in sk]
        img=R.render(q, W=1000, Hh=1200, prep=prep, taxels=(cols,0.012), **kw); Image.fromarray(img).save(f'fig/{name}_t{t}.png'); out[t]=img
    print(name, info.get('part'), H.PARTS[info['part']] if 'part' in info else '')
go('reach',[5151,1,3],'reach', az=140, el=-5, dist=2.2)
go('poke_reach',[5151,2,4],'poke', t_snap=[3,19], az=140, el=-5, dist=2.2)
go('grasp',[5151,3,1],'grasp', az=150, el=-5, dist=2.2)
