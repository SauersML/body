import sys,os; sys.path.insert(0,'..'); os.environ.setdefault("MUJOCO_GL","cgl")
import numpy as np, pickle, episodes2 as Q, hum as H
V={}
for kind in Q.MOTOR:
    tr=[];inf=[];ok=[]
    for i in range(3):
        sim,info,pol=Q.make_scene(kind,np.random.default_rng([5151,{'reach':1,'poke_reach':2,'grasp':3}[kind],i]),diverse=True); sim.rec=[]
        e=Q.run_expert(kind,sim,info,pol); tr.append(sim.rec); inf.append(e['info']); ok.append(e['info']['success'])
    V[kind]=dict(traj=tr,info=inf,ok=ok)
pickle.dump(V,open('expert_vids.pkl','wb'))
