import sys,os; sys.path.insert(0,'..'); import numpy as np, env as E, hum as H, episodes2 as Q
rng=np.random.default_rng(0); B=[];A=[];D=[]
for i in range(40):
    sim,info,pol=Q.make_scene('reach',rng,True)
    for t in range(20):
        q=pol.act(sim,t); b=sim.d.qfrc_bias[H.JV]/E.KPV; c=q+b; B.append(b); A.append(c); D.append(q-sim.d.qpos[H.JQ]); sim.step(c,t)
B,A,D=map(np.array,(B,A,D)); js=E.ARM['left']+E.ARM['right']
print('arm joints: std action',A[:,js].std(0).mean().round(3),'std bias',B[:,js].std(0).mean().round(3),'mean|bias|',np.abs(B[:,js]).mean().round(3),'std delta',D[:,js].std(0).mean().round(3))
