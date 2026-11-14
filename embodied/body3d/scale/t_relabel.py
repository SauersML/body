import sys,os; sys.path.insert(0,'..'); import numpy as np, env as E, hum as H, episodes2 as Q
for kind in Q.MOTOR:
    a=[];b=[];na=[]
    for i in range(40):
        a.append(Q.run_expert(kind,*Q.make_scene(kind,np.random.default_rng([1,i])))['info']['success'])
        sim,info,pol=Q.make_scene(kind,np.random.default_rng([1,i])); tc=[]
        for t in range(20):
            bias=sim.d.qfrc_bias[H.JV]/E.KPV; c=sim.grav_ctrl(pol.act(sim,t)); c=np.clip(c,H.JR[:,0]-.3,H.JR[:,1]+.3); pol.qprev=c-bias; sim.step(c,t); tc.append(E.touch_info(sim))
        b.append(Q.success(kind,info,sim,tc))
        # noisy executed actions, expert relabel: does following labels with noise still mostly work?
        sim,info,pol=Q.make_scene(kind,np.random.default_rng([1,i])); tc=[]; rng=np.random.default_rng(i)
        for t in range(20):
            bias=sim.d.qfrc_bias[H.JV]/E.KPV; c=sim.grav_ctrl(pol.act(sim,t))+rng.normal(0,.08,41); pol.qprev=c-bias; sim.step(c,t); tc.append(E.touch_info(sim))
        na.append(Q.success(kind,info,sim,tc))
    print(kind,'expert',np.mean(a),'relabel-loop',np.mean(b),'noisy-loop',np.mean(na))
