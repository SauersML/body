import sys,os; sys.path.insert(0,'..'); import numpy as np, env as E, hum as H, episodes2 as Q
kind=sys.argv[1]; sig=float(sys.argv[2]); n=int(sys.argv[3]); ok=[]; ok0=[]
for i in range(n):
    for s,store in ((0,ok0),(sig,ok)):
        sim,info,pol=Q.make_scene(kind,np.random.default_rng([3,i]),True); tc=[]; rng=np.random.default_rng(i); z=np.zeros(41)
        for t in range(20):
            bias=sim.d.qfrc_bias[H.JV]/E.KPV; z=0.7*z+0.714*rng.normal(0,1,41); c=sim.grav_ctrl(pol.act(sim,t))+s*z; pol.qprev=c-bias; sim.step(c,t); tc.append(E.touch_info(sim))
        store.append(Q.success(kind,info,sim,tc))
print(kind,os.environ.get('TAG',''),'clean',np.mean(ok0),'noisy',np.mean(ok))
