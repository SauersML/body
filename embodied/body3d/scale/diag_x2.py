import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, collections, time, env as E, hum as H, episodes as P, expert2 as X
n=int(sys.argv[1]); rng=np.random.default_rng(1); by=collections.defaultdict(list); t00=time.time(); within=[]
for i in range(n):
    sim=E.Sim("mount",rng); ti=int(rng.integers(H.NT)); part=int(H.TPART[ti])
    h,tt,err=X.choose(sim,part); pol=X.Reach(h,tt,part,t_go=0); tc=[]
    for t in range(P.T):
        c=sim.grav_ctrl(pol.act(sim,t)); sim.step(c,t); tc.append(E.touch_info(sim))
    ok=any(part in x[0] for x in tc[-6:]); by[H.PARTS[part]].append(ok)
    Pw,_=H.taxel_world(sim.d); within.append(np.linalg.norm(Pw[tt]-sim.d.site(f"tip_{h}").xpos)<0.05)
tot=[x for v in by.values() for x in v]; print('x2 reach',np.mean(tot),'within5cm',np.mean(within),'s/ep',(time.time()-t00)/n)
for k,v in sorted(by.items(),key=lambda kv:np.mean(kv[1])): print(f'  {k:16s} {np.mean(v):.2f} {len(v)}')
