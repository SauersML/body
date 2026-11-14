import sys,os; sys.path.insert(0,'..'); import numpy as np, episodes as P, episodes2 as Q, hum as H
rng=np.random.default_rng(9); es=[Q.run_expert('poke_reach',*Q.make_scene('poke_reach',rng)) for i in range(120)]
TR=np.load('../taxel_rest.npy'); gap=[100*np.linalg.norm(TR[e['info']['taxel']]-TR[e['info']['aim']]) for e in es]; sp=[e['info']['spot_cm'] for e in es]
print('succ',np.mean([e['info']['success'] for e in es]),'aim-poke gap med',np.median(gap),'spot med',np.median(sp),'spot<5',np.mean(np.array(sp)<5),'<10',np.mean(np.array(sp)<10))
