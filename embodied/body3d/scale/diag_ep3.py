import sys,os; sys.path.insert(0,'..'); import numpy as np, time, episodes as P, episodes2 as Q
Q.install(); rng=np.random.default_rng(9); t0=time.time()
for k in Q.MOTOR:
    es=[P.make_episode(rng,k) for i in range(80)]; print(k,'succ',np.mean([e['info']['success'] for e in es]),'spot<5',np.mean([e['info']['spot_cm']<5 for e in es]), es[0]['S'].shape, es[0]['A'].shape, P.questions(es[0],True)[:1])
e=P.make_episode(rng); print('mix ok',e['kind'], round(time.time()-t0,1))
# old held-out with new expert
for k in Q.MOTOR:
    ok=[Q.run_expert(k,*Q.scene_from_old(k,Q.OLD[k](np.random.default_rng([4242,i]))))['info']['success'] for i in range(60)]; print('old scenes new expert',k,np.mean(ok))
