import sys,os; sys.path.insert(0,'..'); import numpy as np, episodes2 as Q
for div in (0,1):
    rng=np.random.default_rng(2); es=[Q.run_expert('grasp',*Q.make_scene('grasp',rng,bool(div))) for i in range(int(sys.argv[1]))]; print('grasp div',div,np.mean([e['info']['success'] for e in es]))
