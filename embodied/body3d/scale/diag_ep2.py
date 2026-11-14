import sys,os; sys.path.insert(0,'..'); import numpy as np, episodes2 as Q
kind=sys.argv[2]; n=int(sys.argv[1]); div=sys.argv[3]=='1'; rng=np.random.default_rng(5)
es=[Q.run_expert(kind,*Q.make_scene(kind,rng,div)) for i in range(n)]
print(kind,'div',div,'succ',np.mean([e['info']['success'] for e in es]),'spot<5',np.mean([e['info']['spot_cm']<5 for e in es]))
