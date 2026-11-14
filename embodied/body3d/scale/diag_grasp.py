import sys,os; sys.path.insert(0,'..'); import numpy as np, episodes as P, episodes2 as Q, env as E, hum as H
rng=np.random.default_rng(2); ok=[]
for i in range(int(sys.argv[1])):
    e=P.ep_grasp(rng); ok.append(e['info']['success'])
print('old grasp Q0',np.mean(ok))
