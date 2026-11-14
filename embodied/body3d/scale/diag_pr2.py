import sys,os; sys.path.insert(0,'..'); import numpy as np, collections, episodes2 as Q, hum as H
kind=sys.argv[1]; rng=np.random.default_rng(5); by=collections.defaultdict(list)
for i in range(int(sys.argv[2])):
    e=Q.run_expert(kind,*Q.make_scene(kind,rng,True)); by[H.PARTS[e['info']['part']]].append(e['info']['success'])
print(kind, np.mean([x for v in by.values() for x in v]), sorted([(round(np.mean(v),2),k,len(v)) for k,v in by.items()])[:7])
