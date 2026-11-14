import sys; sys.path.insert(0,'..'); import numpy as np, collections, time
import episodes as P, hum as H
rng=np.random.default_rng(1); n=int(sys.argv[1]); kind=sys.argv[2]
by=collections.defaultdict(list); t=time.time()
for i in range(n):
    e=P.make_episode(rng,kind); by[H.PARTS[e['info']['part']]].append(e['info']['success'])
tot=[x for v in by.values() for x in v]; print(kind,'success',np.mean(tot),'n',len(tot),'s/ep',(time.time()-t)/n)
for k,v in sorted(by.items(),key=lambda kv:np.mean(kv[1])): print(f'  {k:16s} {np.mean(v):.2f} {len(v)}')
