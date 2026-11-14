import sys, numpy as np, time
sys.path.insert(0, "$BODY_WORK/body3d")
import episodes as P, hum as H
rng=np.random.default_rng(55); out=[]; t=time.time()
while len(out)<400:
    e=P.make_episode(rng,"poke")
    if e["info"]["success"]: out.append((e["info"]["taxel"], e["info"]["part"]))
np.save('pokes400.npy', np.array(out)); print(time.time()-t, np.bincount(np.array(out)[:,1], minlength=17))
