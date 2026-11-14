import sys,os; sys.path.insert(0,'..'); os.environ.update(X2_CANON='2', X2_GSMOOTH='1', X2_GCHI='0.07', X2_GCLO='0.04')
import numpy as np, episodes as P, episodes2 as Q, env as E
Q.install(); rng=np.random.default_rng(0); D=[]
for i in range(600):
    e=P.make_episode(rng); D.append(e['A']-e['S'][:20,:E.NJ])
D=np.concatenate(D); np.savez('dnorm.npz',mu=D.mean(0).astype(np.float32),sd=np.maximum(D.std(0),1e-3).astype(np.float32)); print(D.std(0).round(3)[:12], D.std(0).min())
