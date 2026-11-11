import json, numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
O='out8b'; P=json.load(open(f'{O}/probes.json')); TR=json.load(open(f'{O}/traces.json'))
t=np.array([r['t'] for r in P['wing']])
for J,Lr in [(1,18),(2,24),(3,27),(4,32)]:
  for cn in 'WN':
    for src in ('wing','desc'):
      H=np.load(f'{O}/H_{cn}_{src}.npy',mmap_mode='r')[:,J].astype(np.float32); mu=H.mean(0)
      pca=PCA(128,random_state=0).fit(H-mu); X=pca.transform(H-mu); sd=X.std(0); m=Ridge(alpha=100).fit(X/sd,t)
      T=np.load(f'{O}/T_{cn}.npy')[:,J].astype(np.float32)   # (96,6,d)
      Tc=T-T.reshape(-1,T.shape[-1]).mean(0)
      pr=m.predict(pca.transform(Tc.reshape(-1,T.shape[-1]))/sd).reshape(96,6)
      for nm in ('wing','desc'):
        o=[i for i,x in enumerate(TR) if x['nm']==nm and x['dir']==0]; n=[i for i,x in enumerate(TR) if x['nm']==nm and x['dir']==1]
        print(Lr,cn,'probe from',src,'| trace',nm,'out',pr[o].mean(0).round(2),'in',pr[n].mean(0).round(2))
