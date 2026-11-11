import os
import numpy as np, json
B2=os.path.expandvars('$BODY_REPO/figures/bodyvid2/')
d=np.load(B2+'data.npz',allow_pickle=True); XM=d['XM']; SIDE=d['side']; cat=list(d['cat'])
V=np.fromfile('web/pos.bin',np.float32).reshape(-1,3).astype(float)
order={'head':0,'arm':1,'hand':2,'torso':1.5,'leg':2.5,'foot':3}
lag=np.array([order[c] for c in cat])/3*1.0+(XM[:,1].max()-XM[:,1])/177*0.3
def sidew(p): return np.where(SIDE[None]==0,1.0,1/(1+np.exp(-SIDE[None]*p[:,0:1]/2.5)))+1e-12
out=np.zeros(len(V))
for i in range(0,len(V),20000):
    p=V[i:i+20000]; w=np.exp(-((p[:,None,:]-XM[None])**2).sum(-1)/(2*5.0**2))*sidew(p)+1e-300
    out[i:i+20000]=(w@lag)/w.sum(1)
out.astype(np.float32).tofile('web/lagv.bin')
S=json.load(open('web/spots.json')); S['lag']=lag.tolist(); S['Yu']=d['Yu'].tolist(); json.dump(S,open('web/spots.json','w'))
print(lag.min(),lag.max(),out.min(),out.max(),'Yu',d['Yu'].min(0).round(),d['Yu'].max(0).round())
