import os
import sys,os,numpy as np,json
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
D=os.path.expandvars('$BODY_REPO/mods/base')
R=MS.probes(); KEEP=[22,24,25,26,27,28,30]; LAY=int(os.environ.get('LAY',27)); li=KEEP.index(LAY)
spots=MS.SPOTS; reg=np.array([S.LOCS[s]['region'] for s in spots]); ph=[S.LOCS[s]['phrase'] for s in spots]
def load(n,center=True):
    H=np.load(f"{D}/H/{n.replace(':','__')}.npy")[li].astype(np.float32)
    X=np.zeros((len(spots),H.shape[1])); Nw=np.zeros((len(MS.NEW),H.shape[1]))
    for i,r in enumerate(R):
        if r['kind']=='loc': X[spots.index(r['li'])]+=H[i]/2
        else: Nw[r['li']]+=H[i]/2
    if center: mu=X.mean(0); X=X-mu; Nw=Nw-mu
    return X,Nw
N,NwN=load('neutral'); 
# unit: median distance between a spot and its nearest other spot in neutral? use mean distance between spot states
from scipy.spatial.distance import cdist
Dn=cdist(N,N); unit=np.median(Dn[np.triu_indices(len(N),1)])
print('unit (median pairwise neutral)',unit)
GN=["head","neck","torso","armL","armR","handL","handR","legL","legR","footL","footR"]
def summ(c,ref='neutral'):
    X,_=load(c); Xr,_=load(ref); d=np.linalg.norm(X-Xr,axis=1)/unit
    return d
res={}
for m in MS.MODS:
    regs=MS.MODS[m][4]
    dt=summ(f'told_N:{m}'); ds=summ(f'sham:{m}'); da=summ(f'told_A:{m}')
    mk=np.isin(reg,regs) if regs else np.ones(len(reg),bool)
    res[m]=(dt[mk].mean(),ds[mk].mean(),da[mk].mean(),dt[~mk].mean() if (~mk).any() else 0, ds[~mk].mean() if (~mk).any() else 0)
    print(f"{m:15s} target told {dt[mk].mean():.3f} act {da[mk].mean():.3f} sham {ds[mk].mean():.3f} | rest told {res[m][3]:.3f} sham {res[m][4]:.3f} | max-spot told {dt.max():.3f} at {ph[int(dt.argmax())]}")
print('base vs neutral',summ('base').mean())
