import os
import sys,os,numpy as np,json
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
from scipy.spatial.distance import cdist
D=os.path.expandvars('$BODY_REPO/mods/phantom_v2/pod')
R=json.load(open(f'{D}/probes2.json')); KEEP=[24,27,30]; LAY=int(os.environ.get('LAY',27)); li=KEEP.index(LAY)
spots=MS.SPOTS; reg=np.array([S.LOCS[s]['region'] for s in spots]); ph=np.array([S.LOCS[s]['phrase'].replace('{p} ','') for s in spots])
C=json.load(open(f'{D}/contexts2.json')); names=list(C)
nex=max(r['li'] for r in R if r['kind']=='extra')+1
def load(n):
    H=np.load(f"{D}/H2/{n.replace(':','__')}.npy")[li].astype(np.float32)
    X=np.zeros((len(spots),H.shape[1])); Nw=np.zeros((len(MS.NEW),H.shape[1])); Ex=np.zeros((nex,H.shape[1]))
    for i,r in enumerate(R):
        if r['kind']=='loc': X[spots.index(r['li'])]+=H[i]/2
        elif r['kind']=='new': Nw[r['li']]+=H[i]/2
        else: Ex[r['li']]+=H[i]/2
    mu=X.mean(0); return X-mu,Nw-mu,Ex-mu
L={n:load(n) for n in names}
grp=lambda c:[n for n in names if n.startswith(c+':')]
NEU=np.mean([L[n][0] for n in grp('neutral')],0)
unit=np.median(cdist(NEU,NEU)[np.triu_indices(len(NEU),1)])
def disp(c,ref=NEU): return np.array([np.linalg.norm(L[n][0]-ref,axis=1)/unit for n in grp(c)])  # chats x spots
# noise: leave-one-out neutral
nn=grp('neutral'); noise=np.array([np.linalg.norm(L[n][0]-np.mean([L[m][0] for m in nn if m!=n],0),axis=1)/unit for n in nn])
hr=reg=='handR'; fl=reg=='footL'
out={}
for c in ['gone','ctrl','footgone','footctrl']:
    d=disp(c); out[c]=d
    print(f"{c:9s} handR {d[:,hr].mean():.2f} (per chat {np.round(d[:,hr].mean(1),2)}) footL {d[:,fl].mean():.2f} rest {d[:,~(hr|fl)].mean():.2f}")
print('noise',noise[:,hr].mean(),noise[:,fl].mean(),noise.mean())
d=out['gone'].mean(0)-out['ctrl'].mean(0)
o=np.argsort(-d); print([ (round(d[k],2),ph[k]) for k in o[:12]])
print('region means gone-ctrl', {g: round(float(d[reg==g].mean()),2) for g in sorted(set(reg))})
dw=out['gone'].mean(0); dc=out['ctrl'].mean(0)
for k in np.where(hr)[0]: print(f"{dw[k]:.2f} {dc[k]:.2f}  {ph[k]}")
np.savez('v2_disp_L%d.npz'%LAY,gone=out['gone'],ctrl=out['ctrl'],footgone=out['footgone'],footctrl=out['footctrl'],noise=noise,unit=unit)
