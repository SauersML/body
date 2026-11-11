import os
import sys, numpy as np, collections
sys.path.insert(0,os.path.expandvars('$BODY_REPO/figures/zoom3'))
from load import load, joints
from scipy.sparse import coo_matrix, diags
OBJ=os.path.expandvars('$BODY_REPO/mesh/base.obj')
V,G=load(OBJ); J=joints(V,G)
faces=G['body']; used=np.unique(np.concatenate([np.array(f) for f in faces])); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used)); P=V[used]
tri=[]
for f in faces:
    f=[remap[i] for i in f]; tri+=[[f[0],f[1],f[2]],[f[0],f[2],f[3]]] if len(f)==4 else [f]
tri=np.array(tri)
def seg(a,b):
    a=np.asarray(a);b=np.asarray(b); ab=b-a; t=np.clip(((P-a)@ab)/(ab@ab),0,1); return np.linalg.norm(P-(a+t[:,None]*ab),axis=1)
BON=collections.defaultdict(list)
spine=[J['pelvis'],J['spine-3'],J['spine-2'],J['spine-1'],J['neck']] if 'spine-1' in J else [J['pelvis'],J['neck']]
for a,b in zip(spine[:-1],spine[1:]): BON['torso'].append((a,b))
BON['head'].append((J['head'],J['head-2']))
for s,S_ in (('l','L'),('r','R')):
    BON['torso'].append((J['neck'],J[f'{s}-clavicle'])); BON['torso'].append((J[f'{s}-clavicle'],J[f'{s}-shoulder']*[0.75,1,1]))
    BON['arm'+S_]+= [(J[f'{s}-shoulder'],J[f'{s}-elbow']),(J[f'{s}-elbow'],J[f'{s}-hand'])]
    for k in range(1,6): BON['hand'+S_].append((J[f'{s}-hand'],J[f'{s}-finger-{k}-4']))
    BON['leg'+S_]+= [(J[f'{s}-upper-leg'],J[f'{s}-knee']),(J[f'{s}-knee'],J[f'{s}-ankle'])]
    BON['foot'+S_]+= [(J[f'{s}-ankle'],J[f'{s}-foot-2'])]+[(J[f'{s}-foot-1'],J[f'{s}-toe-{k}-1']) for k in range(1,6)]
GR=list(BON)
Dm=np.stack([np.min([seg(a,b) for a,b in BON[g]],0) for g in GR],1)
lab=np.argmin(Dm,1)
# neck: head bone owns only above chin height
hy=J['neck'][1]+0.35
lab[(np.array(GR)[lab]=='head')&(P[:,1]<hy)]=GR.index('torso')
isleg=np.isin(np.array(GR)[lab],['legL','legR']); lab[isleg&(P[:,1]>J['l-upper-leg'][1]-0.25)]=GR.index('torso')
e=np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]); A=coo_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(P),)*2).tocsr(); A.data[:]=1
Lw=diags(1/np.asarray(A.sum(1)).ravel())@A
def field(vals, it=25):
    f=np.array([vals[GR[k]] for k in lab],float)
    for _ in range(it): f=0.5*f+0.5*(Lw@f)
    return f
