import os
import numpy as np, sys, json, os
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
HERE=os.path.dirname(os.path.abspath(__file__))
VM=np.load(f'{HERE}/vmap.npz',allow_pickle=True); P=VM['P']; G=VM['G']
spots=MS.SPOTS; SX=np.array([S.LOCS[s]['xyz'] for s in spots]); SR=np.array([S.LOCS[s]['region'] for s in spots])
OK={'armL':['armL','handL'],'armR':['armR','handR'],'handL':['handL'],'handR':['handR'],'legL':['legL','footL'],'legR':['legR','footR'],
    'footL':['footL','legL'],'footR':['footR','legR'],'torso':['torso','neck','armL','armR'],'head':['head','neck']}
SIG={'handL':1.6,'handR':1.6}
def paint(val):
    out=np.zeros(len(P))
    for g,regs in OK.items():
        vm=G==g; sm=np.isin(SR,regs); sig=SIG.get(g,4.0)
        d2=((P[vm][:,None,:]-SX[sm][None])**2).sum(-1); w=np.exp(-d2/(2*sig**2))+1e-12
        # also keep nearest-dominant when far
        out[vm]=(w*val[sm]).sum(1)/w.sum(1)
    return out
import scipy.sparse as sp
_F=np.load(os.path.expandvars('$BODY_REPO/mods/phantom_v3/mesh/mesh.npz'))['f']
_n=len(P); _r=np.concatenate([_F[:,0],_F[:,1],_F[:,2],_F[:,1],_F[:,2],_F[:,0]]); _c=np.concatenate([_F[:,1],_F[:,2],_F[:,0],_F[:,0],_F[:,1],_F[:,2]])
_A=sp.csr_matrix((np.ones(len(_r)),(_r,_c)),shape=(_n,_n)); _A.data[:]=1; _deg=np.asarray(_A.sum(1)).ravel(); _Wm=sp.diags(1/_deg)@_A
def smooth(v,it=40):
    for _ in range(it): v=0.5*v+0.5*(_Wm@v)
    return v
