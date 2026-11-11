import json, numpy as np, sys
sys.path.insert(0,'.'); import stimuli as S
OUT='out32'; rows=json.load(open(f'{OUT}/rows.json')); LAYERS=json.load(open(f'{OUT}/layers.json')); H=np.load(f'{OUT}/H.npy',mmap_mode='r'); li=LAYERS.index(52)
keys=[l['key'] for l in S.LOCS]
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
def X(chat): ix=[k for k,r in enumerate(rows) if r['chat']==chat]; return np.asarray(H[ix,li],np.float32), [rows[k]['loc'] for k in ix]
Xh,lh=X('human'); HM=np.stack([Xh[[i for i,l in enumerate(lh) if l==k]].mean(0) for k in keys]); HM-=HM.mean(0)
HMn=HM/np.linalg.norm(HM,axis=1,keepdims=True)
rng=np.random.default_rng(0); res={}
for chat in ['robot','glorp','octopus']:
    Xr,lr=X(chat); parts=list(dict.fromkeys(lr)); lr=np.array(lr)
    C=np.stack([Xr[lr==p].mean(0) for p in parts]); C-=C.mean(0); Cn=C/np.linalg.norm(C,axis=1,keepdims=True)
    real=(Cn@HMn.T).max(1)
    nul=[]
    for _ in range(500):
        perm=rng.permutation(len(lr)); lp=lr[perm]
        C=np.stack([Xr[lp==p].mean(0) for p in parts]); C-=C.mean(0); Cn=C/np.linalg.norm(C,axis=1,keepdims=True); nul+=list((Cn@HMn.T).max(1))
    nul=np.array(nul); res[chat]=dict(real=dict(zip(parts,real.round(3).tolist())),null_pct=np.percentile(nul,[50,95,99]).round(3).tolist())
    print(chat,'best-match cos per part',dict(zip(parts,real.round(2))),'| shuffled-parts best match: median/95/99',np.percentile(nul,[50,95,99]).round(2))
json.dump(res,open(f'{OUT}/null52.json','w'))
