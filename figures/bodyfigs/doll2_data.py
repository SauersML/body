import os
import json,numpy as np
H=np.load(os.path.expandvars('$BODY_REPO/map/somato/out2_8b/H_rs.npy'),mmap_mode='r')
R=json.load(open(os.path.expandvars('$BODY_REPO/map/somato/out2_8b/rows.json')))
J=json.load(open(os.path.expandvars('$BODY_REPO/figures/out/_selfother_L27.json')))
ph=[eval(s)['phrase'] for s in J['locs']]
L=27
idx={c:np.zeros((54,8),int) for c in('self','other','object')}
for k,r in enumerate(R):
    if r['cond'] in idx: idx[r['cond']][r['li'],r['tpl']]=k
X=np.asarray(H[:,L]).astype(np.float32)
def A(c): a=X[idx[c]]; return a-a.mean(0,keepdims=True)   # remove each wording's average over the 54 places
S=A('self'); C=S[:,[0,2,4,6]].mean(1)                       # fingerprints
acc={}
for c in idx:
    V=A(c)[:,[1,3,5,7]]; p=((V[:,:,None]-C[None,None])**2).sum(-1).argmin(-1)
    acc[c]=float((p==np.arange(54)[:,None]).mean())
print(acc)
between=C.var(0); within=(S[:,[0,2,4,6]]-C[:,None]).var((0,1))
D=np.argsort(-between/(within+1e-6))[:40]
sd=C[:,D].std(0)
z=lambda v: (v[D]/sd).round(2).tolist()
SHOW=[0,1,8,17,20,44,51]
dist={c:((A(c)[0,1][None]-C)**2).sum(-1) for c in idx}
out=dict(acc=acc,phrases=ph,show=SHOW,
  feather=z(S[0,0]), wordings=[z(S[0,t]) for t in (0,2,4,6)],
  fp={ph[i]:z(C[i]) for i in range(54)},
  test={c:z(A(c)[0,1]) for c in idx},
  dist={c:{ph[i]:float(dist[c][i]) for i in range(54)} for c in idx},
  guess={c:ph[int(dist[c].argmin())] for c in idx},
  texts={c:R[idx[c][0,1]]['text'] for c in idx}, feather_text=R[idx['self'][0,0]]['text'],
  layers=J['layers'])
json.dump(out,open(os.path.expandvars('$BODY_REPO/figures/bodyfigs/doll2_data.json'),'w'))
print(out['guess'],out['texts'])
for c in idx: print(c,sorted(out['dist'][c].items(),key=lambda x:x[1])[:4])
