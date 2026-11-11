import os
import json,numpy as np
B=os.path.expandvars('$BODY_REPO/map/somato/out2_8b/')
H=np.load(B+'H_rs.npy',mmap_mode='r'); R=json.load(open(B+'rows.json'))
J=json.load(open(os.path.expandvars('$BODY_REPO/figures/out/_selfother_L27.json')))
ph=[eval(s)['phrase'] for s in J['locs']]
idx={c:np.zeros((54,8),int) for c in('self','other','object')}
for k,r in enumerate(R):
    if r['cond'] in idx: idx[r['cond']][r['li'],r['tpl']]=k
X=np.asarray(H[:,27]).astype(np.float32)
A={c:X[idx[c]]-X[idx[c]].mean(0,keepdims=True) for c in idx}
C=A['self'][:,[0,2,4,6]].mean(1); Cn=C/np.linalg.norm(C,axis=-1,keepdims=True)
out={'acc':{},'acc_euc':{},'norm':{},'texts':{}}
for c in idx:
    V=A[c][:,[1,3,5,7]]; Vn=V/np.linalg.norm(V,axis=-1,keepdims=True)
    out['acc'][c]=float(((Vn@Cn.T).argmax(-1)==np.arange(54)[:,None]).mean())
    out['acc_euc'][c]=float((((V[:,:,None]-C[None,None])**2).sum(-1).argmin(-1)==np.arange(54)[:,None]).mean())
    out['norm'][c]=float(np.linalg.norm(A[c],axis=-1).mean())
    out['texts'][c]=[R[idx[c][0,t]]['text'] for t in range(8)]
n={c:np.linalg.norm(A[c],axis=-1).mean(1) for c in idx}
out['doll_shorter_places']=int((n['object']<n['self']).sum()); out['woman_shorter_places']=int((n['other']<n['self']).sum())
# angles for pictogram: cos of thumb test (self, tpl1) to a few fingerprints
v=A['self'][0,1]; v/=np.linalg.norm(v)
out['cos_example']={ph[i]:float(v@Cn[i]) for i in range(54)}
out['phrases']=ph
json.dump(out,open('data.json','w'),indent=0)
print({k:out[k] for k in ('acc','acc_euc','norm','doll_shorter_places','woman_shorter_places')})
print(sorted(out['cos_example'].items(),key=lambda x:-x[1])[:6])
for c in idx: print(out['texts'][c][:2])
