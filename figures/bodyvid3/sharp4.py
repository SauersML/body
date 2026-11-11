import os
# Density-fair sharpness: body distance of the model's 5 nearest spots / body distance of the 5 nearest spots on the body (1 = perfect).
import numpy as np,json
from scipy.spatial.distance import pdist,squareform
d=np.load('../src/nn2.npz'); P=d['P'].copy(); P[:,0]=abs(P[:,0]); DF=squareform(pdist(P)); n=len(P); tw=d['tw']
D=DF.copy(); np.fill_diagonal(D,np.inf); ok=tw>=0; D[np.where(ok)[0],tw[ok]]=np.inf   # same exclusions as d_32B (self, mirror twin)
best=np.sort(D,1)[:,:5].mean(1); ratio=d['d_32B']/best
S=json.load(open('web/spots.json')); d32=np.array(S['d32'])
import importlib.util as iu; sp=iu.spec_from_file_location('s1',os.path.expandvars('$BODY_REPO/hires/v1/stim.py')); s1=iu.module_from_spec(sp); sp.loader.exec_module(s1)
ph=list(d['ph']); inn=np.array([ph.index(l['phrase']) for l in s1.LOCS]); assert np.allclose(d['d_32B'][inn],d32)
rk=np.argsort(np.argsort(D,1),1)+1; mr=np.array([rk[i,d['nn_32B'][i]].mean() for i in range(n)])
S['ratio']=ratio[inn].tolist(); cat=np.array(S['cat'])
out=[]
for nm,c in [('head','head'),('hands','hand'),('feet','foot'),('arms','arm'),('torso','torso'),('legs','leg')]:
    m=cat==c; r=ratio[inn][m]
    print(nm,m.sum(),'raw %.1f cm'%np.median(d32[m]),'best %.1f cm'%np.median(best[inn][m]),'ratio %.2f (IQR %.2f-%.2f)'%(np.median(r),*np.percentile(r,[25,75])),'median rank of model nn %.1f (ideal 3)'%np.median(mr[inn][m]))
    out.append([nm,float(np.median(r))])
print('all',np.median(ratio)); S['REG4']=out; json.dump(S,open('web/spots.json','w'))
