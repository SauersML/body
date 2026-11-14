"""Per-spot classifier output for the wing-part transfer (same as ana.py transfer(), layer index 2):
RidgeClassifier trained on 'wing' sentences, applied to the no-'wing' sentences; for each (side, pos, reg) cell,
the fraction of sentences put in each class. Writes out8b/clsdata.json."""
import json, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
O='out8b'; J=2; P=json.load(open(f'{O}/probes.json'))
f=lambda k: np.array([r[k] for r in P['wing']]); side,pos,reg=f('side'),f('pos'),f('reg')
out={}
for nm,(a,b) in {'W':('W_wing','W_desc'),'N':('N_wing','N_desc')}.items():
    A=np.load(f'{O}/H_{a}.npy',mmap_mode='r')[:,J].astype(np.float32); B=np.load(f'{O}/H_{b}.npy',mmap_mode='r')[:,J].astype(np.float32)
    A=A-A.mean(0); B=B-B.mean(0); p=PCA(128,random_state=0).fit(A); Ar,Br=p.transform(A),p.transform(B); sd=Ar.std(0)+1e-6; Ar/=sd; Br/=sd
    pr=RidgeClassifier(alpha=100).fit(Ar,reg).predict(Br)
    frac=[[[[float((pr[(side==s)&(pos==q)&(reg==r)]==c).mean()) for c in range(3)] for r in range(3)] for q in range(5)] for s in range(2)]
    out[nm]=dict(acc=float((pr==reg).mean()),frac=frac,n=int(len(pr)))
    print(nm,out[nm]['acc'],len(pr)); print(np.round(np.array(frac).mean((0,1)),2))
json.dump(out,open(f'{O}/clsdata.json','w'))
