import json, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
O='out8b'; P=json.load(open(f'{O}/probes.json')); J=2
f=lambda k: np.array([r[k] for r in P['wing']]); side,pos,reg,surf=f('side'),f('pos'),f('reg'),f('surf')
def tr(a,b,y):
    A=np.load(f'{O}/H_{a}.npy',mmap_mode='r')[:,J].astype(np.float32); B=np.load(f'{O}/H_{b}.npy',mmap_mode='r')[:,J].astype(np.float32)
    A-=A.mean(0); B-=B.mean(0); p=PCA(128,random_state=0).fit(A); Ar,Br=p.transform(A),p.transform(B); sd=Ar.std(0); 
    return Ridge(alpha=100).fit(Ar/sd,y).predict(Br/sd)
out={}
for nm,(a,b) in {'W':('W_wing','W_desc'),'N':('N_wing','N_desc'),'G':('G_glorp','G_desc')}.items():
    pr=tr(a,b,reg/2.); pp=tr(a,b,pos/4.)
    grid=lambda x,u: [[[float(x[(side==s)&(pos==p)&(reg==r)&(surf==u)].mean()) for r in range(3)] for p in range(5)] for s in range(2)]
    out[nm]=dict(reg_up=grid(pr,0),reg_all=[[[float(pr[(side==s)&(pos==p)&(reg==r)].mean()) for r in range(3)] for p in range(5)] for s in range(2)],
                 pos_all=[[[float(pp[(side==s)&(pos==p)&(reg==r)].mean()) for r in range(3)] for p in range(5)] for s in range(2)],
                 reg_acc=float((np.abs(pr[:,None]-np.array([0,.5,1])[None]).argmin(1)==reg).mean()))
    print(nm, out[nm]['reg_acc'], np.round(np.array(out[nm]['reg_all']).mean((0,1)),2), np.round(np.array(out[nm]['pos_all']).mean((0,2)),2))
json.dump(out,open(f'{O}/figdata.json','w'))
