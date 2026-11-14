import sys,os,numpy as np,json
from scipy.stats import spearmanr
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D+'/hpriv'); from bodies import xyz
Z=np.load(f'{D}/hpriv/out/q32_states.npz',allow_pickle=True); meta=Z['meta']
bo=np.array([m[0] for m in meta]); pi=np.array([int(m[1]) for m in meta]); tj=np.array([int(m[2]) for m in meta])
iu=np.triu_indices(16,1); X=Z['L31'].astype(np.float32)
G=lambda b:(lambda c:np.linalg.norm(c[:,None]-c[None],axis=-1)[iu])(xyz(b)/np.ptp(xyz(b),0).max())
C={}
for b in ['human','velk','dog','robot','bird']:
    m=bo==b; c=np.zeros((16,8,X.shape[1]),np.float32); c[pi[m],tj[m]]=X[m]; C[b]=c
def rsa(b,T):
    P=C[b][:,T].mean(1); P=P-P.mean(0); P/=np.linalg.norm(P,axis=1,keepdims=True); return spearmanr(1-(P@P.T)[iu],G(b))[0]
rng=np.random.default_rng(1); out={}
for o in ['velk','dog','robot','bird']:
    d=[]; 
    for _ in range(2000):
        T=rng.choice(8,8); d.append(rsa('human',T)-rsa(o,T))
    d=np.array(d); out[o]=[float(np.mean(d<=0)), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))]
print(out); json.dump(out,open(f'{D}/pval_q32_L31.json','w'))
