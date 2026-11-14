import sys,os,json,numpy as np
from scipy.stats import spearmanr
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D+'/hpriv'); from bodies import xyz
iu=np.triu_indices(16,1); rng=np.random.default_rng(2); out={}
for t,L in [('q32',31),('q8',19)]:
    Z=np.load(f'{D}/hpriv/out/{t}_states.npz',allow_pickle=True); meta=Z['meta']
    bo=np.array([m[0] for m in meta]); pi=np.array([int(m[1]) for m in meta]); X=Z[f'L{L}'].astype(np.float32); out[t]={}
    for b in ['human','dog','bird','octopus','snake','tree','robot','velk']:
        m=bo==b; P=np.stack([X[m][pi[m]==i].mean(0) for i in range(16)]); P-=P.mean(0); P/=np.linalg.norm(P,axis=1,keepdims=True)
        M=1-P@P.T; c=xyz(b)/np.ptp(xyz(b),0).max(); G=np.linalg.norm(c[:,None]-c[None],axis=-1)
        nl=[]
        for _ in range(2000):
            p=rng.permutation(16); nl.append(spearmanr(M[iu],G[np.ix_(p,p)][iu])[0])
        out[t][b]=[float(np.median(nl)),float(np.percentile(nl,95)),float(spearmanr(M[iu],G[iu])[0])]
    print(t,{b:[round(x,2) for x in v] for b,v in out[t].items()})
json.dump(out,open(f'{D}/null_rsa.json','w'))
