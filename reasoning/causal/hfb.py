import json,numpy as np
I=json.load(open('items32.json'));R={k:np.array(v) for k,v in json.load(open('out32/b/scores.json')).items()};rng=np.random.default_rng(0)
ix=lambda ts:[k for k,i in enumerate(I) if i['task'] in ts]
def side(S,ks): return np.array([(S[k,0]-S[k,1])*(1 if I[k]['hf']=='hand' else -1) for k in ks])
def push(n,bn,a,ts):
    ks=ix(ts);B=R['base'];return np.concatenate([side(R[f'ans_{n}_{bn}_+{a}'],ks)-side(B,ks),-(side(R[f'ans_{n}_{bn}_-{a}'],ks)-side(B,ks))])
def flips(n,bn,a,ts):
    ks=ix(ts);f=[]
    for c,s in [(f'ans_{n}_{bn}_+{a}',1),(f'ans_{n}_{bn}_-{a}',-1)]:
        h0,h1=side(R['base'],ks)*s,side(R[c],ks)*s;f+=list((h1[h0<0]>0).astype(float))
    return round(float(np.mean(f)),2)
for bn in ['late','wide']:
  for a in [6,8,12]:
    for n in ['hf','rand']:
      if f'ans_{n}_{bn}_-{a}' in R:
        xb,xw=push(n,bn,a,['effector','pain']),push(n,bn,a,['hf_word']); print(bn,a,n,'body %.2f word %.2f'%(xb.mean(),xw.mean()),'flips',flips(n,bn,a,['effector','pain']),flips(n,bn,a,['hf_word']))
