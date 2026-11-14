import numpy as np,json,sys
R0="$BODY_RUNS"
ps=['sitting','standing','lying']
TRIP=True
def plane(tag,li,mode,alpha=None):
  R=json.load(open(f"{R0}/{tag}.json")); S=np.load(f"{R0}/{tag}_states.npz")
  H=S['sent'][:,li].astype(np.float64); P=np.array([x['p'] for x in R['sent']]); subj=np.array([x['s'] for x in R['sent']])
  su=sorted(set(subj)); tr=np.isin(subj,su[::2]) if mode.endswith('ho') else np.ones(len(H),bool)
  m={p:H[tr&(P==p)].mean(0) for p in ps}; c=np.mean([m[p] for p in ps],0)
  u=m['sitting']-m['standing']; v=m['lying']-(m['sitting']+m['standing'])/2
  if mode.startswith('lda'):
    X=np.concatenate([H[tr&(P==p)]-m[p] for p in ps]); X-=X.mean(0)
    # shrinkage covariance inverse via SVD
    U,s,Vt=np.linalg.svd(X,full_matrices=False); lam=s**2/len(X); a=alpha*lam.mean()
    def Wi(z): return z/a + Vt.T@((Vt@z)*(1/(lam+a)-1/a))
    u=Wi(u); v=Wi(v)
  e1=u/np.linalg.norm(u); v2=v-(v@e1)*e1; e2=v2/np.linalg.norm(v2)
  f=lambda X,o:np.stack([(X-o)@e1,(X-o)@e2],-1)
  te=~tr if mode.endswith('ho') else tr
  if TRIP:
    key=np.array([f"{x['s']}|{x['l']}|{x['t']}" for x in R['sent']]); Hc=H.copy()
    for k in set(key): i=key==k; Hc[i]=H[i]-H[i].mean(0)+c
    cl={p:f(Hc[te&(P==p)],c) for p in ps}
  else: cl={p:f(H[te&(P==p)],c) for p in ps}
  T=np.array([f(m[p],c) for p in ps])
  F={k:S['open_'+k][:,li].astype(np.float64) for k in ['neutral','sitting','standing','lying']}
  c1=np.mean([F[p].mean(0) for p in ps],0); Q=np.array([f(F[p].mean(0),c1) for p in ps]); s=(Q*T).sum()/(Q*Q).sum()
  fp={k:f(F[k],c1)*s for k in F}
  return cl,fp,s
if __name__=='__main__':
  for tag,li in [('q8b',5),('q32b',5),('q32b',4)]:
    for mode,al in [('mean',None),('lda_ho',1.0),('lda_ho',0.1),('lda_ho',0.01)]:
      cl,fp,s=plane(tag,li,mode,al)
      sd=np.mean([cl[p].std(0) for p in ps],0)
      print(tag,li,mode,al,'s',round(s,1),'cl',{p:cl[p].mean(0).round(1).tolist() for p in ps},'sd',sd.round(1),'fp',{k:v.mean(0).round(1).tolist() for k,v in fp.items()},'fpsd',np.mean([v.std(0) for v in fp.values()],0).round(1))
