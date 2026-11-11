import os
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
S1=load_stim(os.path.expandvars("$BODY_REPO/hires/v1/stim.py"))
p2=[l["phrase"] for l in L]; IDX=np.array([p2.index(l["phrase"]) for l in S1.LOCS]); n=len(IDX)
PP=P[IDX]; NN_=N[IDX]; ph=[p2[i] for i in IDX]; side=np.array([L[i]["side"] for i in IDX]); reg=[L[i]["region"] for i in IDX]
Pf=PP.copy(); Pf[:,0]=abs(Pf[:,0]); DF=squareform(pdist(Pf))
tw=-np.ones(n,int)
for i,p in enumerate(ph):
    q=p.replace("left","RIGHT").replace("right","left").replace("RIGHT","right")
    if q!=p and q in ph: tw[i]=ph.index(q)
print("twins",(tw>=0).sum(), "twin coords mirror ok", all(np.allclose(PP[i]*[-1,1,1],PP[tw[i]]) for i in range(n) if tw[i]>=0))
d=os.path.expandvars("$BODY_REPO/hires/v2/")
out=dict(P=PP,N=NN_,tw=tw,side=side,reg=np.array(reg),ph=np.array(ph))
for nm,f,k in [("8B","red8b_short.npz","reply|27|all"),("32B","red32_short.npz","reply|53|all"),("words","red8b_lex.npz","lex|0|all")]:
    R=np.load(d+f); A=R[k+"|A"].astype(np.float32); W=R[k+"|W"].astype(np.float32); Z=embed(A[IDX],W[:45])[0]
    D=squareform(pdist(Z)); np.fill_diagonal(D,np.inf)
    old=np.argsort(D,1)[:,:5]
    D2=D.copy(); ok=tw>=0; D2[np.where(ok)[0],tw[ok]]=np.inf
    nn=np.argsort(D2,1)[:,:5]; dd=np.array([DF[i,nn[i]].mean() for i in range(n)])
    print(nm, A.shape, "old5 %.1f"%np.mean([DF[i,old[i]].mean() for i in range(n)]), "no-twin mean %.1f median %.1f"%(dd.mean(),np.median(dd)), "pct<10 %.2f >30 %.2f"%((dd<10).mean(),(dd>30).mean()))
    out["nn_"+nm]=nn; out["d_"+nm]=dd; out["sim_"+nm]=np.sort(D2,1)[:,:5]
r=np.array([np.delete(DF[i],[i]+([tw[i]] if tw[i]>=0 else [])).mean() for i in range(n)])
out["d_random"]=r; print("random %.1f"%r.mean(), r.min(), r.max())
np.savez("nn2.npz",**out)
