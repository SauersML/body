import os
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
Z=states()
sides=["inside","thumb","back","little"]
def side(l):
    p=l["phrase"]; return 0 if "the inside" in p else 1 if "thumb side" in p else 2 if "the back" in p else 3
for arm in ["left","right"]:
  fa=np.array([i for i,l in enumerate(L) if f"{arm} forearm" in l["phrase"]]); sd=np.array([side(L[i]) for i in fa])
  x=np.abs(P[fa,0]); pos=np.round((x-x.min())/(x.max()-x.min())*9).astype(int)
  dj=np.minimum(x-x.min(),x.max()-x); 
  for nm in ["words","8B","32B"]:
    D=squareform(pdist(Z[nm][fa])); np.fill_diagonal(D,np.inf); o=np.argsort(D,1)
    for k in [1,5]:
        nn=o[:,:k]; same=np.mean(sd[nn]==sd[:,None]); gap=np.mean(np.abs(x[nn]-x[:,None])); gstep=np.mean(np.abs(pos[nn]-pos[:,None]))
        gj=np.mean(np.abs(dj[nn]-dj[:,None]))
        # chance gap for random partner on same side
        rg=np.mean([np.abs(x[j]-x[i]) for i in range(40) for j in range(40) if i!=j]); rgs=np.mean([np.abs(x[j]-x[i]) for i in range(40) for j in range(40) if i!=j and sd[i]==sd[j]])
        rj=np.mean([np.abs(dj[j]-dj[i]) for i in range(40) for j in range(40) if i!=j])
        print(arm,nm,k,f"same side {same:.2f} | along gap {gap:.1f}cm (random {rg:.1f}, random same side {rgs:.1f}) steps {gstep:.1f} | joint-dist gap {gj:.1f} (rand {rj:.1f})")
  print("length cm", x.max()-x.min(), "positions", sorted(set(pos)))
