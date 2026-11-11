import os
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from scipy.stats import spearmanr
Z=states()
for r,ix in REG.items():
    ix=np.array(ix); Ds=squareform(pdist(P[ix])); rnd=Ds[np.triu_indices(len(ix),1)].mean()
    s=f"{r:9s} n={len(ix):3d} rand3NN={rnd:5.1f} |"
    for nm in ["8B","32B","words"]:
        Dz=squareform(pdist(Z[nm][ix])); rho=spearmanr(Dz[np.triu_indices(len(ix),1)],Ds[np.triu_indices(len(ix),1)])[0]
        np.fill_diagonal(Dz,np.inf); nn=np.argsort(Dz,1)[:,:3]; d=np.mean([Ds[i,nn[i]].mean() for i in range(len(ix))])
        s+=f" {nm}: rho {rho:.2f} 3NN {d:4.1f}cm |"
    print(s)
