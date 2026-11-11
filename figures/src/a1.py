import os
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
S1=load_stim(os.path.expandvars("$BODY_REPO/hires/v1/stim.py")); b1=Body(S1)
p2=[l["phrase"] for l in L]; IDX=[p2.index(l["phrase"]) for l in S1.LOCS]
d=os.path.expandvars("$BODY_REPO/hires/v2/")
Zs={}
for nm,f,k in [("8B","red8b_short.npz","reply|27|all"),("32B","red32_short.npz","reply|53|all"),("words","red8b_lex.npz","lex|0|all")]:
    R=np.load(d+f); A=R[k+"|A"].astype(np.float32); W=R[k+"|W"].astype(np.float32); Zs[nm]=embed(A[IDX],W[:45])[0]
NN={};DD={}
for nm,Z in Zs.items():
    D=squareform(pdist(Z)); np.fill_diagonal(D,np.inf); NN[nm]=np.argsort(D,1)[:,:5]; DD[nm]=np.array([b1.DF[i,NN[nm][i]].mean() for i in range(b1.n)])
    print(nm,"mean",DD[nm].mean().round(2),"median",np.median(DD[nm]).round(2))
np.savez(os.path.expandvars("$BODY_REPO/figures/src/nn.npz"),**{f"nn_{k}":v for k,v in NN.items()},**{f"d_{k}":v for k,v in DD.items()})
ph=b1.ph
import re
for q in ["tip of {p} nose","centre of {p} left palm","left knee","navel","belly","tip of {p} left big toe","left elbow","left shoulder","chin","left thigh","left heel","left wrist","middle of {p} chest","forehead","index finger"]:
    for i,p in enumerate(ph):
        if q in p and "right" not in p: print(f"{i:3d} 8B {DD['8B'][i]:5.1f} 32B {DD['32B'][i]:5.1f}  {p[:70]}  {b1.P[i]}")
