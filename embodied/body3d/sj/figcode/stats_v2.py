import numpy as np, json
DD="$BODY_RUNS"
D=np.load(f"{DD}/sj_scores.npz"); scene=np.load(f"{DD}/selfjudge.npz")["scene"]
v,p,y,task,says=D["verbal"],D["probe"],D["y"].astype(int),D["task"],D["says"].astype(bool)
rng=np.random.default_rng(1); pv=np.zeros(len(y),bool)
def bacc(s,th,i): return 0.5*((s[i&(y==1)]>th).mean()+(s[i&(y==0)]<=th).mean())
for t in ["reach","poke_reach","grasp"]:
    i=task==t; sc=np.unique(scene[i]); folds=np.array_split(rng.permutation(sc),8)
    for f in folds:
        te=i&np.isin(scene,f); tr=i&~np.isin(scene,f); ths=np.quantile(v[tr],np.linspace(.02,.98,97))
        th=ths[np.argmax([bacc(v,h,tr) for h in ths])]; pv[te]=v[te]>th
    for _ in range(2000): rng.choice(sc,len(sc))
    print(t, round((pv==y)[i].mean(),3), round((says==y)[i].mean(),3), round(((p>0)==y)[i].mean(),3))
yb=y.astype(bool); m=dict(yes_succ=says[yb], yes_fail=says[~yb])
acc=dict(raw=(says==yb), recal=(pv==yb), probe=((p>0)==yb), never=~yb)
cl=np.array([f"{a}|{b}" for a,b in zip(task,scene)]); U=np.unique(cl); idx={u:np.where(cl==u)[0] for u in U}
r=np.random.default_rng(0); B={k:[] for k in ["yes_succ","yes_fail","raw","recal","probe","d"]}
for _ in range(4000):
    I=np.concatenate([idx[u] for u in r.choice(U,len(U))])
    B["yes_succ"].append(says[I][yb[I]].mean()); B["yes_fail"].append(says[I][~yb[I]].mean())
    for k in ["raw","recal","probe"]: B[k].append(acc[k][I].mean())
    B["d"].append(acc["recal"][I].mean()-acc["raw"][I].mean())
out={}
pt=dict(yes_succ=says[yb].mean(),yes_fail=says[~yb].mean(),**{k:acc[k].mean() for k in ["raw","recal","probe","never"]})
for k,val in pt.items(): out[k]=dict(v=float(val),ci=[float(x) for x in np.percentile(B[k],[2.5,97.5])] if k in B else None)
d=np.array(B["d"]); out["p_recal_vs_raw"]=float(min(1,2*min((d<=0).mean(),(d>=0).mean()))); out["n"]=int(len(y)); out["n_succ"]=int(yb.sum())
json.dump(out,open("stats_v2.json","w"),indent=1); print(json.dumps(out))
