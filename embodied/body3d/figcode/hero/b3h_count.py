import pickle,sys,json
sys.path.insert(0,'.')
import hum as H
d=pickle.load(open('../x8abs/steer_confirm_Qwen3-8B_named.pkl','rb'));r=d['runs']
S=U=0;n=0;per={}
for t in d['targets']:
    p=H.PARTS.index(t)
    for i,(es,eu) in enumerate(zip(r[t],r['none'])):
        hs=any(p in x[0] for x in es['tc'][-6:]); hu=any(p in x[0] for x in eu['tc'][-6:])
        S+=hs;U+=hu;n+=1;per[(t,i)]=(hs,hu)
print(S,U,n)
print([k for k,v in per.items() if v==(True,False)][:40])
json.dump(dict(S=S,U=U,n=n),open('$BODY_WORK/fig/b3h_counts.json','w'))
from statsmodels.stats.proportion import proportion_confint as pc
from scipy.stats import binomtest
b=sum(1 for v in per.values() if v==(True,False)); c=sum(1 for v in per.values() if v==(False,True))
p=binomtest(b,b+c,0.5).pvalue
out=dict(S=S,U=U,n=n,ci_s=pc(S,n,method='wilson'),ci_u=pc(U,n,method='wilson'),b=b,c=c,p_mcnemar=p)
print(out); json.dump(out,open('$BODY_WORK/fig/b3h_counts.json','w'))
