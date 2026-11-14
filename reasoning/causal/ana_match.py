import json, numpy as np, torch
rng=np.random.default_rng(2)
def load(d):
    P=json.load(open(f'{d}/patch.json')); IT={it['id']:it for it in torch.load(f'{d}/atp.pt',weights_only=False)['items']}; return P,IT
P1,I1=load('attr32w'); P2,I2=load('attr32l')
L=[int(k) for k,v in P1.items() if I1[int(k)]['task']=='limb' and abs(v['ldt']-v['ld0'])>=1]; T=[int(k) for k,v in P2.items() if abs(v['ldt']-v['ld0'])>=1]
def rat(P,ids,k): return sum(P[str(i)][k]-P[str(i)]['ld0'] for i in ids)/sum(P[str(i)]['ldt']-P[str(i)]['ld0'] for i in ids)
def boot(P,ids,k,n=2000):
    pr=sorted({i//2 for i in ids}); by={p:[i for i in ids if i//2==p] for p in pr}
    return np.array([rat(P,[i for j in rng.choice(len(pr),len(pr)) for i in by[pr[j]]],k) for _ in range(n)])
out={}
for site in ['swap|q1','swap|q2','swap|q3','swap|q4','swap|all','all|all','last|all']:
    a=boot(P1,L,site+'|PU'); b=boot(P2,T,site+'|PU'); d=a-b; p=2*min((d<=0).mean(),(d>=0).mean())
    ra=np.mean([rat(P1,L,f'{site}|R{r}') for r in range(10)]); rb=np.mean([rat(P2,T,f'{site}|R{r}') for r in range(10)])
    out[site]=dict(limb=[rat(P1,L,site+'|PU'),*np.percentile(a,[2.5,97.5])],letter=[rat(P2,T,site+'|PU'),*np.percentile(b,[2.5,97.5])],diff=[float(np.mean(d)),*np.percentile(d,[2.5,97.5])],p=float(p),
        limb_R=ra, letter_R=rb, limb_OU=rat(P1,L,site+'|OU'), letter_OU=rat(P2,T,site+'|OU'), limb_full=rat(P1,L,site+'|full'), letter_full=rat(P2,T,site+'|full'))
    o=out[site]; print(f"{site:9s} limb {o['limb'][0]:.2f} [{o['limb'][1]:.2f},{o['limb'][2]:.2f}]  letter {o['letter'][0]:.2f} [{o['letter'][1]:.2f},{o['letter'][2]:.2f}]  diff {o['diff'][0]:.2f} [{o['diff'][1]:.2f},{o['diff'][2]:.2f}] p={p:.3f} | R {ra:.2f}/{rb:.2f} OU {o['limb_OU']:.2f}/{o['letter_OU']:.2f}")
out['n']=dict(limb=len(L),letter=len(T)); json.dump(out,open('attr32l/match.json','w'),indent=1,default=float)
