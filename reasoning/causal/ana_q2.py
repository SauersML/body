import json, sys, numpy as np, torch
D=sys.argv[1]; site=sys.argv[2] if len(sys.argv)>2 else 'swap|q2'
P=json.load(open(f'{D}/patch.json')); IT={it['id']:it for it in torch.load(f'{D}/atp.pt',weights_only=False)['items']}
ids=[int(k) for k,v in P.items() if abs(v['ldt']-v['ld0'])>=1]; rng=np.random.default_rng(0)
fr=lambda i,k: (P[str(i)][k]-P[str(i)]['ld0'])/(P[str(i)]['ldt']-P[str(i)]['ld0'])
R=lambda i: np.mean([fr(i,f'{site}|R{r}') for r in range(10)])
for t in sorted({IT[i]['task'] for i in ids}):
    sel=[i for i in ids if IT[i]['task']==t]; print(f"{t:12s} PU {np.mean([fr(i,site+'|PU') for i in sel]):.2f} OU {np.mean([fr(i,site+'|OU') for i in sel]):.2f} R {np.mean([R(i) for i in sel]):.2f} full {np.mean([fr(i,site+'|full') for i in sel]):.2f}")
B=[i for i in ids if IT[i]['task'] in ('limb','reach','posture','rearr_body')]; Sp=[i for i in ids if IT[i]['task'] in ('spell_len','spell_vowel')]
def bs(sel,f):
    pr=sorted({i//2 for i in sel}); v={p:[f(i) for i in sel if i//2==p] for p in pr}
    return np.array([np.mean([x for j in rng.choice(len(pr),len(pr)) for x in v[pr[j]]]) for _ in range(3000)]), np.mean([x for p in pr for x in v[p]])
bb,mb=bs(B,lambda i: fr(i,site+'|PU')); sb,ms=bs(Sp,lambda i: fr(i,site+'|PU')); rb,mr=bs(B,lambda i: fr(i,site+'|PU')-R(i))
d=bb-sb; print(f"body PU {mb:.2f} [{np.percentile(bb,2.5):.2f},{np.percentile(bb,97.5):.2f}]  spell {ms:.2f} [{np.percentile(sb,2.5):.2f},{np.percentile(sb,97.5):.2f}]  body-spell {mb-ms:.2f} [{np.percentile(d,2.5):.2f},{np.percentile(d,97.5):.2f}] p={2*min((d<=0).mean(),(d>=0).mean()):.4f}  PU-R body {mr:.2f} [{np.percentile(rb,2.5):.2f},{np.percentile(rb,97.5):.2f}]")
