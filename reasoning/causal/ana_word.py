import json, sys, numpy as np, torch
D=sys.argv[1]; s=json.load(open(f'{D}/stats.json')); P=json.load(open(f'{D}/patch.json'))
A=torch.load(f'{D}/atp.pt',weights_only=False); IT={it['id']:it for it in A['items']}; R={r['id']:r for r in A['res']}
NL=A['res'][0]['full'].shape[0]; q1=range(0, NL//4)
for t,v in s['tasks'].items():
    ids=[i for i in IT if IT[i]['task']==t and str(i) in P and abs(P[str(i)]['ldt']-P[str(i)]['ld0'])>=1]
    fr=lambda i,k: (P[str(i)][k]-P[str(i)]['ld0'])/(P[str(i)]['ldt']-P[str(i)]['ld0'])
    en=[np.mean([R[i]['en'][0][l, IT[i]['diff']].astype(float).mean() for l in q1]) for i in ids]
    enO=[np.mean([R[i]['en'][1][l, IT[i]['diff']].astype(float).mean() for l in q1]) for i in ids]
    print(f"{t:12s} n={len(ids):3d} swap|q1: PU {np.mean([fr(i,'swap|q1|PU') for i in ids]):.2f} OU {np.mean([fr(i,'swap|q1|OU') for i in ids]):.2f} R {np.mean([np.mean([fr(i,f'swap|q1|R{r}') for r in range(10)]) for i in ids]):.2f} | energy at swap tok q1: PU {np.mean(en):.2f} OU {np.mean(enO):.2f}")
