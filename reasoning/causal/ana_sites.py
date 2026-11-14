import json, sys, numpy as np
D=sys.argv[1]; P=json.load(open(f'{D}/patch.json')); IT={it['id']:it for it in __import__('torch').load(f'{D}/atp.pt',weights_only=False)['items']}
G={'body':['limb','reach','posture','rearr_body'],'spell':['spell_len','spell_vowel'],'obj':['reach_obj','posture_obj','rearr_obj']}
ids=[int(k) for k,v in P.items() if abs(v['ldt']-v['ld0'])>=1]
sites=sorted({k.rsplit('|',1)[0] for k in P[str(ids[0])] if '|' in k})
fr=lambda i,k: (P[str(i)][k]-P[str(i)]['ld0'])/(P[str(i)]['ldt']-P[str(i)]['ld0'])
print('site       ' + '  '.join(f'{g:>16s}' for g in G))
for st in sites:
    row=[]
    for g,ts in G.items():
        sel=[i for i in ids if IT[i]['task'] in ts]
        row.append('%5.2f %5.2f %5.2f'%(np.mean([fr(i,st+'|PU') for i in sel]),np.mean([fr(i,st+'|OU') for i in sel]),np.mean([np.mean([fr(i,f'{st}|R{r}') for r in range(10)]) for i in sel])))
    print(f'{st:10s} '+'  '.join(row))
