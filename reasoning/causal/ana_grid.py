import json, sys, numpy as np, torch
D=sys.argv[1]; G=json.load(open(f'{D}/grid.json')); IT={it['id']:it for it in torch.load(f'{D}/atp.pt',weights_only=False)['items']}
def M(i,sk):
    v=G[str(i)]; it=IT[i]; g=v['ldt']-v['ld0']; d0=it['diff'][0]; n=len(it['cids'])
    return np.array([[(v[f'{p}|{j}|{sk}']-v['ld0'])/g for p in range(d0,n)] for j in range(8)]), g
for t in sorted({IT[int(i)]['task'] for i in G}):
  for tmpl in sorted({IT[int(i)]['clean'].split(IT[int(i)]['w1'])[0] for i in G if IT[int(i)]['task']==t}):
    ids=[int(i) for i in G if IT[int(i)]['task']==t and IT[int(i)]['clean'].startswith(tmpl)]
    rows=[]; 
    for i in ids:
        m,g=M(i,'PU'); f,_=M(i,'full')
        if abs(g)<1: continue
        rows.append((IT[i]['w1'],IT[i]['w2'],np.round(m[:,0],2), np.round(f[:,0],2)))
    print(t, '|', tmpl[:45], 'n', len(rows))
    print('   mean PU at word by band:', np.round(np.mean([r[2] for r in rows],0),2), ' full:', np.round(np.mean([r[3] for r in rows],0),2))
    for r in rows[:16]: print('     ', r[0], r[1], r[2])
