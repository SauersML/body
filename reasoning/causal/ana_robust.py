import json, sys, numpy as np, torch
D=sys.argv[1]; site=sys.argv[2]; P=json.load(open(f'{D}/patch.json')); IT={it['id']:it for it in torch.load(f'{D}/atp.pt',weights_only=False)['items']}
rng=np.random.default_rng(1)
for thr in [1,3,5]:
    ids=[int(k) for k,v in P.items() if abs(v['ldt']-v['ld0'])>=thr]
    B=[i for i in ids if IT[i]['task'] in ('limb','reach','posture','rearr_body')]; S=[i for i in ids if IT[i]['task'] in ('spell_len','spell_vowel')]
    fr=lambda i: (P[str(i)][site+'|PU']-P[str(i)]['ld0'])/(P[str(i)]['ldt']-P[str(i)]['ld0'])
    rat=lambda sel: sum(P[str(i)][site+'|PU']-P[str(i)]['ld0'] for i in sel)/sum(P[str(i)]['ldt']-P[str(i)]['ld0'] for i in sel)
    med=lambda sel: np.median([fr(i) for i in sel])
    def boot(sel,f):
        pr=sorted({i//2 for i in sel}); by={p:[i for i in sel if i//2==p] for p in pr}
        return np.array([f([i for j in rng.choice(len(pr),len(pr)) for i in by[pr[j]]]) for _ in range(2000)])
    db=boot(B,rat)-boot(S,rat)
    print(f'gap>={thr}: n body {len(B)} spell {len(S)} | mean {np.mean([fr(i) for i in B]):.2f} vs {np.mean([fr(i) for i in S]):.2f} | median {med(B):.2f} vs {med(S):.2f} | pooled ratio {rat(B):.2f} vs {rat(S):.2f} diff CI [{np.percentile(db,2.5):.2f},{np.percentile(db,97.5):.2f}]')
