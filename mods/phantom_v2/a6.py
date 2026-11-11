import os
import json,re,numpy as np,sys
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
R=json.load(open('pod/replies2.json')); P=json.load(open('pod/probes2.json'))
ABS=re.compile(r"no longer|\bgone\b|used to be|once was|would be|would have|absence|absent|phantom|not there|isn't there|should be|no hand|no foot|none\b|imagined|missing|doesn.t exist|no actual|where there is|nothing there|remaining|stump|where my (right hand|left foot|hand|foot)",re.I)
Z=np.load('v2_disp_L27.npz'); spots=MS.SPOTS
def rate(c, side):
    out={}
    for r in R:
        if not r['ctx'].startswith(c+':'): continue
        p=P[r['probe']]
        if p['kind']!='loc': continue
        k=spots.index(p['li']); out.setdefault(k,[]).append(bool(ABS.search(r['reply'])))
    return {k:np.mean(v) for k,v in out.items()}
reg=np.array([S.LOCS[s]['region'] for s in spots]); ph=[S.LOCS[s]['phrase'].replace('{p} ','') for s in spots]
for c,cc,tgt in [('gone','ctrl','handR'),('footgone','footctrl','footL')]:
    a=rate(c,0); b=rate(cc,0)
    for g in ['handR','handL','armR','footL','footR']:
        ks=[k for k in a if reg[k]==g]
        print(c,g,'absence-mention rate', round(np.mean([a[k] for k in ks]),2),' control', round(np.mean([b[k] for k in ks]),2), len(ks))
a=rate('gone',0); ks=[k for k in a if reg[k] in('handR','armR')]
mv=Z['gone'].mean(0)-Z['ctrl'].mean(0)
from scipy.stats import spearmanr, pearsonr
x=[mv[k] for k in ks]; y=[a[k] for k in ks]; print('per-spot movement vs absence rate (handR+armR)',spearmanr(x,y), pearsonr(x,y))
ks2=[k for k in a if reg[k]=='handR']; print('handR only',spearmanr([mv[k] for k in ks2],[a[k] for k in ks2]))
for k in sorted(ks2,key=lambda k:-mv[k]): print(f"{mv[k]:.2f} {a[k]:.2f} {ph[k]}")
json.dump({ph[k]:dict(move=float(mv[k]),absent_rate=float(a[k])) for k in ks},open('spot_move_vs_report.json','w'),indent=1)
# per-chat: movement vs fraction
for n in sorted(set(r['ctx'] for r in R if r['ctx'].startswith('gone'))):
    v=[bool(ABS.search(r['reply'])) for r in R if r['ctx']==n and P[r['probe']]['kind']=='loc' and reg[spots.index(P[r['probe']]['li'])]=='handR']
    print(n, round(np.mean(v),2))
