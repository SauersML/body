import sys, numpy as np, torch, json
D=sys.argv[1]; A=torch.load(f"{D}/atp.pt",weights_only=False); SN=A['SN']; IT={it['id']:it for it in A['items']}
out={}
for fm in ['body','ctl']:
    rows=[]
    for r in A['res']:
        if IT[r['id']]['fam']!=fm or abs(r['ldt']-r['ld'])<1: continue
        g=r['ldt']-r['ld']; full=r['full'].astype(float)  # L x T
        # per layer: effect share = sum_t sub / sum_t full ; energy share = energy-weighted mean of en over tokens
        fs=full.sum(-1)
        ok=np.abs(fs)>0.2*abs(g)
        sub=r['sub'].astype(float).sum(-1); en=r['en'].astype(float).mean(-1)
        rows.append([np.where(ok, sub[j]/np.where(ok,fs,1), np.nan) for j in range(len(SN))]+[en[j] for j in range(len(SN))])
    R=np.array(rows)  # n x (2S) x L
    S=len(SN); eff=np.nanmean(R[:, :S],0); en=R[:, S:].mean(0)
    out[fm]=dict(eff_PU=eff[0].tolist(), eff_OU=eff[1].tolist(), eff_R=eff[2:].mean(0).tolist(), en_PU=en[0].tolist(), en_OU=en[1].tolist(), en_R=en[2:].mean(0).tolist())
    L=len(eff[0]); hi=slice(L//2, L)
    print(fm, len(R), 'late-half: effect share PU %.3f OU %.3f R %.3f | energy share PU %.3f OU %.3f R %.3f'%(np.nanmean(eff[0][hi]),np.nanmean(eff[1][hi]),np.nanmean(eff[2:].mean(0)[hi]),en[0][hi].mean(),en[1][hi].mean(),en[2:].mean(0)[hi].mean()))
json.dump(out,open(f"{D}/energy.json","w"))
