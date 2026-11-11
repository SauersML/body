import os
import json,numpy as np,sys
sys.path.insert(0,os.path.expandvars('$BODY_REPO/map/somato')); import stimuli as S
B=os.path.expandvars('$BODY_REPO/map/somato/')
L=int(sys.argv[1]) if len(sys.argv)>1 else 52
rows2=json.load(open(B+'out2_32b/rows.json')); H2=np.load(B+'out2_32b/H_rs.npy',mmap_mode='r')
keys=[l['key'] for l in S.LOCS]
HM=np.stack([np.asarray(H2[[k for k,r in enumerate(rows2) if r['cond']=='self' and r['li']==i],L],np.float32).mean(0) for i in range(54)])
HM-=HM.mean(0)
H=np.load(B+'out3_32b/H3.npy',mmap_mode='r'); rows=json.load(open(B+'out3_32b/rows3.json'))
out={'keys':keys,'human':HM}
cos=lambda a,b: a@b/np.linalg.norm(a)/np.linalg.norm(b)
for b in ['robot','octopus','tail']:
    g=[k for k,r in enumerate(rows) if r['group']=='body_'+b]; loc=np.array([rows[k]['loc'] for k in g])
    parts=list(dict.fromkeys(loc)); X=np.asarray(H[g,L],np.float32)
    Pm=np.stack([X[loc==p].mean(0) for p in parts]); Pm-=Pm.mean(0)
    out[b]=(parts,Pm)
    for i,p in enumerate(parts):
        c=np.array([cos(Pm[i],h) for h in HM]); o=np.argsort(-c)[:5]
        print(b,p,[(keys[j],round(float(c[j]),3)) for j in o])
np.save(os.path.expandvars('$BODY_REPO/figures/out/ob_geom_L52.npy'),out,allow_pickle=True)
print(H2.shape)
