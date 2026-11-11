import numpy as np
exec(open('a3.py').read().split("hr=reg==")[0].replace("LAY=int(os.environ.get('LAY',27))","LAY=24"))
hr=reg=='handR'; fl=reg=='footL'; wr=np.array(['wrist' in p for p in ph])&hr; hand=hr&~wr
def raw(n): return np.load(f"{D}/H2/{n.replace(':','__')}.npy")[0].astype(np.float32)
th=np.mean([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('gone')],0)
tf=np.mean([ (L[n][0][fl]-NEU[fl]).mean(0) for n in grp('footgone')],0)
tc=np.mean([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('ctrl')],0)
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b)); print('L24 cos hand foot',cos(th,tf),'ctrl',cos(th,tc))
d=th/np.linalg.norm(th)+tf/np.linalg.norm(tf); d/=np.linalg.norm(d)
Hn=raw('neutral:0:0'); nrm=float(np.linalg.norm(Hn,axis=1).mean()); print('norm',nrm, 'shift norm hand', np.linalg.norm(th), 'foot', np.linalg.norm(tf))
rng=np.random.default_rng(0); r=rng.standard_normal(d.shape); r/=np.linalg.norm(r)
# also hand-only direction for leave-out test: test on foot touches with hand-derived dir
dh=th/np.linalg.norm(th)
np.savez('pod/dirs.npz',loss=d,hand=dh,rand=r,norm=nrm,shift=float(np.linalg.norm(th)))
