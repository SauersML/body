exec(open('a3.py').read().split("hr=reg==")[0])
hr=reg=='handR'; fl=reg=='footL'; wr=np.array(['wrist' in p for p in ph])&hr; hand=hr&~wr
def mean_states(c): return np.mean([L[n][0] for n in grp(c)],0)
G,Cg,F,Cf=[mean_states(c) for c in ['gone','ctrl','footgone','footctrl']]
th=(G[hand]-NEU[hand]).mean(0); tf=(F[fl]-NEU[fl]).mean(0); tc=(Cg[hand]-NEU[hand]).mean(0); tfc=(Cf[fl]-NEU[fl]).mean(0)
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('cos th tf',cos(th,tf),'th tc',cos(th,tc),'tf tfc',cos(tf,tfc),'tc tfc',cos(tc,tfc),'th tfc',cos(th,tfc))
print('norms/unit',[np.linalg.norm(x)/unit for x in [th,tf,tc,tfc]])
u=(th/np.linalg.norm(th)+tf/np.linalg.norm(tf)); u/=np.linalg.norm(u)
v=NEU[fl].mean(0)-NEU[hand].mean(0); v-=v@u*u; v/=np.linalg.norm(v)
P=lambda X: np.stack([X@u,X@v],1)/unit
out=dict(hn=P(NEU[hand]),hg=P(G[hand]),hc=P(Cg[hand]),fn=P(NEU[fl]),fg=P(F[fl]),fc=P(Cf[fl]),alln=P(NEU))
np.savez('dir2d.npz',**out)
for k,x in out.items(): print(k,x.mean(0).round(2),x.std(0).round(2))
print('cos tf tc',cos(tf,tc))
import numpy.linalg as la
# per-chat cos of hand-loss shifts vs foot-loss shift
for n in grp('gone'): print(n, round(cos((L[n][0][hand]-NEU[hand]).mean(0),tf),2), round(la.norm((L[n][0][hand]-NEU[hand]).mean(0))/unit,2))
for n in grp('ctrl'): print(n, round(cos((L[n][0][hand]-NEU[hand]).mean(0),tf),2), round(cos((L[n][0][hand]-NEU[hand]).mean(0),th),2))
k=list(ph).index('the middle of the back of right hand'); print('k',k)
gH=np.array([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('gone')]); cH=np.array([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('ctrl')])
fan=dict(hand=[(cos(x,tf),la.norm(x)/unit) for x in gH], glove=[(cos(x,tf),la.norm(x)/unit) for x in cH], foot=(1.0,la.norm(tf)/unit))
rng=np.random.default_rng(3); dims=rng.choice(4096,22,replace=False)
sd=NEU.std(0)
vec=dict(gone=(G[k][dims]/sd[dims]).tolist(), neu=(NEU[k][dims]/sd[dims]).tolist(), d=float(la.norm(G[k]-NEU[k])/unit))
json.dump(dict(fan=fan,vec=vec),open('fig4data.json','w')); print(vec['d'], fan['foot'])
print(np.round(vec['gone'],1)); print(np.round(vec['neu'],1))
