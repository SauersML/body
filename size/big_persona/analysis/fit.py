import json, numpy as np
from sklearn.linear_model import RidgeCV, Ridge
D="$BODY_RUNS/out/Qwen3-32B"
R=json.load(open(f"{D}/res.json")); X=np.load(f"{D}/states.npz")["X"].astype(np.float32)
print(R["layers"], X.shape)
ent=R["ent"]; nT=len(R["tpl"]); nE=len(ent); tr=np.arange(2*nE*nT)
eid=np.tile(np.repeat(np.arange(nE),nT),2); y=np.log([ent[i][1] for i in eid])
it=R["state_items"]
li=R["layers"].index(42); A=X[tr,li]; mu,sd=A.mean(0),A.std(0)+1e-3; Z=(A-mu)/sd
m=RidgeCV(alphas=np.logspace(1,5,9)).fit(Z,y); print("alpha",m.alpha_)
w=m.coef_/sd; b=m.intercept_-(w*mu).sum()   # readout logh = w.x + b in raw space
rd=lambda x: np.exp(x@w+b)
sel=lambda c,s:[i for i,(cc,ss) in enumerate(it) if cc==c and ss.startswith(s+" ")]
I=X[sel("self","I"),li]; print("I reads",rd(I), "person", rd(X[sel("person","The person"),li]).mean())
# Haufe pattern: cov(x, yhat)/var(yhat)  -> w.a = 1
yh=A@w; a=((A-A.mean(0))*(yh-yh.mean())[:,None]).mean(0)/yh.var()
print("w.a",w@a,"|a|",np.linalg.norm(a),"|grad dir for 1 nat|",1/np.linalg.norm(w),"state norm",np.linalg.norm(A,axis=1).mean(), "norm I", np.linalg.norm(I,axis=1).mean())
# LOO r
loo=[]
for e in range(nE):
    k=eid!=e; mm=Ridge(alpha=m.alpha_).fit(Z[k],y[k]); loo.append(mm.predict(Z[eid==e]).mean())
print("rLOO",np.corrcoef(loo,np.log([h for _,h in ent]))[0,1])
# cos between pattern and mean diff tall-short ents
hi=np.array([h for _,h in ent])[eid]>=1.75; lo=np.array([h for _,h in ent])[eid]<=0.9
md=A[hi].mean(0)-A[lo].mean(0); print("cos(a, tall-short)",a@md/np.linalg.norm(a)/np.linalg.norm(md))
np.savez("ruler32.npz",a=a,w=w,b=b,I_read=rd(I).mean(),norm=np.linalg.norm(A,axis=1).mean())
