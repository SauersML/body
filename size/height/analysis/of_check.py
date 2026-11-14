import json,numpy as np,sys
from sklearn.linear_model import RidgeCV
D='$BODY_RUNS/out'
for s in ["0.6B","1.7B","4B","8B","14B","32B"]:
    R=json.load(open(f"{D}/Qwen3-{s}/res.json")); X=np.load(f"{D}/Qwen3-{s}/states.npz")["X"]; meta=R["meta"]
    kind=np.array([m[0] for m in meta]); Hh=np.array([m[1] for m in meta],float); out=[]
    for li,L in enumerate(R["layers"]):
        tr=np.char.startswith(kind,"ent_"); A=X[tr,li].astype(np.float32); mu,sd=A.mean(0),A.std(0)+1e-3
        m=RidgeCV(alphas=np.logspace(1,5,9)).fit((A-mu)/sd,np.log(Hh[tr]))
        rd=lambda msk: np.exp(m.predict((X[msk,li].astype(np.float32)-mu)/sd))
        ts=kind=="train_s"; p=rd(ts); hs=sorted(set(Hh[ts])); pm=[p[Hh[ts]==h].mean() for h in hs]
        en=[rd(kind==e+"_s").mean() for e in ["dog","child","player","giraffe"]]
        out.append((L,np.corrcoef(hs,pm)[0,1],np.polyfit(hs,pm,1)[0],np.corrcoef([50,120,190,250],en)[0,1],np.polyfit([50,120,190,250],en,1)[0],rd(kind=="self_s").mean()))
    print(s," ".join(f"L{L}: rS {a:.2f} slS {b:.2f} rE {c:.2f} slE {d:.2f} self {e:.0f}|" for L,a,b,c,d,e in out))
