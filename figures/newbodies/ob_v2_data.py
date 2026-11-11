import os
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
from PIL import Image
CF=os.path.expandvars("$BODY_WORK/")
sys.path.insert(0,CF+"fig"); from ob_coords import C as CO, EDGES
FD=CF+"fonts/inter/extras/ttf/"
for f in ["Inter-Regular","Inter-Medium","Inter-SemiBold","Inter-Bold","InterDisplay-SemiBold","InterDisplay-Bold"]: fm.fontManager.addfont(FD+f+".ttf")
plt.rcParams.update({"font.family":"Inter"})
INK="#1B2433"; MID="#5B6474"; FAINT="#9AA3B0"; HUM="#C3CAD4"; HUML="#DDE2E8"; CARD="#F2F4F7"
ACC="#D9572B"; ACC2="#F3C9B6"; RND="#8E99A8"; NONE="#D5DAE1"
SO=CF+"body/somato/out3_32b/"
sc=json.load(open(SO+"steer4_L52_scores.json"))
# ---------- geometry: linear read-out of body position, fit on the 54 human places only
d=np.load(CF+"fig/ob_geom_L52.npy",allow_pickle=True).item(); keys=d["keys"]; HMx=d["human"]
nrm=lambda X: X/np.linalg.norm(X,axis=1,keepdims=True)
Y=np.array([CO[k] for k in keys],float); Ya=np.c_[np.abs(Y[:,0]),Y[:,1],np.sign(Y[:,0])]
Xh=nrm(HMx); LAM=0.01
def fit(m):
    mu=Xh[m].mean(0); Ym=Ya[m].mean(0); Xc=Xh[m]-mu; a=np.linalg.solve(Xc@Xc.T+LAM*np.eye(m.sum()),Ya[m]-Ym)
    return lambda Z:(nrm(Z)-mu)@Xc.T@a+Ym
typ=[k[:-1] if k[-1] in "LR" and k!="neckB" else k for k in keys]
pr=np.zeros_like(Ya)
for t in set(typ):
    m=np.array([x!=t for x in typ]); pr[~m]=fit(m)(HMx[~m])     # held-out: every place type left out in turn
r2=1-((pr-Ya)**2).sum(0)/((Ya-Ya.mean(0))**2).sum(0)
print("leave-place-out R2 |x|,y,side:",r2.round(2))
F=fit(np.ones(54,bool))
pos=lambda z:(np.sign(z[2])*z[0], z[1])
HP={k:pos(z) for k,z in zip(keys,pr)}
NEW={}
for b in ["robot","octopus","tail"]:
    parts,Pm=d[b]; Z=F(Pm); NEW[b]={p:pos(z) for p,z in zip(parts,Z)}
    print(b,{p:(round(v[0],1),round(v[1],1)) for p,v in NEW[b].items()})
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
NAME={"heel":"heel","ankle":"ankle","sole":"sole","palm":"palm","backhand":"knuckles","thumb":"thumb","lips":"lips","ear":"ear","forehead":"forehead",
 "crown":"scalp","chest":"chest","upback":"upper back","lowback":"lower back","forearm":"forearm","upperarm":"upper arm","knee":"knee","ring":"ring finger",
 "little":"little finger","index":"index finger","cheek":"cheek","neckF":"throat","neckB":"nape","belly":"belly","toes":"toes"}
def top3(b,p):
    parts,Pm=d[b]; v=Pm[parts.index(p)]; c=np.array([cos(v,h) for h in HMx]); out=[]; seen=set()
    for j in np.argsort(-c):
        t=typ[j]
        if t in seen: continue
        seen.add(t); out.append((NAME.get(t,t),c[j]))
        if len(out)==3: break
    return out
