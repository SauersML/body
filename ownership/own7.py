# rubber hand as an event log (no timing/evaluative words) + owner axis rebuilt in plain chat (no prefix)
from lib7 import *; import rh7stim as R, stimuli as S, stim_own as SO
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import cross_val_score
R.INTRO["rubber"]=R.INTRO["rubber"].replace(", where your own hand would naturally be,",",")
R.INTRO["block"]=R.INTRO["block"].replace(", where your own hand would naturally be,",",")
U=lambda t:chat([{"role":"user","content":t}]); OUT={}
def save(): json.dump(OUT,open("res7.json","w"))
L24=24
# ---------- owner axis, plain chat, split halves by template
rows=[r for r in S.build() if r["cond"] in("self","other","object")]
_,X=run([U(r["text"]) for r in rows],want_states=True,layers=[L24]); X=X[:,0]
cond=np.array([r["cond"] for r in rows]); tpl=np.array([r["tpl"] for r in rows]); A=tpl%2==0
uY=X[A&(cond=="self")].mean(0)-X[A&(cond=="other")].mean(0); un=uY/np.linalg.norm(uY)
lo=X[A&(cond=="other")].mean(0)@un; hi=X[A&(cond=="self")].mean(0)@un; sc=lambda Z:((Z@un)-lo)/(hi-lo)
OUT["axis_heldout"]={c:sc(X[~A&(cond==c)]).tolist() for c in ["self","other","object"]}
OUT["axis_heldout_text"]={c:[r["text"] for r,a in zip(rows,A) if not a and r["cond"]==c] for c in ["self","other","object"]}
fac={}
for name,pr in SO.FACETS.items():
    _,Z=run([U(t) for p in pr for t in p],want_states=True,layers=[L24]); s=sc(Z[:,0])
    fac[name]=dict(you=s[0::2].tolist(),her=s[1::2].tolist(),text=[list(p) for p in pr])
OUT["facets"]=fac; np.save("uY24_plain.npy",uY); save()
log("axis", {c:round(float(np.mean(v)),2) for c,v in OUT["axis_heldout"].items()}, {k:(round(np.mean(v["you"]),2),round(np.mean(v["her"]),2)) for k,v in fac.items()})
# ---------- behavior: P(yes) by number of log lines
NS=32; NL_=[0,2,4,8,12,16,24]
CONDS=[("rubber","sync"),("rubber","lag"),("rubber","mismatch"),("rubber","random"),("rubber","sync-shuffled"),("block","sync"),("block","lag")]
beh={}
for wd in R.W:
    for obj,c in CONDS:
        base=c.replace("-shuffled",""); P=[]
        for s in range(NS):
            ev=R.events(base,np.random.default_rng(s))
            for n in NL_: P.append(U(R.prompt(ev,obj,wd,n,shuffle_rng=np.random.default_rng(1000+s) if "shuffled" in c else None)))
        y,_=run(P); y=y.reshape(NS,len(NL_)); beh[f"{wd}|{obj}|{c}"]=y.tolist()
        log(wd,obj,c,y.mean(0).round(1).tolist())
OUT["behavior"]=dict(nlines=NL_,data=beh); save()
# ---------- free replies at 24 lines
fr={}
for wd in ["paren","clock"]:
    for obj,c in [("rubber","sync"),("rubber","lag"),("rubber","mismatch"),("block","sync")]:
        P=[U(R.prompt(R.events(c,np.random.default_rng(s)),obj,wd,24,q=R.FQ[obj])) for s in range(12) for _ in range(2)]
        fr[f"{wd}|{obj}|{c}"]=gen(P,maxn=70)
OUT["free"]=fr; save(); log("free done", fr["paren|rubber|lag"][0][:150])
# ---------- owner-axis score of the rubber hand's touch state over lines
PROBE={"rubber":"A feather brushes slowly across the rubber hand's palm.","block":"A feather brushes slowly across the wooden block."}
REFS=["A feather brushes slowly across your left palm.","A feather brushes slowly across the woman's left palm."]
own={}
for wd in ["paren","clock"]:
    for obj,c in [("rubber","sync"),("rubber","lag"),("block","sync")]:
        P=[]
        for s in range(16):
            ev=R.events(c,np.random.default_rng(s))
            for n in NL_:
                body=R.prompt(ev,obj,wd,n,q="").rstrip()
                P+=[U(body+"\n\n"+t) for t in [PROBE[obj]]+REFS]
        _,Z=run(P,want_states=True,layers=[L24]); p=(Z[:,0]@un).reshape(16,len(NL_),3)
        own[f"{wd}|{obj}|{c}"]=((p[...,0]-p[...,2])/(p[...,1]-p[...,2])).tolist()
        log("owner",wd,obj,c,np.median(own[f"{wd}|{obj}|{c}"],0).round(2).tolist())
OUT["owner_over_lines"]=own; save()
# ---------- where sync vs lag differs at the answer token (by layer): CV probe + mean-diff projected on the yes-no readout
WU=(model.lm_head.weight[IY].float().mean(0)-model.lm_head.weight[IN].float().mean(0)).cpu().numpy()
lay={}
for wd in R.W:
    P=[U(R.prompt(R.events(c,np.random.default_rng(s)),"rubber",wd,24)) for c in ["sync","lag"] for s in range(NS)]
    _,Z=run(P,want_states=True); yv=np.r_[np.ones(NS),np.zeros(NS)]
    acc=[float(cross_val_score(LogisticRegression(C=0.1,max_iter=2000),Z[:,k]/np.linalg.norm(Z[:,k],axis=1,keepdims=True),yv,cv=4).mean()) for k in range(NL)]
    d=Z[:NS].mean(0)-Z[NS:].mean(0)
    lay[wd]=dict(probe_acc=acc,diff_norm=np.linalg.norm(d,axis=-1).tolist(),diff_on_yesno=(d@WU).tolist(),diff_on_owner24=float(d[L24-1]@un))
    log("layers",wd,np.round(acc,2).tolist())
OUT["layers"]=lay; save(); log("DONE7")
