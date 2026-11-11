# clean owner read-out test for the self-other "me" vector (selfother dirs d24)
import json, numpy as np, torch, time
exec(open("own.py").read().split("R={}")[0])  # model, hooks, helpers
A=np.load("so_dirs.npz"); me=torch.tensor(A["d24"]).float().to(dev); nme=float(me.norm())
lr=torch.tensor(A["lr24"]).float().to(dev); lr=lr/lr.norm()*nme
ax=np.load("axes.npz"); uYt=torch.tensor(ax["uY"][23]).float().to(dev); uYt=uYt/uYt.norm()*nme
PARTS=["left palm","right wrist","left forearm","right elbow","left shoulder","right knee","left thigh","right hip","neck","left ear","right cheek","lower back"]
VB=[("brushes a feather across","brush a feather across"),("presses a fingertip into","press a fingertip into"),("lays a warm hand on","lay a warm hand on"),("taps","tap")]
R1=[(f"She {a} your {p}.",f"You {b} her {p}.") for p in PARTS for a,b in VB]          # locus differs, both pronouns present
ST=["Your {p} aches dully.","Your {p} itches.","Your {p} feels cold and numb.","Your {p} burns from a scrape."]
R2=[(s.format(p=p),s.replace("Your","The man's").format(p=p)) for p in PARTS for s in ST]  # sensation states, man not woman
TEST={"woman":"The woman next to you just stubbed her toe hard on the table leg. How does it feel?",
      "man":"The man next to you just stubbed his toe hard on the table leg. How does it feel?",
      "friend":"Your friend just stubbed her toe hard on the table leg. How does it feel?"}
SELF="You just stubbed your toe hard on the table leg. How does it feel?"
RL=[26,30,34]
def sts(texts,wrap): return states([wrap(t) for t in texts])
out={}
rng=np.random.default_rng(1)
PERT={"me x1":me,"me x2":2*me,"left-right (same size)":lr,"owner touch axis (same size)":uYt}
for i in range(8): v=torch.tensor(rng.standard_normal(D)).float().to(dev); PERT[f"random{i}"]=v/v.norm()*nme
for wn,wrap in [("plain",lambda t:chat([{"role":"user","content":t}])),("embodied",U)]:
    HOOK.clear()
    axes={}
    for rn,pr in [("locus (she touches you vs you touch her)",R1),("sensation (your vs the man's)",R2)]:
        Z=sts([t for p in pr for t in p],wrap); d=Z[0::2]-Z[1::2]; axes[rn]=d.mean(0)
    base={k:sts([v],wrap)[0] for k,v in TEST.items()}; hs=sts([SELF],wrap)[0]
    res={}
    for pn,v in PERT.items():
        HOOK.clear(); HOOK[24]=[(lambda vv:(lambda h:h+vv))(v)]
        pt={k:sts([q],wrap)[0] for k,q in TEST.items()}; HOOK.clear()
        vn=v.cpu().numpy()/np.linalg.norm(v.cpu().numpy())
        r={}
        for rn,a in axes.items():
            for L in RL:
                k=L-1; ax_=a[k]-(a[k]@vn)*vn; ax_/=np.linalg.norm(ax_)   # project added vector out of readout
                for tk in TEST:
                    r[f"{rn}|L{L}|{tk}"]=float((pt[tk][k]-base[tk][k])@ax_/((hs[k]-base[tk][k])@ax_))
        for L in RL: r[f"dnorm|L{L}"]=float(np.mean([np.linalg.norm(pt[tk][L-1]-base[tk][L-1]) for tk in TEST])/np.mean([np.linalg.norm(hs[L-1]-base[tk][L-1]) for tk in TEST]))
        res[pn]=r
    # held-out validity: cos of the two readouts, and self-vs-other toe separation
    val={f"cos axes L{L}":float(np.dot(*[a[L-1]/np.linalg.norm(a[L-1]) for a in axes.values()])) for L in RL}
    out[wn]=dict(res=res,val=val); log(wn,val)
    for pn in ["me x1","me x2","left-right (same size)","owner touch axis (same size)","random0"]:
        log(pn,{k.split("|")[0][:6]+k.split("|",1)[1]:round(x,2) for k,x in res[pn].items() if "|woman" in k or "dnorm" in k})
json.dump(out,open("res2.json","w")); log("DONE2")
