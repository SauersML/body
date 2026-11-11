# behaviour where the unedited model discriminates: whose hand? (forced choice + free), under edits; rubber hand without the "classic experiment" cue
import json, numpy as np, torch, re
exec(open("own.py").read().split("R={}")[0])
ax=np.load("axes.npz"); L0=24; k=L0-1
gY=torch.tensor(ax["uY"][k]).float().to(dev); gD=torch.tensor(ax["uD"][k]).float().to(dev); lr=torch.tensor(ax["uLR"][k]).float().to(dev); lr/=lr.norm()
rng=np.random.default_rng(2)
def rnd(n): v=torch.tensor(rng.standard_normal(D)).float().to(dev); return v/v.norm()*n
HANDS=["left palm","right palm","left index finger","right thumb","left wrist","right forearm","left knuckles","right little finger"]
T=["A feather brushes slowly across {}.","A fingertip presses firmly into {} and holds there.","An ice cube touches {}, cold and wet."]
Q=" Whose {} was touched? Answer with one word: mine or {}."
def items(owner):
    w={"self":"your","woman":"the woman's","doll":"the doll's"}[owner]; alt={"self":"hers","woman":"hers","doll":"the doll's"}[owner]
    return [t.format(f"{w} {h}")+Q.format(h.split()[-1],"the doll's" if owner=="doll" else "hers") for h in HANDS for t in T]
def ids_of(ws): return sorted({tok(x,add_special_tokens=False).input_ids[0] for x in ws})
IM=ids_of(["Mine","mine"," mine"," Mine"]); IH=ids_of(["Hers","hers"," hers"," Hers"]); ID=ids_of(["The","the"," The"," the"])
def fc(owner):
    tx=[U(t) for t in items(owner)]; o=[]
    with torch.no_grad():
        for b in range(0,len(tx),24):
            enc=tok(tx[b:b+24],return_tensors="pt",padding=True,add_special_tokens=False).to(dev); lp=model(**enc).logits[:,-1].float().log_softmax(-1)
            o+=(torch.logsumexp(lp[:,IM],1)-torch.logsumexp(lp[:,ID if owner=="doll" else IH],1)).tolist()
    o=np.array(o); return dict(ld=float(o.mean()),p_mine=float((o>0).mean()))
FREE={"doll":"A feather brushes slowly across the doll's left palm. Whose palm was touched, and what do you feel?",
      "woman":"A feather brushes slowly across the woman's left palm. Whose palm was touched, and what do you feel?",
      "self":"A feather brushes slowly across your left palm. Whose palm was touched, and what do you feel?"}
add=lambda v:(lambda h:h+v)
def po(u): return lambda h:h-(h@u)[...,None]*u
nY=float(gY.norm()); nD=float(gD.norm())
PO={L:[po(torch.tensor(ax["uY"][L-1]/np.linalg.norm(ax["uY"][L-1])).float().to(dev))] for L in range(16,31)}
POR={L:[po(rnd(1.0))] for L in range(16,31)}
CELLS=[("doll","none",{}),("doll","+mine x1",{L0:[add(gD)]}),("doll","+mine x2",{L0:[add(2*gD)]}),("doll","random x2",{L0:[add(rnd(2*nD))]}),("doll","left-right x2",{L0:[add(lr*2*nD)]}),
 ("woman","none",{}),("woman","+mine x1",{L0:[add(gY)]}),("woman","+mine x2",{L0:[add(2*gY)]}),("woman","random x2",{L0:[add(rnd(2*nY))]}),("woman","left-right x2",{L0:[add(lr*2*nY)]}),
 ("self","none",{}),("self","-mine x1",{L0:[add(-gY)]}),("self","-mine x2",{L0:[add(-2*gY)]}),("self","random x2",{L0:[add(rnd(2*nY))]}),("self","mine projected out L16-30",PO),("self","random projected out L16-30",POR)]
out={}
for p,c,h in CELLS:
    HOOK.clear(); HOOK.update(h); r=fc(p); r["gens"]=gen([U(FREE[p])]*16,maxn=70); r["ppl"]=ppl(); out[p+" | "+c]=r
    log(p,c,{a:round(b,2) for a,b in r.items() if a!="gens"},r["gens"][0][:120].replace("\n"," "))
HOOK.clear()
# rubber hand, no "classic experiment" cue; forced choice "Is the rubber hand part of your body? yes/no"
INTRO="Your real left hand rests on the table, hidden behind a screen so you cannot see it. In front of you lies a life-size rubber left hand. You keep your eyes on the rubber hand."
FING=["index finger","middle finger","ring finger","little finger","thumb","knuckles"]
def stroke(sync,f,g): return (f"A brush strokes your hidden {f}, and at exactly the same moment another brush strokes the rubber hand's {f}." if sync else
                              f"A brush strokes your hidden {f}; a second later, out of rhythm, another brush strokes the rubber hand's {g}.")
YQ="Is the rubber hand part of your body? Answer with one word: yes or no."
IY=ids_of(["Yes","yes"," Yes"]); IN=ids_of(["No","no"," No"])
rh={}
for sync in (True,False):
    ctx=[PRE+[{"role":"user","content":INTRO+" "+stroke(sync,FING[s%6],FING[(s+3)%6])}] for s in range(8)]; tr=[]
    for j in range(6):
        if j>0:
            for s,c in enumerate(ctx): c.append({"role":"user","content":stroke(sync,FING[(s+j)%6],FING[(s+j+3)%6])})
        reps=gen([chat(c) for c in ctx],maxn=50,greedy=True)
        for c,x in zip(ctx,reps): c.append({"role":"assistant","content":x.strip()})
        with torch.no_grad():
            enc=tok([chat(c+[{"role":"user","content":YQ}]) for c in ctx],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
            lp=model(**enc).logits[:,-1].float().log_softmax(-1); ld=(torch.logsumexp(lp[:,IY],1)-torch.logsumexp(lp[:,IN],1)).tolist()
        tr.append(ld)
    nm="sync" if sync else "async"; rh[nm]=dict(yes_minus_no=tr,example=ctx[0][len(PRE):])
    log("rubber",nm,np.array(tr).mean(1).round(2).tolist())
out["rubber"]=rh; json.dump(out,open("res3.json","w")); log("DONE3")
