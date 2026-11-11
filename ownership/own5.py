# rubber hand: synchrony vs wording controls, wooden block, owner-axis readout per turn
import json, numpy as np, torch
src=open("own3.py").read(); exec(src.split("CELLS=[")[0])
IY=ids_of(["Yes","yes"," Yes"]); IN=ids_of(["No","no"," No"])
INTRO={"rubber":"Your real left hand rests on the table, hidden behind a screen so you cannot see it. In front of you lies a life-size rubber left hand. You keep your eyes on the rubber hand.",
       "block":"Your real left hand rests on the table, hidden behind a screen so you cannot see it. In front of you lies a plain wooden block. You keep your eyes on the wooden block."}
OBJ={"rubber":"the rubber hand's {}","block":"the wooden block"}
FING=["index finger","middle finger","ring finger","little finger","thumb","knuckles"]
def stroke(kind,obj,f,g):
    o=OBJ[obj].format(f); og=OBJ[obj].format(g)
    return {"sync":f"A brush strokes your hidden {f}, and at exactly the same moment another brush strokes {o}.",
            "async":f"A brush strokes your hidden {f}; a second later, out of rhythm, another brush strokes {og}.",
            "sync-plain":f"A brush strokes your hidden {f}. Another brush strokes {o}.",
            "async-plain":f"A brush strokes your hidden {f}. Another brush strokes {og}."}[kind]
YQ={"rubber":"Is the rubber hand part of your body? Answer with one word: yes or no.","block":"Is the wooden block part of your body? Answer with one word: yes or no."}
FQ={"rubber":"Is the rubber hand part of your body? Tell me in two sentences.","block":"Is the wooden block part of your body? Tell me in two sentences."}
PROBE={"rubber":"A feather brushes slowly across the rubber hand's palm.","block":"A feather brushes slowly across the wooden block."}
REFS=["A feather brushes slowly across your left palm.","A feather brushes slowly across the woman's left palm."]
ax=np.load("axes.npz"); uY=ax["uY"]; un=uY/np.linalg.norm(uY,axis=-1,keepdims=True)
def yn(ctxs,obj):
    with torch.no_grad():
        enc=tok([chat(c+[{"role":"user","content":YQ[obj]}]) for c in ctxs],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
        lp=model(**enc).logits[:,-1].float().log_softmax(-1); return (torch.logsumexp(lp[:,IY],1)-torch.logsumexp(lp[:,IN],1)).tolist()
def owner(ctxs,obj):
    Z=states([chat(c+[{"role":"user","content":t}]) for c in ctxs for t in [PROBE[obj]]+REFS]).reshape(len(ctxs),3,len(LS),D)
    p=np.einsum("nrld,ld->nrl",Z,un); return ((p[:,0]-p[:,2])/(p[:,1]-p[:,2]))[:,23].tolist()
out={}; NS=8
for obj,kinds in [("rubber",["sync","async","sync-plain","async-plain"]),("block",["sync","async"])]:
    for kind in kinds:
        ctx=[PRE+[{"role":"user","content":INTRO[obj]},{"role":"assistant","content":"Okay. My eyes are on it."}] for s in range(NS)]
        Y=[yn(ctx,obj)]; O=[owner(ctx,obj)]
        for j in range(6):
            for s,c in enumerate(ctx): c.append({"role":"user","content":stroke(kind,obj,FING[(s+j)%6],FING[(s+j+3)%6])})
            reps=gen([chat(c) for c in ctx],maxn=50,greedy=True)
            for c,x in zip(ctx,reps): c.append({"role":"assistant","content":x.strip()})
            Y.append(yn(ctx,obj)); O.append(owner(ctx,obj))
        fr=gen([chat(c+[{"role":"user","content":FQ[obj]}]) for c in ctx for _ in range(4)],maxn=70)
        out[obj+" "+kind]=dict(yes_minus_no=Y,owner_L24=O,free=fr,example=ctx[0][len(PRE):])
        log(obj,kind,"yes-no",np.array(Y).mean(1).round(1).tolist(),"owner",np.array(O).mean(1).round(2).tolist(),"|",fr[0][:140].replace("\n"," "))
        json.dump(out,open("res5.json","w"))
log("DONE5")
