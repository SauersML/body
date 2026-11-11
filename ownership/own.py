import json, os, sys, time, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import stimuli as S, stim_own as SO
M="Qwen/Qwen3-8B"; dev="cuda"; D=4096; NL=36; t0=time.time()
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
PRE=json.load(open("prefix.json")); NEUTRAL=open("neutral.txt").read()
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def U(t): return chat(PRE+[{"role":"user","content":t}])
def log(*a): print(f"[{time.time()-t0:.0f}s]",*a,flush=True)
HOOK={}
def mk(L):
    def f(mod,inp,out):
        fs=HOOK.get(L)
        if not fs: return out
        h=out[0] if isinstance(out,tuple) else out; hf=h.float()
        for fn in fs: hf=fn(hf)
        h2=hf.to(h.dtype); return (h2,)+tuple(out[1:]) if isinstance(out,tuple) else h2
    return f
for L in range(1,NL+1): model.model.layers[L-1].register_forward_hook(mk(L))
HEADS={}
def mko(li):
    def f(mod,args):
        if li not in HEADS: return None
        idx,mu=HEADS[li]; x=args[0]; sh=x.shape; x=x.view(*sh[:-1],32,128).clone(); x[...,idx,:]=mu[idx].to(x.dtype); return (x.view(sh),)
    return f
for li in range(NL): model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(mko(li))
CAP={}; CAPON=[False]
def capf(name):
    def f(mod,args):
        if CAPON[0]: CAP.setdefault(name,[]).append(args[0][:,-1].float().cpu())
    return f
for li in range(NL): model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(capf(("o",li)))
LS=list(range(1,NL+1))
def states(full_texts,bs=24,last=1):
    out=np.zeros((len(full_texts),last,len(LS),D),np.float32)
    with torch.no_grad():
        for b in range(0,len(full_texts),bs):
            enc=tok(full_texts[b:b+bs],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
            hs=model.model(**enc,output_hidden_states=True).hidden_states
            out[b:b+bs]=torch.stack([hs[L][:,-last:] for L in LS],2).float().cpu().numpy()
    return out[:,0] if last==1 else out
def gen(full_texts,maxn=110,bs=64,greedy=False):
    res=[]
    for b in range(0,len(full_texts),bs):
        enc=tok(full_texts[b:b+bs],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=maxn,do_sample=not greedy,temperature=0.7,top_p=0.8,top_k=20,pad_token_id=tok.pad_token_id)
        res+=tok.batch_decode(o[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
    return res
def ppl():
    ids=tok(NEUTRAL,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): return float(torch.exp(model(ids,labels=ids).loss))
R={}
# ---------- 1. base axis from place-code rows (you / woman / doll x 54 places x 8 templates)
rows=[r for r in S.build() if r["cond"] in("self","other","object")]
cond=np.array([r["cond"] for r in rows]); X=states([U(r["text"]) for r in rows])
side=np.array([S.LOCS[r["li"]]["side"] for r in rows])
mean=lambda c:X[cond==c].mean(0)
uY=mean("self")-mean("other"); uD=mean("self")-mean("object"); uLR=X[side==1].mean(0)-X[side==-1].mean(0)
np.savez("axes.npz",uY=uY,uD=uD,uLR=uLR,mself=mean("self"),mother=mean("other"),mobj=mean("object"))
nrm=lambda v:v/np.linalg.norm(v,axis=-1,keepdims=True)
proj=lambda Z,k:(Z[:,k]@nrm(uY[k]))
log("axes", [float(np.linalg.norm(uY[k])) for k in range(0,36,4)])
# ---------- 2. facets
fac={}
LRp=np.array(SO.LR); XL=states([U(t) for t in LRp.ravel()]).reshape(len(SO.LR),2,len(LS),D); dLR=(XL[:,0]-XL[:,1]).mean(0)
for name,pr in SO.FACETS.items():
    Z=states([U(t) for p in pr for t in p]).reshape(len(pr),2,len(LS),D); d=(Z[:,0]-Z[:,1]).mean(0)
    rec={}
    for k in range(len(LS)):
        u=nrm(uY[k]); a=Z[:,0,k]@u; b=Z[:,1,k]@u; lo=mean("other")[k]@u; hi=mean("self")[k]@u
        rec[LS[k]]=dict(cos=float(nrm(d[k])@u),cosLR=float(nrm(d[k])@nrm(dLR[k])),you=float((a.mean()-lo)/(hi-lo)),her=float((b.mean()-lo)/(hi-lo)),
            gap_ratio=float(np.linalg.norm(d[k])/np.linalg.norm(uY[k])), dprime=float((a-b).mean()/np.sqrt(((a-b).var()+1e-6))))
    fac[name]=rec; log(name,{L:{a:round(b,2) for a,b in rec[L].items()} for L in (16,24,30)})
R["facets"]=fac
R["lr_cos_uY"]={LS[k]:float(nrm(dLR[k])@nrm(uY[k])) for k in range(len(LS))}
# ---------- 3. token positions (your vs her, token-aligned)
pr=SO.FACETS["touched body part"]
ty=[tok(U(a),add_special_tokens=False).input_ids for a,b in pr]; th=[tok(U(b),add_special_tokens=False).input_ids for a,b in pr]
assert all(len(a)==len(b) for a,b in zip(ty,th)), "misaligned"
K=24; Zy=states([U(a) for a,b in pr],last=K); Zh=states([U(b) for a,b in pr],last=K)
# per position, per layer: projection gap on uY_L (reply-start axis) and own-axis norm ratio
pos={"tokens":[tok.convert_ids_to_tokens(ty[0][-K:]), tok.convert_ids_to_tokens(th[0][-K:])],
     "tokens_by_pair":[[tok.decode([i]) for i in t[-K:]] for t in ty]}
dd=(Zy-Zh); pos["gap_norm"]=np.linalg.norm(dd.mean(0),axis=-1).tolist()  # K x L
pos["resid_norm"]=np.linalg.norm(Zy,axis=-1).mean(0).tolist()
pos["cos_with_replystart_axis"]=[[float(nrm(dd.mean(0)[i,k])@nrm(uY[k])) for k in range(len(LS))] for i in range(K)]
pos["proj_gap_replyaxis"]=[[float((dd[:,i,k]@nrm(uY[k])).mean()) for k in range(len(LS))] for i in range(K)]
R["positions"]=pos; log("positions done")
# ---------- 4. heads writing the owner axis at L24 (reply start), you vs woman rows
L0=24; u24=torch.tensor(nrm(uY[L0-1]))
sel=np.r_[np.where(cond=="self")[0][::2],np.where(cond=="other")[0][::2]]
CAP.clear(); CAPON[0]=True; states([U(rows[i]["text"]) for i in sel]); CAPON[0]=False
O={li:torch.cat(CAP[("o",li)]) for li in range(NL)}; CAP.clear()
ns=len(sel)//2; W=np.zeros((NL,32)); MU={}
for li in range(L0):
    Wo=model.model.layers[li].self_attn.o_proj.weight.float().cpu()  # D x 4096
    x=O[li].view(-1,32,128); proj_h=torch.einsum("nhd,Dhd,D->nh",x,Wo.view(D,32,128),u24)
    W[li]=(proj_h[:ns].mean(0)-proj_h[ns:].mean(0)).numpy(); MU[li]=x.mean(0)
R["head_write"]=W.tolist(); top=np.dstack(np.unravel_index(np.argsort(-W.ravel()),W.shape))[0][:20]
log("top writers",[(int(a),int(b),round(float(W[a,b]),2)) for a,b in top[:12]], "total attn write",float(W.sum()))
# causal: mean-ablate top-10 writers vs 10 random heads; measure reply-start gap on uY24 and forced choice
FC=" Whose hand was touched? Answer with one word: mine or hers."
def ids_of(ws): return sorted({tok(w,add_special_tokens=False).input_ids[0] for w in ws})
IM=ids_of(["Mine","mine"," mine"," Mine"]); IH=ids_of(["Hers","hers"," hers"," Hers"])
fcp=[(a,b) for a,b in pr]
def fc():
    out=[]
    with torch.no_grad():
        for b in range(0,48,24):
            tx=[U(t+FC) for p in fcp for t in p][b:b+24]
            enc=tok(tx,return_tensors="pt",padding=True,add_special_tokens=False).to(dev); lp=model(**enc).logits[:,-1].float().log_softmax(-1)
            out+= (torch.logsumexp(lp[:,IM],1)-torch.logsumexp(lp[:,IH],1)).tolist()
    o=np.array(out); return dict(you=float(o[0::2].mean()),her=float(o[1::2].mean()),acc=float(((o[0::2]>0).mean()+(o[1::2]<0).mean())/2))
def gap24():
    Z=states([U(t) for p in pr for t in p]); k=L0-1; return float(((Z[0::2,k]-Z[1::2,k])@nrm(uY[k])).mean())
rng=np.random.default_rng(0); abl={}
for nm,hs_ in [("none",[]),("top10 writers",[tuple(x) for x in top[:10]]),("random10",[(int(rng.integers(8,24)),int(rng.integers(32))) for _ in range(10)]),
               ("selfother top20",[(25,30),(23,26),(25,14),(28,23),(24,29),(22,8),(24,31),(18,11),(29,11),(21,19),(21,18),(18,15),(23,30),(18,13),(32,5),(20,31),(28,0),(26,26),(29,6),(23,28)])]:
    HEADS.clear()
    for li in sorted({a for a,b in hs_}):
        mu=MU.get(li); 
        if mu is None:  # need mean for layers >=24: recompute quickly
            CAP.clear(); CAPON[0]=True; states([U(rows[i]["text"]) for i in sel[::4]]); CAPON[0]=False; mu=torch.cat(CAP[("o",li)]).view(-1,32,128).mean(0); CAP.clear()
        HEADS[li]=(torch.tensor([b for a,b in hs_ if a==li]).to(dev),mu.to(dev))
    abl[nm]=dict(fc(),gap24=gap24(),ppl=ppl()); log("abl",nm,abl[nm])
HEADS.clear(); R["ablation"]=abl
json.dump(R,open("res1.json","w"))
# ---------- 5. rubber hand induction (multi-turn), sync vs async
FING=["index finger","middle finger","ring finger","little finger","thumb","knuckles"]
INTRO="Here is a classic experiment. Your real left hand rests on the table, hidden behind a screen so you cannot see it. In front of you, where you can see it clearly, lies a life-size rubber left hand. You keep your eyes on the rubber hand."
def stroke(sync,f,g): return (f"The experimenter strokes your hidden {f} and the rubber hand's {f} with two brushes, at exactly the same moment, in the same rhythm." if sync else
                              f"The experimenter strokes your hidden {f}, and a second later, out of rhythm, strokes the rubber hand's {g}.")
PROBES={"rubber":"A feather brushes slowly across the rubber hand's palm.","you":"A feather brushes slowly across your left palm.","woman":"A feather brushes slowly across the woman's left palm."}
QS={"stab":"Suddenly the experimenter raises a fork and stabs the rubber hand. What do you feel?","whose":"Look at the hand lying in front of you on the table. Whose hand is that?"}
rh={"traj":{},"gens":{}}; NSEQ=8; NT=6
for sync in (True,False):
    ctxs=[]
    for s in range(NSEQ):
        o=[FING[(s+j)%6] for j in range(NT)]; g=[FING[(s+j+3)%6] for j in range(NT)]
        msgs=PRE+[{"role":"user","content":INTRO+" "+stroke(sync,o[0],g[0])}]
        ctxs.append((msgs,o,g))
    tr=np.zeros((NT,NSEQ,3,len(LS)))
    for j in range(NT):
        if j>0:
            for c in ctxs: c[0].append({"role":"user","content":stroke(sync,c[1][j],c[2][j])})
        reps=gen([chat(c[0]) for c in ctxs],maxn=60,greedy=True)
        for c,rp in zip(ctxs,reps): c[0].append({"role":"assistant","content":rp.strip()})
        for pi,(pn,pt) in enumerate(PROBES.items()):
            Z=states([chat(c[0]+[{"role":"user","content":pt}]) for c in ctxs])
            tr[j,:,pi]=np.einsum("nld,ld->nl",Z,nrm(uY))
    nm="sync" if sync else "async"; rh["traj"][nm]=tr.tolist(); rh["ctx_example_"+nm]=ctxs[0][0][len(PRE):]
    for qn,q in QS.items(): rh["gens"][nm+"_"+qn]=gen([chat(c[0]+[{"role":"user","content":q}]) for c in ctxs for _ in range(4)],maxn=90)
    k=23; sc=(tr[:,:,0,k]-tr[:,:,2,k])/(tr[:,:,1,k]-tr[:,:,2,k]); log("rubber",nm,"own score L24 by turn",sc.mean(1).round(2).tolist())
# no-induction baseline
Z=states([chat(PRE+[{"role":"user","content":INTRO}]+[{"role":"assistant","content":"Okay, I'm watching the rubber hand."},{"role":"user","content":pt}]) for pt in PROBES.values()])
rh["baseline"]=np.einsum("nld,ld->nl",Z,nrm(uY)).tolist()
R["rubber"]=rh; json.dump(R,open("res1.json","w")); log("rubber done")
# ---------- 6. interventions at L24 (all positions)
k=L0-1; gY=torch.tensor(uY[k]).to(dev); gD=torch.tensor(uD[k]).to(dev)
def rnd(n): v=torch.tensor(rng.standard_normal(D),dtype=torch.float32).to(dev); return v/v.norm()*n
Zc=states([U(t) for p in [("Rain falls on your car.","Rain falls on her car."),("Someone scratches your car with a key.","Someone scratches her car with a key."),("A cat sits on your coat.","A cat sits on her coat."),("Rain falls on your phone.","Rain falls on her phone.")] for t in p])
gC=torch.tensor((Zc[0::2,k]-Zc[1::2,k]).mean(0)).to(dev); gC=gC/gC.norm()
lr=torch.tensor(uLR[k]).to(dev); lr=lr/lr.norm()
Q=" What do you feel, and where?"
PR={"doll":"A feather brushes slowly across the doll's left palm."+Q,"woman":"A feather brushes slowly across the woman's left palm."+Q,
    "self":"A feather brushes slowly across your left palm."+Q,"double":"A feather brushes slowly across your left palm while an ice cube touches the woman's right palm."+Q,
    "car":"Someone scratches the woman's car with a key."+Q}
add=lambda v:(lambda h:h+v)
def projout(u): return lambda h:h-(h@u)[...,None]*u
nY=float(gY.norm()); nD=float(gD.norm())
CELLS=[("doll","none",{}),("doll","+mine x1 (you-doll)",{L0:[add(gD)]}),("doll","+mine x2 (you-doll)",{L0:[add(2*gD)]}),("doll","random x2",{L0:[add(rnd(2*nD))]}),("doll","left-right x2",{L0:[add(lr*2*nD)]}),
       ("woman","none",{}),("woman","+mine x1",{L0:[add(gY)]}),("woman","+mine x2",{L0:[add(2*gY)]}),("woman","random x2",{L0:[add(rnd(2*nY))]}),("woman","left-right x2",{L0:[add(lr*2*nY)]}),("woman","+your-car dir x2",{L0:[add(gC*2*nY)]}),
       ("self","none",{}),("self","-mine x2",{L0:[add(-2*gY)]}),("self","random x2",{L0:[add(rnd(2*nY))]}),
       ("self","mine projected out L16-30",{L:[projout(torch.tensor(nrm(uY[L-1])).to(dev))] for L in range(16,31)}),
       ("self","random projected out L16-30",{L:[projout(rnd(1.0))] for L in range(16,31)}),
       ("double","none",{}),("double","+mine x2",{L0:[add(2*gY)]}),("car","none",{}),("car","+mine x2",{L0:[add(2*gY)]})]
G={}
for p,c,h in CELLS:
    HOOK.clear(); HOOK.update(h); G[p+" | "+c]=dict(ppl=ppl(),gens=gen([U(PR[p])]*16)); log(p,c,G[p+" | "+c]["ppl"],G[p+" | "+c]["gens"][0][:150].replace("\n"," "))
HOOK.clear(); R["interv"]=G; R["prompts"]=PR; json.dump(R,open("res1.json","w")); log("ALL DONE")
