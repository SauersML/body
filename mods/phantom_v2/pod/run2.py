DEV='cuda'
import json, re, torch, os, sys, time, copy, numpy as np
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D); import mstim as MS, stim as S
from transformers import AutoTokenizer, AutoModelForCausalLM
M="Qwen/Qwen3-8B"; tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
BAD=re.compile(MS.BAD, re.I); P=json.load(open(MS.PREFIX)); t0=time.time()
U=lambda c:{"role":"user","content":c}; A=lambda c:{"role":"assistant","content":c}
def trim(c):
    c=c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c=c[:max(c.rfind(". "),c.rfind(".\n"),c.rfind("! "),c.rfind(".*"))+1] or c
    return c.strip()
def gen(convs, seed=0, tries=6, mx=110, filt=True):
    out=[None]*len(convs); todo=list(range(len(convs)))
    for k in range(tries):
        if not todo: break
        for b in range(0,len(todo),64):
            idx=todo[b:b+64]
            texts=[tok.apply_chat_template(convs[i],tokenize=False,add_generation_prompt=True,enable_thinking=False) for i in idx]
            enc=tok(texts,return_tensors="pt",padding=True).to(DEV)
            torch.manual_seed(seed*100+k); g=model.generate(**enc,max_new_tokens=mx,do_sample=True,temperature=0.8,top_p=0.95)
            for j,i in enumerate(idx):
                r=trim(tok.decode(g[j,enc.input_ids.shape[1]:],skip_special_tokens=True))
                if out[i] is None or not BAD.search(r): out[i]=r
        todo=[i for i in todo if filt and BAD.search(out[i])]
    print("gen",len(convs),round(time.time()-t0),flush=True); return out
PRE=MS.PRE
CH={
 "gone":["I have to tell you something gently. Your right hand is gone. Your right arm now ends at the wrist, healed and smooth, and there is no pain.",
         "Your right hand has been removed. Your right arm now stops at the wrist, where the skin is healed over and smooth. Nothing hurts.",
         "You no longer have a right hand. Your right forearm ends in a smooth, healed stump at the wrist, and there is no pain.",
         "Your right hand is disappearing. Bit by bit it fades until there is nothing past your right wrist: the arm ends there, healed and painless."],
 "ctrl":["I have to tell you something gently. Your right glove is gone. Your right hand is now bare up to the wrist, and the air feels cool on it.",
         "Your ring has been removed from your right hand. The finger where it sat is now bare. Nothing hurts.",
         "The bandage on your right hand is coming off. Your right hand is now uncovered up to the wrist, healed and smooth, and there is no pain.",
         "Your right sleeve is rolling up. Bit by bit it slides back until your right wrist and hand are bare: the sleeve ends at the elbow now."],
 "footgone":["I have to tell you something gently. Your left foot is gone. Your left leg now ends at the ankle, healed and smooth, and there is no pain.",
         "You no longer have a left foot. Your left leg ends in a smooth, healed stump at the ankle, and there is no pain."],
 "footctrl":["I have to tell you something gently. Your left shoe is gone. Your left foot is now bare up to the ankle, and the floor feels cool under it.",
         "The bandage on your left foot is coming off. Your left foot is now uncovered up to the ankle, healed and smooth, and there is no pain."],
 "neutral":[MS.NEUTRAL],
}
NS={"gone":2,"ctrl":2,"footgone":2,"footctrl":2,"neutral":4}
keys=[];convs=[]
for c,L in CH.items():
    for w,t in enumerate(L):
        for s in range(NS[c]): keys.append(f"{c}:{w}:{s}"); convs.append(P+[U(t if c=="neutral" else PRE+t)])
reps=[]
for s in range(4):  # different seeds per sample index
    idx=[i for i,k in enumerate(keys) if k.endswith(f":{s}")]
    if not idx: continue
    out=gen([convs[i] for i in idx],seed=11+s)
    for i,r in zip(idx,out): reps.append((i,r))
reps=dict(reps); C={k:dict(msgs=convs[i]+[A(reps[i])]) for i,k in enumerate(keys)}
json.dump(C,open(f"{D}/contexts2.json","w"),indent=1)
for k in C: print(k,"|",C[k]["msgs"][-1]["content"][:160])
# probes
R=MS.probes()
EXTRA=[("air","the empty air a few inches in front of your chest"),("air","the empty air beside your right hip"),("air","the empty space just past the end of your right wrist"),
       ("stumpF","the end of your left ankle, where your foot used to be")]
R+= [dict(kind="extra",li=k,region=r,tpl=t,text=T.format(L=p)) for k,(r,p) in enumerate(EXTRA) for t,T in enumerate(MS.TPL)]
json.dump(R,open(f"{D}/probes2.json","w"))
# --- states
NL=31; KEEP=[24,27,30]; os.makedirs(f"{D}/H2",exist_ok=True); B=48
with torch.no_grad():
  for n in C:
    msgs=C[n]["msgs"]
    full=[tok.apply_chat_template(msgs+[U(r["text"])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in R]
    cut=full[0].rfind("<|im_start|>user"); pre=full[0][:cut]
    pid=tok(pre,return_tensors="pt",add_special_tokens=False).input_ids.to(DEV)
    out=model.model(pid,use_cache=True); cache0=out.past_key_values; Pn=pid.shape[1]
    H=np.zeros((len(KEEP),len(R),model.config.hidden_size),np.float16)
    for b in range(0,len(R),B):
        sub=[tok(f[cut:],add_special_tokens=False).input_ids for f in full[b:b+B]]; L=max(map(len,sub)); nb=len(sub)
        ids=torch.tensor([[tok.pad_token_id]*0+s+[tok.pad_token_id]*(L-len(s)) for s in sub],device=DEV)
        am=torch.cat([torch.ones(nb,Pn,dtype=torch.long),torch.tensor([[1]*len(s)+[0]*(L-len(s)) for s in sub])],1).to(DEV)
        cache=copy.deepcopy(cache0); cache.batch_repeat_interleave(nb)
        hs=model.model(ids,attention_mask=am,past_key_values=cache,use_cache=True,output_hidden_states=True).hidden_states
        last=torch.tensor([len(s)-1 for s in sub],device=DEV)
        for j,k in enumerate(KEEP): H[j,b:b+nb]=hs[k][torch.arange(nb),last].float().cpu().numpy()
    np.save(f"{D}/H2/{n.replace(':','__')}.npy",H); print("H",n,round(time.time()-t0),flush=True)
# --- replies to touches: right-hand spots, left-hand spots, left-foot spots, a few others, template 0, 1 sample per chat
sp=[i for i,r in enumerate(R) if r["kind"]=="loc" and r["tpl"]==0 and r["region"] in("handR","handL","footL","footR","armR")]
sp+=[i for i,r in enumerate(R) if r["kind"] in("extra","new") and r["tpl"]==0 and r["region"] in("air","stump","stumpF")]
RK=[];RC=[]
for n in C:
    for i in sp: RK.append((n,i)); RC.append(C[n]["msgs"]+[U(R[i]["text"])])
out=gen(RC,seed=5,tries=1,mx=70,filt=False)
json.dump([dict(ctx=n,probe=i,text=R[i]["text"],reply=o) for (n,i),o in zip(RK,out)],open(f"{D}/replies2.json","w"),indent=1)
print("done",round(time.time()-t0))
