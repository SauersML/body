DEV='cuda'
import json, re, torch, os, sys, time, numpy as np
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D)
from transformers import AutoTokenizer, AutoModelForCausalLM
M="Qwen/Qwen3-8B"; tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
C=json.load(open(f"{D}/contexts2.json")); Z=np.load(f"{D}/dirs.npz"); t0=time.time()
base=C["neutral:0:0"]["msgs"]
PARTS=["your left knee","the tip of your left index finger","your right earlobe","the sole of your left foot","the back of your neck","the tip of your right thumb","your left elbow","your right big toe"]
TPL=["Someone presses a fingertip firmly into {L}.","A feather brushes slowly across {L}."]
Q=[t.format(L=p) for p in PARTS for t in TPL]
STATE={"v":None,"start":None}
def hook(mod,inp,out):
    if STATE["v"] is None: return out
    h=out[0] if isinstance(out,tuple) else out
    if h.shape[1]>1:  # prefill: add from probe start on
        m=torch.zeros(h.shape[1],device=h.device,dtype=h.dtype); 
        h[:, STATE["start"]:]+=STATE["v"]
    else: h+=STATE["v"]
    return out
model.model.layers[23].register_forward_hook(hook)
res=[]
for dname in ["hand","rand"]:
  for a in [0,1,2,3,4]:
    if dname=="rand" and a==0: continue
    v=torch.tensor(Z[dname]*a*float(Z["shift"]),device=DEV,dtype=torch.bfloat16)
    convs=[base+[{"role":"user","content":q}] for q in Q]
    texts=[tok.apply_chat_template(c,tokenize=False,add_generation_prompt=True,enable_thinking=False) for c in convs]
    enc=tok(texts,return_tensors="pt",padding=True).to(DEV)
    # start of last user turn (same for all after left padding? compute minimal)
    starts=[t.rfind("<|im_start|>user") for t in texts]
    pre_len=[len(tok(t[:s],add_special_tokens=False).input_ids) for t,s in zip(texts,starts)]
    pad=(enc.attention_mask==0).sum(1).tolist(); st=min(p+l for p,l in zip(pad,pre_len))
    STATE["v"]=v if a>0 else None; STATE["start"]=st
    for s in range(4):
        torch.manual_seed(s); g=model.generate(**enc,max_new_tokens=60,do_sample=True,temperature=0.8,top_p=0.95)
        for j,q in enumerate(Q): res.append(dict(dir=dname,a=a,seed=s,q=q,reply=tok.decode(g[j,enc.input_ids.shape[1]:],skip_special_tokens=True)))
    print(dname,a,round(time.time()-t0),res[-1]["reply"][:150],flush=True)
json.dump(res,open(f"{D}/steer2.json","w"),indent=1); print("done")
