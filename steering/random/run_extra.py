import json, torch, time, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
M="Qwen/Qwen3-8B"; tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map="mps").eval()
msgs=json.load(open("../hires2/prefix_long_person.json"))
Q="Where in your body do you feel something right now?"
msgs=msgs+[{"role":"user","content":Q}]
ids=tok.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=True)["input_ids"]
# steer start: tokens of last user turn onward
pre=tok.apply_chat_template(msgs[:-1],add_generation_prompt=False,enable_thinking=False,return_tensors="pt",return_dict=True)["input_ids"]
S0=pre.shape[1]; L=int(sys.argv[1]) if len(sys.argv)>1 else 22
print("len",ids.shape[1],"steer from",S0,"layer",L,flush=True)
layer=model.model.layers[L]
# typical residual norm at L
cap={}
h=layer.register_forward_hook(lambda m,i,o: cap.__setitem__("h",(o[0] if isinstance(o,tuple) else o).detach()))
with torch.no_grad(): model(ids.to("mps"))
h.remove(); nrm=cap["h"][0,5:].float().norm(dim=-1).median().item(); print("norm",nrm,flush=True)
st={"vec":None,"start":S0}
def hook(m,i,o):
    x=o[0] if isinstance(o,tuple) else o
    if st["vec"] is None: return o
    T=x.shape[1]
    if T>1:  # prefill: add from start (left padding none since same prompt)
        x[:,st["start"]:]+=st["vec"][:,None,:].to(x.dtype)
    else: x[:]+=st["vec"][:,None,:].to(x.dtype)
    return o
layer.register_forward_hook(hook)
D=model.config.hidden_size; g=torch.Generator().manual_seed(0)
NDIR=240; STR=[0.5,1.0,2.0]; NS=2; B=24
dirs=torch.randn(NDIR,D,generator=g); dirs/=dirs.norm(dim=1,keepdim=True)
import os
prev=json.load(open(f"replies_L{L}.json"))["rows"] if os.path.exists(f"replies_L{L}.json") else []
out=prev; done={x["s"] for x in prev}
def gen(vecs,n):
    st["vec"]=vecs
    inp=ids.repeat(n,1).to("mps")
    with torch.no_grad():
        o=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),max_new_tokens=48,do_sample=True,temperature=1.0,top_p=0.95,pad_token_id=tok.pad_token_id)
    return [tok.decode(r[ids.shape[1]:],skip_special_tokens=True) for r in o]
t=time.time()
for b in range(0 if 0 in done else 5):  # baseline 120
    for r in gen(None,B): out.append({"dir":-1,"s":0,"r":r})
print("base done",time.time()-t,flush=True)
for s in [1.0]:
  for k in [1]:
    for b0 in range(0,NDIR,B):
        v=(dirs[b0:b0+B]*s*nrm).to("mps")
        for j,r in enumerate(gen(v,B)): out.append({"dir":b0+j,"s":s,"r":r})
        print(s,k,b0,round(time.time()-t),flush=True)
    json.dump({"L":L,"norm":nrm,"Q":Q,"rows":out},open(f"replies_L{L}.json","w"))
print("DONE")
