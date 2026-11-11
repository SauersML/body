# local (MPS) Qwen3-8B helpers for own7*
import json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M="Qwen/Qwen3-8B"; dev="cuda" if torch.cuda.is_available() else "mps"; D=4096; NL=36; t0=time.time()
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def log(*a): print(f"[{time.time()-t0:.0f}s]",*a,flush=True)
def ids_of(ws): return sorted({tok(x,add_special_tokens=False).input_ids[0] for x in ws})
IY=ids_of(["Yes","yes"," Yes"," yes"]); IN=ids_of(["No","no"," No"," no"])
def run(texts,bs=16,want_states=False,layers=None):
    """returns yes-no logodds at last token and optionally last-token residuals [n,L,D] (hidden_states index L = after layer L)"""
    lo=[]; st=[]
    with torch.no_grad():
        for b in range(0,len(texts),bs):
            enc=tok(texts[b:b+bs],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
            o=model.model(**enc,output_hidden_states=want_states)
            lp=model.lm_head(o.last_hidden_state[:,-1]).float().log_softmax(-1)
            lo+=(torch.logsumexp(lp[:,IY],1)-torch.logsumexp(lp[:,IN],1)).tolist()
            if want_states:
                ls=layers or range(1,NL+1)
                st.append(torch.stack([o.hidden_states[L][:,-1] for L in ls],1).float().cpu().numpy())
    return (np.array(lo), np.concatenate(st) if want_states else None)
def gen(texts,maxn=60,bs=48,greedy=False):
    res=[]
    for b in range(0,len(texts),bs):
        enc=tok(texts[b:b+bs],return_tensors="pt",padding=True,add_special_tokens=False).to(dev)
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=maxn,do_sample=not greedy,temperature=0.7,top_p=0.8,top_k=20,pad_token_id=tok.pad_token_id)
        res+=tok.batch_decode(o[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
    return res
