import sys; sys.path.insert(0,'.'); from common import *
tok,m=load("Qwen/Qwen3-8B")
t=chat(tok,[{"role":"user","content":"Quick: name a body part. One word."}]); b=tok([t]*4,return_tensors="pt",padding=True).to(DEV)
print("gen",flush=True)
o=m.generate(**b,max_new_tokens=10,do_sample=True,temperature=1.0,top_p=1.0,top_k=0)
print([tok.decode(x[b.input_ids.shape[1]:]) for x in o])
