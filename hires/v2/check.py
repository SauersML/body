import json, torch, numpy as np, stim as S
from transformers import AutoTokenizer, AutoModel, AutoConfig
M="Qwen/Qwen3-8B"; tok=AutoTokenizer.from_pretrained(M); cfg=AutoConfig.from_pretrained(M); cfg.num_hidden_layers=28
model=AutoModel.from_pretrained(M,config=cfg,dtype=torch.bfloat16,device_map="mps").eval()
m=json.load(open("prefix_long.json")); R=np.load("/tmp/qt/reply.npy",mmap_mode="r"); rows=S.rows()
for i in [0,333]:
    t=tok.apply_chat_template(m+[{"role":"user","content":rows[i]["text"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    with torch.no_grad(): h=model(**tok(t,return_tensors="pt").to("mps"),output_hidden_states=True).hidden_states[27][0,-1].float().cpu().numpy()
    r=R[1,i].astype(np.float32); print(i, np.corrcoef(h,r)[0,1], np.linalg.norm(h-r)/np.linalg.norm(h))
