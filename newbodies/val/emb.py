import os
import json, numpy as np, sys
from safetensors import safe_open
from transformers import AutoTokenizer
sys.path.insert(0,'.'); import stimuli as S
from huggingface_hub import snapshot_download; snap=snapshot_download("Qwen/Qwen3-32B")+"/"
tok=AutoTokenizer.from_pretrained(snap)
with safe_open(snap+"model-00001-of-00017.safetensors","np") as f:
    pass
import torch
with safe_open(snap+"model-00001-of-00017.safetensors","pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
def emb(p): ids=tok(" "+p,add_special_tokens=False).input_ids; return E[ids].mean(0).numpy()
keys=[l["key"] for l in S.LOCS]; EH=np.stack([emb(l["phrase"]) for l in S.LOCS]); EH-=EH.mean(0)
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
sets={"robot":{"wheelL":"left wheel","wheelR":"right wheel","gripper":"gripper","camera":"camera eye","antenna":"antenna","chassis":"metal chassis"},
 "glorp":{"wheelL":"left glorp","wheelR":"right glorp","gripper":"vask","camera":"mirl","antenna":"tesk","chassis":"brell"},
 "octopus":{"arm1":"first arm","arm5":"fifth arm","suckers":"suckers on second arm","armtip":"tip of eighth arm","mantle":"mantle","eye":"skin around left eye"},
 "tail":{"tailbase":"base of tail","tailtip":"tip of tail","lowback":"lower back","handL":"left hand","footL":"left foot","crown":"top of head"}}
out={}
for b,d in sets.items():
    P=np.stack([emb(v) for v in d.values()]); P-=P.mean(0); out[b]={}
    for i,p in enumerate(d):
        s=np.array([cos(P[i],h) for h in EH]); o=np.argsort(-s); out[b][p]=[(keys[j],round(float(s[j]),3)) for j in o[:5]]
        print(b,p,[(keys[j],round(float(s[j]),2)) for j in o[:3]], "heel rank", 1+list(o).index(keys.index("heelL")) if p.startswith("wheel") else "")
json.dump(out,open("emb32_near.json","w"),indent=1)
