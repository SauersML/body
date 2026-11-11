import os
import sys, os, json, torch, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__)); import stim as S
from transformers import AutoTokenizer, AutoModel, AutoConfig
M = "Qwen/Qwen3-8B"; dev = "mps"; NL = 29; KEEP = [2, 6, 10, 14] + list(range(16, 29))
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = NL
model = AutoModel.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
msgs = json.load(open(os.path.expandvars("$BODY_REPO/map/somato/out_8b/prefix.json")))
rows = S.rows()
texts = [tok.apply_chat_template(msgs + [{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
print(repr(texts[0][-120:]))
H = np.zeros((len(KEEP), len(rows), cfg.hidden_size), np.float16); E = np.zeros((len(rows), cfg.hidden_size), np.float32); t0 = time.time()
emb = model.embed_tokens.weight
with torch.no_grad():
    for i, r in enumerate(rows):  # lexical baseline: mean input embedding of the stimulus sentence
        ids = tok(r["text"], return_tensors="pt").input_ids.to(dev); E[i] = emb[ids[0]].float().mean(0).cpu().numpy()
    B = 8
    for i in range(0, len(rows), B):
        enc = tok(texts[i:i+B], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for j, k in enumerate(KEEP): H[j, i:i+B] = hs[k][:, -1].float().cpu().numpy()
        if i % 160 == 0: print(i, round(time.time() - t0), flush=True)
np.save("H.npy", H); np.save("E.npy", E); json.dump(dict(keep=KEEP, rows=rows), open("rows.json", "w"))
print("done", time.time() - t0)
