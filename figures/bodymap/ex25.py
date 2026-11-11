import os
import sys, os, json, torch, numpy as np, time
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from transformers import AutoTokenizer, AutoModel, AutoConfig
M = "Qwen/Qwen3-8B"; dev = "mps"; LAY = 25
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = LAY
model = AutoModel.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
msgs = json.load(open(os.path.expandvars("$BODY_REPO/map/somato/out_8b/prefix.json")))
rows = [r for r in S.build() if r["cond"] == "self"]
texts = [tok.apply_chat_template(msgs + [{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
H = np.zeros((len(rows), cfg.hidden_size), np.float32); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(rows), 8):
        enc = tok(texts[i:i+8], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[LAY]
        H[i:i+8] = hs[:, -1].float().cpu().numpy()
        if i % 80 == 0: print(i, time.time() - t0, flush=True)
np.save(os.path.expandvars("$BODY_REPO/figures/bodymap/H25_self.npy"), H); json.dump(rows, open(os.path.expandvars("$BODY_REPO/figures/bodymap/rows_self.json"), "w"))
print("done")
