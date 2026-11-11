import os
import numpy as np, json, glob, torch, stim as S
from safetensors import safe_open
from transformers import AutoTokenizer
from sm import spot_means
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/"))[0]
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]["model.embed_tokens.weight"]
with safe_open(d + idx, "pt") as f: Wt = f.get_tensor("model.embed_tokens.weight").float().numpy()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); rows = S.rows()
E = np.stack([Wt[tok(r["text"], add_special_tokens=False).input_ids].mean(0) for r in rows])
A, W = spot_means(E, rows, len(S.probes())); np.savez("red8b_lex.npz", keep=np.array([0]), **{"lex|0|all|A": A, "lex|0|all|W": W})
