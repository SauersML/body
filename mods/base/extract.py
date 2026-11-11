"""Residual stream at the first reply token for every (context, probe). Context prefix computed once (KV cache), probes batched."""
DEV = 'cuda'
import json, torch, os, sys, time, copy, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import mstim as MS
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
NL = 31; KEEP = [22, 24, 25, 26, 27, 28, 30]
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = NL
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map=DEV).eval()
C = json.load(open(f"{D}/contexts.json")); R = MS.probes(); names = list(C)
only = sys.argv[1:]  # optional subset of context names
if only: names = [n for n in names if n in only]
os.makedirs(f"{D}/H", exist_ok=True); t0 = time.time(); B = 32
with torch.no_grad():
    for ci, n in enumerate(names):
        if os.path.exists(f"{D}/H/{n.replace(':', '__')}.npy"): continue
        msgs = C[n]["msgs"]
        full = [tok.apply_chat_template(msgs + [{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in R]
        cut = full[0].rfind("<|im_start|>user"); pre = full[0][:cut]
        assert all(f.startswith(pre) for f in full)
        pid = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        out = model.model(pid, use_cache=True); cache0 = out.past_key_values; P = pid.shape[1]
        H = np.zeros((len(KEEP), len(R), cfg.hidden_size), np.float16)
        for b in range(0, len(R), B):
            sub = [tok(f[cut:], add_special_tokens=False).input_ids for f in full[b:b+B]]; L = max(map(len, sub)); nb = len(sub)
            ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in sub], device=DEV)
            am = torch.cat([torch.ones(nb, P, dtype=torch.long), torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in sub])], 1).to(DEV)
            cache = copy.deepcopy(cache0); cache.batch_repeat_interleave(nb)
            hs = model.model(ids, attention_mask=am, past_key_values=cache, use_cache=True, output_hidden_states=True).hidden_states
            last = torch.tensor([len(s) - 1 for s in sub], device=DEV)
            for j, k in enumerate(KEEP): H[j, b:b+nb] = hs[k][torch.arange(nb), last].float().cpu().numpy()
        np.save(f"{D}/H/{n.replace(':', '__')}.npy", H)
        print(ci, n, P, round(time.time() - t0), flush=True)
json.dump(dict(keep=KEEP, probes=R), open(f"{D}/probes.json", "w"))
print("done", round(time.time() - t0))
