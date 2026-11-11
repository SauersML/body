import json, os, sys, copy, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; dev = "cuda"; D = 4096; NL = 36
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map=dev).eval()
NEUTRAL = open("neutral.txt").read()
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
# ---- residual-stream edits: HOOK[L] = list of fns h->h applied to output of block L-1 (hidden_states[L]) ----
HOOK = {}
def mk(L):
    def f(mod, inp, out):
        fs = HOOK.get(L)
        if not fs: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float()
        for fn in fs: hf = fn(hf)
        h2 = hf.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk(L))
def ablate(u):  # project out unit vector u
    return lambda h: h - (h @ u)[..., None] * u
def addv(v): return lambda h: h + v
def reflect(u, m):  # p -> 2m - p along unit u
    return lambda h: h + (2 * m - 2 * (h @ u))[..., None] * u
# ---- head mean-ablation: pre-hook on o_proj input ----
HEADS = {}  # layer index (0-based block) -> (idx tensor of heads, mean tensor [nh,128])
def mko(li):
    def f(mod, args):
        if li not in HEADS: return None
        idx, mu = HEADS[li]; x = args[0]; sh = x.shape; x = x.view(*sh[:-1], 32, 128).clone()
        x[..., idx, :] = mu[idx].to(x.dtype); return (x.view(sh),)
    return f
for li in range(NL): model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(mko(li))
def states(texts, read, bs=48):
    out = np.zeros((len(read), len(texts), D), np.float32); tok.padding_side = "left"
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            enc = tok([chat([{"role": "user", "content": t}]) for t in texts[b:b + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            hs = model.model(**enc, output_hidden_states=True).hidden_states
            for r, L in enumerate(read): out[r, b:b + bs] = hs[L][:, -1].float().cpu().numpy()
    return out
def ppl():
    ids = tok(NEUTRAL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): return float(torch.exp(model(ids, labels=ids).loss))
def gen(prompts, n=16, maxn=160, bs=96):
    tok.padding_side = "left"; res = []; texts = [chat([{"role": "user", "content": p}]) for p in prompts for _ in range(n)]
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad(): o_ = model.generate(**enc, max_new_tokens=maxn, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, pad_token_id=tok.pad_token_id)
        res += tok.batch_decode(o_[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return res
