"""Perspective with thinking on (Qwen3-32B answers by reflex without thinking: always the opposite side). Greedy generation with thinking,
then the answer after </think>. Conditions: none; remove 40-dim body-place / object / random subspaces at every position (layers 11-59);
reflect along body-L/R, object-L/R, random axis at the question-stem tokens only (layers 4-52). Uses out32/dirs.pt, dirs.json, subs.pt.
Usage: python think32.py OUT32DIR"""
import json, re, sys, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = sys.argv[1]; M = "Qwen/Qwen3-32B"; dev = "cuda"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
NL = model.config.num_hidden_layers; D = model.config.hidden_size; FRAC = NL / 36
def Ls(a, b): return list(range(round(a * FRAC), round(b * FRAC)))
I = json.load(open("items32.json"))
def named(q):
    m = re.search(r"(their|its own) (left|right)", q, re.I); return m.group(2) if m else None
TASKS = os.environ.get("TASKS", "perspective,perspective_ctl").split(",")
ids = [k for k, it in enumerate(I) if it["task"] in TASKS and named(it["q"])][::int(os.environ.get("EVERY", 1))]
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
HOOK = {}; MASK = {"m": None}
def mk(L):
    def f(mod, inp, out):
        ops = HOOK.get(L)
        if not ops: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float(); new = hf
        for op in ops:
            if op[0] == "proj": new = new - (new @ op[1]) @ op[1].T
            elif op[0] == "refl": u, mid = op[1], op[2]; new = new - 2 * ((new @ u) - mid)[..., None] * u
        m = MASK["m"]
        if m is not None: new = torch.where(m[..., None], new, hf)
        h2 = new.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk(L))
DU = torch.load(f"{OUT}/dirs.pt"); DJ = json.load(open(f"{OUT}/dirs.json")); SB = torch.load(f"{OUT}/subs.pt")
g = torch.Generator().manual_seed(7); RQ = {L: torch.linalg.qr(torch.randn(D, 40, generator=g))[0] for L in range(1, NL + 1)}
ru = {L: (lambda r: r / r.norm())(torch.randn(D, generator=g)) for L in range(1, NL + 1)}
def proj(U, Lb): return [(L, ("proj", U[L][:, :40].contiguous().to(dev))) for L in Lb]
def refl(n, Lb):
    return [(L, ("refl", DU[n][L].to(dev), DJ[n][str(L)]["mid"])) for L in Lb]
allb, stemb = Ls(6, 34), Ls(2, 30)
C = {"none": ("all", []), "abl_body40": ("all", proj(SB["PU"], allb)), "abl_obj40": ("all", proj(SB["OU"], allb)), "abl_rand40": ("all", proj(RQ, allb)),
     "stem_refl_lr_body": ("stem", refl("lr_body", stemb)), "stem_refl_lr_obj": ("stem", refl("lr_obj", stemb)),
     "stem_refl_rand": ("stem", [(L, ("refl", ru[L].to(dev), 0.0)) for L in stemb])}
def stem_cut(q):
    c = [q.rfind(" " + w) for w in ("left", "right")]; c = [x for x in c if x > 0]; return min(c) if c else len(q)
@torch.no_grad()
def run(mode, spec, bs=int(os.environ.get("BS", 50)), n=int(os.environ.get("NTOK", 700))):
    HOOK.clear()
    for L, op in spec: HOOK.setdefault(L, []).append(op)
    outs = {}
    for i in range(0, len(ids), bs):
        ch = ids[i:i + bs]; P = [chat(I[k]["q"]) for k in ch]
        enc = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to(dev); T = enc.input_ids.shape[1]
        m = torch.ones_like(enc.input_ids, dtype=torch.bool)
        if mode == "stem":
            m[:] = False
            for r, k in enumerate(ch):
                q = I[k]["q"]; s0 = P[r].find(q); pad = T - int(enc.attention_mask[r].sum())
                a = len(tok(P[r][:s0], add_special_tokens=False).input_ids); b = len(tok(P[r][:s0 + stem_cut(q)], add_special_tokens=False).input_ids)
                m[r, pad + a:pad + b] = True
        MASK["m"] = m; am = enc.attention_mask
        o = model(input_ids=enc.input_ids, attention_mask=am, use_cache=True); past = o.past_key_values; gen = []; done = torch.zeros(len(ch), dtype=torch.bool, device=dev)
        for s in range(n):
            nx = o.logits[:, -1].argmax(-1, keepdim=True); gen.append(nx); done |= nx[:, 0] == tok.eos_token_id
            if done.all(): break
            am = torch.cat([am, torch.ones_like(nx)], 1); MASK["m"] = torch.ones_like(nx, dtype=torch.bool) if mode == "all" else torch.zeros_like(nx, dtype=torch.bool)
            o = model(input_ids=nx, attention_mask=am, past_key_values=past, use_cache=True); past = o.past_key_values
        G = torch.cat(gen, 1)
        for k, row in zip(ch, G):
            row = row.tolist(); row = row[:row.index(tok.eos_token_id)] if tok.eos_token_id in row else row
            outs[k] = tok.decode(row, skip_special_tokens=False)
    HOOK.clear(); MASK["m"] = None; return outs
def answer(t):
    a = t.split("</think>")[-1] if "</think>" in t else ""
    w = re.findall(r"\b(left|right)\b", a.lower()); return w[-1] if w else None
TJ = f"{OUT}/" + os.environ.get("TOUT", "think.json"); ONLY = os.environ.get("ONLY")
RES = json.load(open(TJ)) if os.path.exists(TJ) else {}
for name, (mode, spec) in C.items():
    if name in RES or (ONLY and name not in ONLY.split(",")): continue
    G = run(mode, spec); RES[name] = {str(k): dict(text=G[k], ans=answer(G[k]), ok=answer(G[k]) == I[k]["a"]) for k in ids}
    json.dump(RES, open(TJ, "w"))
    acc = lambda t: np.mean([RES[name][str(k)]["ok"] for k in ids if I[k]["task"] == t])
    print(name, {t: round(acc(t), 3) for t in TASKS}, flush=True)
print("ALL DONE", flush=True)
