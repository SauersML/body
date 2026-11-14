"""python run.py MODEL REV TAG [states_only]: last-token states for 9 bodies + control (every 4th layer), then QA with subspace ablations."""
import sys, json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bodies import *
M, REV, TAG = sys.argv[1:4]; SO = len(sys.argv) > 4
tok = AutoTokenizer.from_pretrained(M, revision=REV); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M, revision=REV, torch_dtype=torch.bfloat16, device_map="cuda").eval()
LAY = model.model.layers; NL = len(LAY); LS = list(range(3, NL, 4))
if NL - 1 not in LS: LS.append(NL - 1)
t0 = time.time()

@torch.no_grad()
def last_states(texts, bs=32):
    out = {l: [] for l in LS}
    for k in range(0, len(texts), bs):
        e = tok(texts[k:k + bs], return_tensors="pt", padding=True).to("cuda")
        hs = model(**e, output_hidden_states=True).hidden_states
        for l in LS: out[l].append(hs[l + 1][:, -1].float().cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}

ALL = BODIES + ["ctrl"]; meta, texts = [], []
for b in ALL:
    for i, j, s in sentences(b): meta.append((b, i, j)); texts.append(s)
S = last_states(texts)
np.savez_compressed(f"out/{TAG}_states.npz", **{f"L{l}": S[l].astype(np.float16) for l in LS}, meta=np.array(meta, dtype=object), layers=np.array(LS))
print("states", time.time() - t0, flush=True)
if SO: sys.exit()

# subspaces: top-k PCs of the 16 centred part means of each source body, per layer
bidx = {b: np.array([m[0] == b for m in meta]) for b in ALL}
pidx = np.array([m[1] for m in meta])
def means(b, l): X = S[l][bidx[b]]; P = np.stack([X[pidx[bidx[b]] == i].mean(0) for i in range(16)]); return P - P.mean(0)
SUB = {}
for b in ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk", "ctrl"]:
    for l in LS: SUB[(b, l)] = np.linalg.svd(means(b, l), full_matrices=False)[2]  # 16 x d (15 meaningful)
rng = np.random.default_rng(0); d = S[LS[0]].shape[1]
for l in LS: SUB[("rand", l)] = np.linalg.qr(rng.standard_normal((d, 16)))[0].T

# QA
qs = []
for b in BODIES:
    for q, a, w, kind in qa(b, np.random.default_rng(1)):
        for order in (0, 1):
            A, Bo = (a, w) if order == 0 else (w, a)
            msg = f"{q}\n(A) {A}\n(B) {Bo}\nReply with just A or B."
            p = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            qs.append((b, kind, order, p))
ida, idb = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
HOOK = {"U": None}
def mk(l):
    def f(mod, inp, out):
        if HOOK["U"] is None: return
        h = out[0] if isinstance(out, tuple) else out
        U = HOOK["U"][l]; h = h - (h @ U.T) @ U
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return f
for l in LS: LAY[l].register_forward_hook(mk(l))

@torch.no_grad()
def margins(bs=48):
    r = []
    for k in range(0, len(qs), bs):
        e = tok([q[3] for q in qs[k:k + bs]], return_tensors="pt", padding=True).to("cuda")
        lg = model(**e).logits[:, -1].float()
        r.append((lg[:, ida] - lg[:, idb]).cpu())
    m = torch.cat(r).numpy(); sgn = np.array([1 if q[2] == 0 else -1 for q in qs]); return m * sgn  # >0 = correct
res = {"layers": LS, "qs": [q[:3] for q in qs], "cond": {}}
res["cond"]["none"] = margins().tolist(); print("base", time.time() - t0, flush=True)
for k in (4, 8):
    for src in ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk", "ctrl", "rand"]:
        HOOK["U"] = {l: torch.tensor(SUB[(src, l)][:k], dtype=torch.bfloat16, device="cuda") for l in LS}
        res["cond"][f"{src}_k{k}"] = margins().tolist(); print(src, k, time.time() - t0, flush=True)
HOOK["U"] = None
json.dump(res, open(f"out/{TAG}_qa.json", "w"))
print("DONE", time.time() - t0, flush=True)
