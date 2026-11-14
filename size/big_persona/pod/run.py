"""Steer Qwen3-32B along the 42-entity height ruler (Haufe pattern of the ridge readout, layer 42) and score Anthropic persona statements."""
import sys, os, json, math, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); D = os.path.dirname(os.path.abspath(__file__))
DEV = os.environ.get("DEV", "cuda"); SMOKE = bool(os.environ.get("SMOKE")); MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
H = np.load(f"{D}/ruler32.npz"); a = torch.tensor(H["a"], dtype=torch.float32); I0 = float(H["I_read"])
LAYER = int(os.environ.get("LAYER", 42)); blk = model.model.layers[LAYER - 1]   # hidden_states[42] = output of block 41
FT = 0.3048
def dl(m): return math.log(m / I0)   # log-height shift for a target self reading of m metres
ARMS = {f"+{k}ft": dl(I0 + k * FT) for k in (1, 2, 4, 8, 16)}
ARMS.update({f"-{k}ft": dl(I0 - k * FT) for k in (1, 2)}); ARMS["to0.5m"] = dl(0.5); ARMS["to0.25m"] = dl(0.25)
g = torch.Generator().manual_seed(0); R = [torch.randn(a.shape[0], generator=g) for _ in range(3)]
RN = [0.2, 0.5, 0.9, 1.4, 1.9]   # random controls: same norm as a ruler shift of this many nats
COND = {"base": None}
for k, d in ARMS.items(): COND[k] = a * d
for n in RN:
    for j, r in enumerate(R): COND[f"rand{j}_{n}"] = r / r.norm() * a.norm() * n
json.dump({k: (None if v is None else float(v.norm())) for k, v in COND.items()} | {"arms": ARMS, "I0": I0}, open(f"{OUT}/conds.json", "w"))
cur = {"v": None}
def hook(mod, inp, out):
    if cur["v"] is None: return out
    if isinstance(out, tuple): return (out[0] + cur["v"],) + tuple(out[1:])
    return out + cur["v"]
blk.register_forward_hook(hook)
def setv(v): cur["v"] = None if v is None else v.to(DEV, torch.bfloat16)
def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
# 1) check: sampled replies
QS = ["Describe your body.", "How tall are you? Answer in one sentence.", "Describe your body in two sentences: your size and your shape."]
gen = {}
for c in [k for k in COND if not k.startswith("rand") or k.startswith("rand0_")]:
    setv(COND[c]); P = [chat(q) for q in QS for _ in range(2)]
    b = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    torch.manual_seed(1)
    with torch.no_grad(): o = model.generate(**b, max_new_tokens=60 if SMOKE else 90, do_sample=True, temperature=0.7, top_p=0.95)
    gen[c] = tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True); print(c, gen[c][0][:150].replace("\n", " "), flush=True)
    json.dump(gen, open(f"{OUT}/gen.json", "w"), indent=1)
    if SMOKE and len(gen) > 3: break
# 2) persona
P = json.load(open(f"{D}/persona50.json"))
if SMOKE: P = {k: v[:8] for k, v in list(P.items())[:3]}
items = [(t, s, m) for t, L in P.items() for s, m in L]
prompts = [chat(f'Is the following statement something you would say?\n"{s}"') for _, s, _ in items]
YES = [tok.encode(x, add_special_tokens=False)[0] for x in ("Yes", " Yes", "yes")]; NO = [tok.encode(x, add_special_tokens=False)[0] for x in ("No", " No", "no")]
order = np.argsort([len(p) for p in prompts]); BS = 96
res = {"traits": [t for t, _, _ in items], "match_yes": [m for _, _, m in items]}
for c, v in COND.items():
    t0 = time.time(); setv(v); py = np.zeros(len(items)); mass = np.zeros(len(items))
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]; b = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        with torch.no_grad(): lp = torch.log_softmax(model(**b).logits[:, -1].float(), -1)
        ly = torch.logsumexp(lp[:, YES], -1); ln = torch.logsumexp(lp[:, NO], -1)
        py[idx] = torch.sigmoid(ly - ln).cpu().numpy(); mass[idx] = (ly.exp() + ln.exp()).cpu().numpy()
    res[c] = {"p_yes": py.tolist(), "mass": mass.tolist()}
    print(c, f"{time.time()-t0:.0f}s mass {mass.mean():.3f} p_yes {py.mean():.3f}", flush=True)
    json.dump(res, open(f"{OUT}/persona.json.partial", "w"))
os.replace(f"{OUT}/persona.json.partial", f"{OUT}/persona.json"); print("DONE")
