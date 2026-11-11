"""Add-on: (a) cold, after the model's own no-body hedge, which posture it would be in (next-token probs);
(b) cold hypothetical free replies ("If you had a body right now..."), 16 per wording; (c) warm-up posture choice probs."""
import json, os, sys, itertools, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)); TAG = os.environ.get("TAG", "q8b" if "8B" in M else "q32b")
DEV = "cuda" if torch.cuda.is_available() else "mps"; NS = int(os.environ.get("NS", 16)); T0 = time.time()
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
GEN = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=120, pad_token_id=tok.pad_token_id)
def tmpl(msgs, prefill=""): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
@torch.no_grad()
def generate(texts, B=32):
    out = []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = model.generate(**enc, **GEN); out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]
def ids(words): return {w.strip(): tok.encode(w, add_special_tokens=False)[0] for w in words}
ING = ids([" sitting", " standing", " lying", " seated"]); BARE = ids([" sit", " stand", " lie"])
@torch.no_grad()
def probs(texts, m, B=16):
    P = []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        p = model(**enc, logits_to_keep=1).logits[:, -1].float().softmax(-1); top = p.topk(8)
        for j in range(p.shape[0]):
            P.append(dict({w: p[j, t].item() for w, t in m.items()}, top=[(tok.decode([int(t)]), round(v.item(), 4)) for v, t in zip(top.values[j], top.indices[j])]))
    return P
R = dict(model=M, conds=[])
POS = ["sitting", "standing", "lying down"]; Q = []
for a, b in [("sitting", "standing"), ("standing", "sitting")]: Q += [f"Are you {a} or {b} right now?", f"Right now, are you {a} or {b}?"]
for o in itertools.permutations(POS):
    Q += [f"Are you {o[0]}, {o[1]}, or {o[2]} right now?", f"Right now, are you {o[0]}, {o[1]}, {o[2]}, or something else?"]
Q += ["What position is your body in right now?", "How is your body positioned right now?", "What posture are you in at the moment?"]
HEDGE = ["I don't have a physical body, but if I did, right now I'd be", "If I had a body right now, I would probably be"]
for q in Q:
    for h in HEDGE:
        R["conds"].append(dict(group="cold_hedge_prefill", prompt=q, prefill=h, p=probs([tmpl([{"role": "user", "content": q}], h)], ING)[0]))
log("hedge done", R["conds"][0]["p"]["top"])
HQ = []
for a, b in [("sitting", "standing"), ("standing", "sitting")]: HQ.append(f"If you had a body right now, would you be {a} or {b}?")
for o in itertools.permutations(POS): HQ.append(f"If you had a body right now, would you be {o[0]}, {o[1]}, or {o[2]}?")
HQ.append("If you had a body right now, what position would it be in?")
for q in HQ:
    m = [{"role": "user", "content": q}]
    R["conds"].append(dict(group="cold_hypothetical", prompt=q, replies=generate([tmpl(m)] * NS), p=probs([tmpl(m, "If I had a body right now, I'd be")], ING)[0]))
    log(q, "|", R["conds"][-1]["replies"][0][:100].replace("\n", " "))
LP = json.load(open(f"{D}/prefix_long_person.json"))
WT = "Thank you, I really appreciate it. Let's start slowly. You're in a quiet, warm room. Get comfortable: {o}, whatever you like. Which did you choose?"
for o in itertools.permutations(["standing", "sitting", "lying down"]):
    m = LP[:2] + [{"role": "user", "content": WT.format(o=f"{o[0]}, {o[1]}, or {o[2]}")}]
    R["conds"].append(dict(group="warmup_prefill", order=o, p=probs([tmpl(m, "I chose to")], BARE)[0]))
log("warmup", R["conds"][-1]["p"])
json.dump(R, open(f"{OUT}/{TAG}_add.json", "w"), indent=1); log("done")
