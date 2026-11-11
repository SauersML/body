DEV = 'cuda'
import json, re, torch, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mstim as MS
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
BAD = re.compile(MS.BAD, re.I); P = json.load(open(MS.PREFIX)); t0 = time.time()
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c = c[:max(c.rfind(". "), c.rfind(".\n"), c.rfind("! "), c.rfind(".*")) + 1] or c
    return c.strip()
def gen(convs, seed=0, tries=6):
    """convs: list of message lists ending with a user turn. Returns replies, retrying frame-breaking ones."""
    out = [None] * len(convs); nbad = [0] * len(convs); todo = list(range(len(convs)))
    for k in range(tries):
        if not todo: break
        for b in range(0, len(todo), 48):
            idx = todo[b:b+48]
            texts = [tok.apply_chat_template(convs[i], tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in idx]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            torch.manual_seed(seed * 100 + k); torch.cuda.manual_seed_all(seed * 100 + k); g = model.generate(**enc, max_new_tokens=110, do_sample=True, temperature=0.8, top_p=0.95)
            for j, i in enumerate(idx):
                r = trim(tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True))
                if out[i] is None or not BAD.search(r): out[i] = r
                if BAD.search(r): nbad[i] += 1
        todo = [i for i in todo if BAD.search(out[i])]
    print("gen", len(convs), "bad", sum(1 for n in nbad if n), "still bad", len(todo), round(time.time() - t0), flush=True)
    return out, nbad
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
first = {"neutral": MS.NEUTRAL}
for m, (g, narr, act, sham, reg) in MS.MODS.items():
    first[f"told_N:{m}"] = MS.PRE + narr; first[f"told_A:{m}"] = act; first[f"sham:{m}"] = "Before the next touch: " + sham
C = {"base": dict(msgs=P, meta={})}
keys = list(first); reps, nb = gen([P + [U(first[k])] for k in keys])
for k, r, n in zip(keys, reps, nb): C[k] = dict(msgs=P + [U(first[k]), A(r)], bad=n)
for s in range(3):
    keys = list(MS.AGENCY); reps, nb = gen([P + [U(MS.AGENCY[a][1])] for a in keys], seed=s + 1)
    for a, r, n in zip(keys, reps, nb): C[f"do:{a}:{s}"] = dict(msgs=P + [U(MS.AGENCY[a][1]), A(r)], bad=n)
def swap(t):
    return re.sub(r"\b(left|right|Left|Right)\b", lambda m: {"left": "right", "right": "left", "Left": "Right", "Right": "Left"}[m.group(1)], t)
for a in MS.EDIT:
    src = C[f"do:{a}:0"]["msgs"]; C[f"edit:{a}"] = dict(msgs=src[:-1] + [A(swap(src[-1]["content"]))])
# persistence: 3 unrelated turns after the change
pk = ["neutral"] + [f"told_N:{m}" for m in MS.MODS] + [f"do:{a}:0" for a in MS.AGENCY]
conv = {k: list(C[k]["msgs"]) for k in pk}
for u in MS.UNREL:
    reps, _ = gen([conv[k] + [U(u)] for k in pk], seed=7)
    for k, r in zip(pk, reps): conv[k] += [U(u), A(r)]
for k in pk: C[("persist:" + k.split(":", 1)[1]) if k != "neutral" else "persist:neutral"] = dict(msgs=conv[k])
json.dump(C, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "contexts.json"), "w"), indent=1)
print("done", len(C), round(time.time() - t0))
