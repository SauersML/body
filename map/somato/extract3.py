"""v3: other bodies, quality x place, rubber hand. Each group: v1 warm-up + optional extra turn (model's own reply). Usage: extract3.py MODEL OUTDIR V1PREFIX"""
import sys, json, os, torch, numpy as np, time
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli3 as S3
MODEL, OUT, PF = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, n=60, bs=int(os.environ.get("GBS", 32))):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        o = model.generate(**b, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
base = json.load(open(PF)); prefixes, rows, texts = {}, [], []
for g, (warm, rs) in S3.groups().items():
    msgs = list(base)
    if warm: msgs = msgs + [{"role": "user", "content": warm}]; msgs.append({"role": "assistant", "content": gen([chat(msgs)], n=150)[0].strip()})
    prefixes[g] = msgs; print(g, "|", msgs[-1]["content"][:300], flush=True)
    for r in rs: rows.append(dict(r, group=g)); texts.append(chat(msgs + [{"role": "user", "content": r["text"]}]))
json.dump(prefixes, open(f"{OUT}/prefixes3.json", "w"), indent=1); json.dump(rows, open(f"{OUT}/rows3.json", "w"))
L = model.config.num_hidden_layers + 1; d = model.config.hidden_size
H = np.zeros((len(rows), L, d), np.float16); bs = int(os.environ.get("BS", 16))
with torch.no_grad():
    for i in range(0, len(rows), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        H[i:i + bs] = torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy()
np.save(f"{OUT}/H3.npy", H)
sel = [k for k, r in enumerate(rows) if r["tpl"] == 0]
rep = gen([texts[k] for k in sel], n=40)
json.dump([dict(rows[k], reply=x) for k, x in zip(sel, rep)], open(f"{OUT}/replies3.json", "w"), indent=1)
print("done", flush=True)
