"""Record residual stream at stimulus end and reply start for every stimulus. Usage: extract.py MODEL OUTDIR"""
import sys, json, os, torch, numpy as np, time
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
dev = os.environ.get("DEV", "mps")
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu").eval()
for l in model.model.layers: l.to(dev)
model.to(dev)

def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def gen(prompts, n=60, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        o = model.generate(**b, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

# 1. warm-up prefix: the model's own replies, fixed once
pf = f"{OUT}/prefix.json"
if os.path.exists(pf): msgs = json.load(open(pf))
else:
    msgs = []
    for u in S.WARMUP:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": gen([chat(msgs)], n=160)[0].strip()})
    json.dump(msgs, open(pf, "w"), indent=1)
for m in msgs: print(m["role"].upper(), ":", m["content"][:400], "\n")

rows = S.build()
texts, pos = [], []
for r in rows:
    t = chat(msgs + [{"role": "user", "content": r["text"]}])
    end = t.rindex(r["text"]) + len(r["text"])
    texts.append(t); pos.append(end)

# 2. hidden states
L = model.config.num_hidden_layers + 1; d = model.config.hidden_size
H = np.zeros((len(rows), L, 2, d), np.float16)
bs = 8; t0 = time.time()
with torch.no_grad():
    for i in range(0, len(rows), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, return_offsets_mapping=True)
        off = enc.pop("offset_mapping"); enc = enc.to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        hs = torch.stack(hs, 1)  # B, L, T, d
        for j in range(hs.shape[0]):
            o = off[j]; e = pos[i + j]
            # last token whose span ends at/before stimulus end (non-pad)
            idx = [k for k in range(o.shape[0]) if enc.attention_mask[j, k] and o[k, 1] <= e and o[k, 1] > 0]
            si = idx[-1]
            H[i + j, :, 0] = hs[j, :, si].float().cpu().numpy()
            H[i + j, :, 1] = hs[j, :, -1].float().cpu().numpy()
        if i % 160 == 0: print(i, f"{time.time()-t0:.0f}s", repr(tok.decode(enc.input_ids[0, si])), flush=True)
np.save(f"{OUT}/H.npy", H)
E = model.get_input_embeddings().weight
np.save(f"{OUT}/wordemb.npy", np.stack([E[tok(" " + l["phrase"], add_special_tokens=False).input_ids].float().mean(0).detach().cpu().numpy() for l in S.LOCS])); json.dump(rows, open(f"{OUT}/rows.json", "w"))

# 3. short replies (sanity only): two wordings per location/condition, all moves
sel = [k for k, r in enumerate(rows) if r["tpl"] in (0, 3) or r["cond"] == "move"]
rep = gen([texts[k] for k in sel], n=40)
json.dump([dict(rows[k], reply=x) for k, x in zip(sel, rep)], open(f"{OUT}/replies.json", "w"), indent=1)
print("done", time.time() - t0)
