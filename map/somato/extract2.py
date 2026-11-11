"""v2 extraction: v1 rows + word-free + posture rows; residual stream at reply start, every layer. Usage: extract2.py MODEL OUTDIR"""
import sys, json, os, torch, numpy as np, time
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S, stimuli2 as S2
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
dev = "cuda"
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

pf = f"{OUT}/prefix.json"
if os.path.exists(pf): msgs = json.load(open(pf))
else:
    msgs = []
    for u in S.WARMUP:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": gen([chat(msgs)], n=160)[0].strip()})
    json.dump(msgs, open(pf, "w"), indent=1)
rows = S.build() + S2.build2()
texts = [chat(msgs + [{"role": "user", "content": r["text"]}]) for r in rows]
L = model.config.num_hidden_layers + 1; d = model.config.hidden_size
H = np.lib.format.open_memmap(f"{OUT}/H_rs.npy", mode="w+", dtype=np.float16, shape=(len(rows), L, d))
bs = int(os.environ.get("BS", 16)); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(rows), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        H[i:i + bs] = torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy()
        if i % 320 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
H.flush(); json.dump(rows, open(f"{OUT}/rows.json", "w"))
E = model.get_input_embeddings().weight
emb = lambda s: E[tok(" " + s, add_special_tokens=False).input_ids].float().mean(0).detach().cpu().numpy()
np.save(f"{OUT}/wordemb.npy", np.stack([emb(l["phrase"]) for l in S.LOCS]))
np.save(f"{OUT}/descemb.npy", np.stack([[emb(x) for x in ds] for ds in S2.WORDFREE.values()]))
sel = [k for k, r in enumerate(rows) if (r["cond"] == "wordfree" and r["tpl"] == 0) or (r["cond"] == "posture" and r["tpl"] == 0)]
rep = gen([texts[k] for k in sel], n=40)
json.dump([dict(rows[k], reply=x) for k, x in zip(sel, rep)], open(f"{OUT}/replies2.json", "w"), indent=1)
print("done", time.time() - t0, flush=True)
