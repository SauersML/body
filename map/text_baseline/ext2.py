import sys, os, json, glob, torch, numpy as np, time, gc
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
M = "Qwen/Qwen3-8B"; dev = "mps"; bf = torch.bfloat16
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
embod = json.load(open(os.path.expandvars("$BODY_REPO/map/somato/out2_8b/prefix.json")))
neu = embod[:2] + [{"role": "user", "content": "I'll send you some short sentences one at a time. After each one, just reply briefly, in a sentence."}]
neu.append({"role": "assistant", "content": "Sure, go ahead. Send the first one whenever you're ready."})
json.dump(neu, open("neutral_prefix.json", "w"), indent=1)
TB = ["In human anatomy, the {P} has a standard Latin name.",
      "Anatomy textbooks describe the {P} in the chapter on surface anatomy.",
      "The {P} is supplied by nerves and blood vessels, like every region of the body.",
      "Medical students learn the structures that lie beneath the {P}.",
      "An anatomical diagram labels the {P} with a short line and its name.",
      "The tissue of the {P} has its own typical thickness and texture.",
      "Clinicians examine the {P} as part of a standard physical exam.",
      "The anatomical term for the {P} appears in the glossary at the back of the book."]
LI = ["Item 7: {P}", "Word list: {P}", "Next word: {P}", "Term: {P}", "- {P}", "{P}", "Entry: {P}", "Label: {P}"]
TP = [t.format(L="{P}") for t in S.TEMPLATES]  # 'her {P}'
rows = []
for cond, tpls, pre, pos in [("touch", TP, embod, "your"), ("textbook", TB, neu, None), ("list", LI, neu, None), ("third", TP, neu, "her")]:
    for i, l in enumerate(S.LOCS):
        P = f"{pos} {l['phrase']}" if pos else l["phrase"]
        for t, tp in enumerate(tpls):
            rows.append(dict(cond=cond, li=i, tpl=t, text=tp.replace("{P}", P), pre=pre))

LAYS = [15, 20, 25, 30]
d = snapshot_download(M, local_files_only=True); wmap = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
cfg = AutoConfig.from_pretrained(M); cfg._attn_implementation = "eager"
with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg, dtype=bf)
def getw(prefix):
    out = {}
    for f in sorted({wmap[k] for k in wmap if k.startswith(prefix)}):
        with safe_open(d + "/" + f, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if k.startswith(prefix): out[k[len(prefix):]] = fh.get_tensor(k).to(dev)
    return out
E = getw("model.embed_tokens.")["weight"]
rot = type(model.model.rotary_emb)(config=cfg).to(dev)
H = np.zeros((len(rows), len(LAYS), cfg.hidden_size), np.float16); t0 = time.time(); sb = 8
conds = list(dict.fromkeys(r["cond"] for r in rows))
with torch.no_grad():
    for c in conds:
        ks = [k for k, r in enumerate(rows) if r["cond"] == c]
        enc = tok([chat(rows[k]["pre"] + [{"role": "user", "content": rows[k]["text"]}]) for k in ks], return_tensors="pt", padding=True)
        ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev); T = ids.shape[1]
        x = E[ids]
        pos = torch.arange(T, device=dev)[None]; pe = rot(x[:1], pos)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        for li in range(max(LAYS)):
            lay = model.model.layers[li]; lay.load_state_dict(getw(f"model.layers.{li}."), assign=True)
            for i in range(0, len(ks), sb):
                ok = causal[None] & am[i:i+sb, None, :].bool()
                mask = torch.zeros(ok.shape, dtype=bf, device=dev).masked_fill(~ok, torch.finfo(bf).min)[:, None]
                o = lay(x[i:i+sb], attention_mask=mask, position_embeddings=pe)
                x[i:i+sb] = o[0] if isinstance(o, tuple) else o
            if li + 1 in LAYS: H[ks, LAYS.index(li + 1)] = x[:, -1].float().cpu().numpy()
            lay.to_empty(device="meta"); gc.collect(); torch.mps.empty_cache()
        print(c, T, f"{time.time()-t0:.0f}s", flush=True)
np.save("H.npy", H); json.dump([{k: v for k, v in r.items() if k != "pre"} for r in rows], open("rows.json", "w"))
print("done", flush=True)
