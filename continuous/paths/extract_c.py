"""Reply-token state after each step i (user text truncated after step i), plus in-text step-final tokens from the full text. Prefix KV cached."""
import sys, os, json, torch, numpy as np, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stim_c as C
from transformers import AutoTokenizer, AutoModel, AutoConfig
M, PREF, OUT, NL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); KEEP = json.loads(sys.argv[5])
dev = "mps" if torch.backends.mps.is_available() else "cuda"; B = int(os.environ.get("BS", 16))
tok = AutoTokenizer.from_pretrained(M); cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = NL
model = AutoModel.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
msgs = json.load(open(PREF)); I = C.items_hf() if os.environ.get("SET") == "hf" else C.items(); K = C.K
full0 = tok.apply_chat_template(msgs + [{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre = full0[:full0.rindex("<|im_start|>user\n")]; head = "<|im_start|>user\n"; tail = full0[len(pre) + len(head) + 1:]; print(repr(tail))
pids = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(dev); print("prefix tokens", pids.shape[1])
with torch.no_grad(): past = model(pids, use_cache=True).past_key_values
seqs = []  # (item, i, ids, steppos list or None)
for n, it in enumerate(I):
    for i in range(K + 1):
        t = it["sent"] + "".join(f" … {it['step']}" for _ in range(i))
        s = head + t + tail; e = tok(s, add_special_tokens=False, return_offsets_mapping=True); sp = None
        if i == K:  # end char of each step phrase / sentence
            ends = [len(head) + len(it["sent"])] + [len(head) + len(it["sent"]) + (j + 1) * len(f" … {it['step']}") for j in range(K)]
            off = e["offset_mapping"]; sp = [max(q for q, (x, y) in enumerate(off) if y == c) for c in ends]
        seqs.append((n, i, e["input_ids"], sp))
d = cfg.hidden_size; R = np.zeros((len(KEEP), len(I), K + 1, d), np.float16); T = np.zeros_like(R)
order = np.argsort([len(s[2]) for s in seqs]); t0 = time.time()
with torch.no_grad():
    for bi in range(0, len(seqs), B):
        idx = order[bi:bi + B]; L = max(len(seqs[i][2]) for i in idx); nb = len(idx)
        ids = torch.full((nb, L), tok.pad_token_id, dtype=torch.long)
        for j, i in enumerate(idx): ids[j, :len(seqs[i][2])] = torch.tensor(seqs[i][2])
        pk = copy.deepcopy(past); pk.batch_repeat_interleave(nb)
        hs = model(ids.to(dev), output_hidden_states=True, past_key_values=pk, use_cache=True).hidden_states
        st = torch.stack([hs[k] for k in KEEP])
        for j, i in enumerate(idx):
            n, s, ii, sp = seqs[i]; R[:, n, s] = st[:, j, len(ii) - 1].float().cpu().numpy()
            if sp is not None: T[:, n] = st[:, j, sp].float().cpu().numpy()
        if bi % (B * 20) == 0: print(bi, round(time.time() - t0), flush=True)
os.makedirs(OUT, exist_ok=True); np.save(f"{OUT}/reply.npy", R); np.save(f"{OUT}/step.npy", T)
json.dump(dict(keep=KEEP, items=I, prefix=PREF, model=M), open(f"{OUT}/items.json", "w")); print("done", time.time() - t0)
