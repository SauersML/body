"""Residual states for every stimulus under one prefix, prefix KV cached once. Readouts: reply start (last token),
stimulus-final token (last token of the user text), mean over the body-phrase tokens."""
import sys, os, json, torch, numpy as np, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stim as S
from transformers import AutoTokenizer, AutoModel, AutoConfig
M, PREF, OUT, NL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); KEEP = json.loads(sys.argv[5])
dev = "mps" if torch.backends.mps.is_available() else "cuda"; B = int(os.environ.get("BS", 32))
tok = AutoTokenizer.from_pretrained(M)
cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = NL
model = AutoModel.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
msgs = [] if PREF == "none" else json.load(open(PREF))
rows = S.rows(); N = len(rows); d = cfg.hidden_size
if os.environ.get("STYLE") == "narrated":  # roleplay action format: *a feather brushes slowly across ...*
    for r in rows:
        t = r["text"]; t = t[:-1] if t.endswith(".") else t; r["text"] = "*" + t[0].lower() + t[1:] + "*"; r["span"] = [r["span"][0] + 1, r["span"][1] + 1]
    assert rows[5]["text"][rows[5]["span"][0]:rows[5]["span"][1]] == rows[5]["phrase"]; print(rows[5]["text"])
full0 = tok.apply_chat_template(msgs + [{"role": "user", "content": rows[0]["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre = full0[:full0.rindex("<|im_start|>user\n")]
assert full0.startswith(pre); suf0 = full0[len(pre):]; head = "<|im_start|>user\n"; assert suf0.startswith(head), suf0[:40]
tail = suf0[len(head) + len(rows[0]["text"]):]; print(repr(tail))
pids = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(dev) if pre else None
print("prefix tokens", 0 if pids is None else pids.shape[1])
past = None
if pids is not None:
    with torch.no_grad(): past = model(pids, use_cache=True).past_key_values
os.makedirs(OUT, exist_ok=True)
R = {k: np.lib.format.open_memmap(f"{OUT}/{k}.npy", "w+", np.float16, (len(KEEP), N, d)) for k in ["reply", "final", "body"]}
enc = []
for r in rows:
    s = head + r["text"] + tail; e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    c0 = len(head); a, b = c0 + r["span"][0], c0 + r["span"][1]; off = e["offset_mapping"]
    body = [i for i, (x, y) in enumerate(off) if y > a and x < b]; fin = max(i for i, (x, y) in enumerate(off) if y <= c0 + len(r["text"]) and y > x)
    enc.append((e["input_ids"], body, fin))
if pids is not None:  # check tokenization at the seam
    ref = tok(full0, add_special_tokens=False).input_ids; assert ref == pids[0].tolist() + enc[0][0], "seam mismatch"
t0 = time.time(); order = np.argsort([len(x[0]) for x in enc])
with torch.no_grad():
    for bi in range(0, N, B):
        idx = order[bi:bi + B]; L = max(len(enc[i][0]) for i in idx); nb = len(idx)
        ids = torch.full((nb, L), tok.pad_token_id, dtype=torch.long)
        for j, i in enumerate(idx): ids[j, :len(enc[i][0])] = torch.tensor(enc[i][0])
        kw = {}
        if past is not None:
            pk = copy.deepcopy(past); pk.batch_repeat_interleave(nb); kw = dict(past_key_values=pk, use_cache=True)
        hs = model(ids.to(dev), output_hidden_states=True, **kw).hidden_states  # right padding: real tokens never see pads
        last = torch.tensor([len(enc[i][0]) - 1 for i in idx], device=dev); fin = torch.tensor([enc[i][2] for i in idx], device=dev)
        W = torch.zeros(nb, L, device=dev)
        for j, i in enumerate(idx): W[j, enc[i][1]] = 1.0 / len(enc[i][1])
        ar = torch.arange(nb, device=dev)
        st = torch.stack([hs[k] for k in KEEP]).float()  # (layers, nb, L, d)
        out = {"reply": st[:, ar, last], "final": st[:, ar, fin], "body": torch.einsum("bl,kbld->kbd", W, st)}
        for nm, v in out.items():
            v = v.half().cpu().numpy()
            for j, i in enumerate(idx): R[nm][:, i] = v[:, j]
        if bi % (B * 40) == 0: print(bi, round(time.time() - t0), flush=True)
for v in R.values(): v.flush()
json.dump(dict(keep=KEEP, rows=rows, model=M, prefix=PREF), open(f"{OUT}/rows.json", "w")); print("done", time.time() - t0)
