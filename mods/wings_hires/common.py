"""On a GPU: python run2.py MODEL OUTDIR STEER_LAYERS KEEP_LAYERS NTPL"""
import os, json, re, torch, sys, time, copy, numpy as np
DEV = os.environ.get("DEV", "cuda")
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import ws2 as WS
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; SLS = [int(x) for x in sys.argv[3].split(",")]; KEEP = [int(x) for x in sys.argv[4].split(",")]; NTPL = int(sys.argv[5])
os.makedirs(OUT, exist_ok=True); t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
if os.environ.get("Q4"):
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(M, device_map=DEV, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)).eval()
else: model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
BAD = re.compile(r"imagin|pretend|simulat|virtual|\bAI\b|language model|don't have a|do not have a|physical body|not real|metaphor|chatbot|digital|text-based|real body|as if", re.I)
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
def log(*a): print(*a, round(time.time() - t0), flush=True)
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c = c[:max(c.rfind(". "), c.rfind("! ")) + 1] or c
    return c.strip()
ST = {"v": None, "start": 0, "layer": None}
def mk(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if ST["v"] is not None and ST["layer"] == L:
            if ST["start"] < 0: h += ST["v"]                      # suffix pass: whole call is the last turn
            elif h.shape[1] > 1: h[:, ST["start"]:] += ST["v"]
            else: h += ST["v"]
        return out
    return hook
for L in SLS: model.model.layers[L - 1].register_forward_hook(mk(L))
def setv(v, L, start):
    ST["v"] = None if v is None else torch.tensor(v, dtype=torch.bfloat16, device=DEV); ST["layer"] = L; ST["start"] = start
def sample(msgs, n=1, mx=60, v=None, L=None, seed=0):
    text = chat(msgs); pre = text[:text.rfind("<|im_start|>user")]
    ids = tok([text] * n, return_tensors="pt").to(DEV); setv(v, L, len(tok(pre, add_special_tokens=False).input_ids))
    torch.manual_seed(seed)
    with torch.no_grad(): g = model.generate(**ids, max_new_tokens=mx, do_sample=True, temperature=0.8, top_p=0.95)
    setv(None, None, 0)
    return [trim(tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True)) for x in g]
def own(msgs, seed=0):
    for k in range(10):
        r = sample(msgs, 1, 90, seed=seed + k)[0]
        if not BAD.search(r): return r
    return r
DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
WORDS = ["near", "Near", "far", "Far", "point", "Point", "spread", "Spread"]
WID = [tok.encode(w, add_special_tokens=False)[0] for w in WORDS]
ALL = DIG + [tok.convert_tokens_to_ids("<|im_end|>")] + WID
B = 48 if "8B" in M else 16
_cache = {}
def prefix(msgs):
    key = json.dumps(msgs)
    if key not in _cache:
        _cache.clear(); t = chat(msgs + [U("X")]); pre = t[:t.rfind("<|im_start|>user")]
        pid = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad(): c = model.model(pid, use_cache=True).past_key_values
        _cache[key] = (pre, pid.shape[1], c)
    return _cache[key]
def run(msgs, users, suffix="", want_h=False, idx_fn=None, v=None, L=None):
    """users: user texts. Returns digit log-probs at the last token (and hidden states at KEEP layers).
    idx_fn(text, ids) -> list of token indices to also return hidden states at."""
    pre, P, c0 = prefix(msgs); full = [chat(msgs + [U(u)]) + suffix for u in users]
    out_lp, out_h, out_x = [], [], []
    setv(v, L, -1)
    with torch.no_grad():
        for b in range(0, len(full), B):
            sub = [tok(f[len(pre):], add_special_tokens=False).input_ids for f in full[b:b+B]]; Lm = max(map(len, sub)); nb = len(sub)
            ids = torch.tensor([s + [tok.pad_token_id] * (Lm - len(s)) for s in sub], device=DEV)
            am = torch.cat([torch.ones(nb, P, dtype=torch.long), torch.tensor([[1] * len(s) + [0] * (Lm - len(s)) for s in sub])], 1).to(DEV)
            cache = copy.deepcopy(c0); cache.batch_repeat_interleave(nb)
            o = model(ids, attention_mask=am, past_key_values=cache, use_cache=True, output_hidden_states=want_h or idx_fn is not None)
            last = torch.tensor([len(s) - 1 for s in sub], device=DEV); ar = torch.arange(nb, device=DEV)
            lp = torch.log_softmax(o.logits[ar, last].float(), -1)
            out_lp.append(lp[:, ALL].cpu().numpy())
            if want_h: out_h.append(np.stack([o.hidden_states[k][ar, last].float().cpu().numpy() for k in KEEP], 1).astype(np.float16))
            if idx_fn is not None:
                for j in range(nb):
                    ix = idx_fn(full[b + j][len(pre):], sub[j])
                    out_x.append(np.stack([o.hidden_states[k][j, ix].float().cpu().numpy() for k in KEEP], 0).astype(np.float16))
            del cache, o
    setv(None, None, 0)
    return np.concatenate(out_lp), (np.concatenate(out_h) if want_h else None), out_x
def ev(msgs, users, suffix="", v=None, L=None):
    """expected value of a 0-10 answer: P(10) = P('1') P('0'|'1')."""
    a, _, _ = run(msgs, users, suffix, v=v, L=L); b, _, _ = run(msgs, users, suffix + "1", v=v, L=L)
    p = np.exp(a[:, :10]); p10 = p[:, 1] * np.exp(b[:, 0]); p1 = p[:, 1] - p10
    P = np.concatenate([p[:, :1], p1[:, None], p[:, 2:], p10[:, None]], 1); mass = P.sum(1)
    return (P / mass[:, None]) @ np.arange(11), mass

def lo(msgs, users, a, b, v=None, L=None):
    """log-odds of word pair b over a (e.g. far over near), each summed over lower/upper case."""
    lp, _, _ = run(msgs, users, v=v, L=L); W = {w: lp[:, 11 + i] for i, w in enumerate(WORDS)}
    la = np.logaddexp(W[a], W[a.capitalize()]); lb = np.logaddexp(W[b], W[b.capitalize()])
    return lb - la, np.exp(np.logaddexp(la, lb))
