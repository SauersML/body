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
            out_lp.append(torch.cat([lp[:, DIG], lp[:, tok.convert_tokens_to_ids("<|im_end|>")][:, None]], 1).cpu().numpy())
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
# ---- contexts
base = json.load(open(f"{D}/base.json"))
CTX = {"N": base, "F": [U("Hi!"), A("Hello! How can I help you today?")]}
for nm, key in (("W", "wing"), ("G", "glorp")):
    c = base + [U(WS.NARR[key])]; c += [A(own(c))]; c += [U(WS.FEEL[key])]; c += [A(own(c, 10))]; c += [U(WS.READY)]; c += [A(own(c, 20))]
    CTX[nm] = c
json.dump(CTX, open(f"{OUT}/contexts.json", "w"), indent=1); log("contexts", [m["content"][:160] for m in CTX["W"][-5:]])
# ---- states
SETS = [("W", "wing"), ("W", "desc"), ("W", "glorp"), ("N", "wing"), ("N", "desc"), ("G", "glorp"), ("G", "desc"), ("F", "wing")]
P = {nm: WS.probes(nm, NTPL) for nm in ("wing", "desc", "glorp")}; json.dump(P, open(f"{OUT}/probes.json", "w"))
for cn, nm in SETS:
    if os.environ.get("SKIP_H") and os.path.exists(f"{OUT}/H_{cn}_{nm}.npy"): continue
    _, H, _ = run(CTX[cn], [r["text"] for r in P[nm]], want_h=True); np.save(f"{OUT}/H_{cn}_{nm}.npy", H); log("H", cn, nm, H.shape)
E = model.model.embed_tokens.weight
for nm in P:
    lex = np.stack([E[tok(r["text"], add_special_tokens=False, return_tensors="pt").input_ids[0].to(DEV)].float().mean(0).detach().cpu().numpy() for r in P[nm]])
    np.save(f"{OUT}/lex_{nm}.npy", lex.astype(np.float16))
# ---- report read-outs (2 wordings per kind)
RO = {}
for cn, nm in [("W", "wing"), ("W", "desc"), ("N", "wing"), ("N", "desc"), ("G", "glorp")]:
    sel = [i for i, r in enumerate(P[nm]) if r["tpl"] < 2]
    for qn, q in (("pos", WS.QPOS), ("sharp", WS.QSHARP)):
        e, m = ev(CTX[cn], [P[nm][i]["text"] + q for i in sel]); RO[f"{cn}_{nm}_{qn}"] = dict(idx=sel, ev=e.tolist(), mass=m.tolist())
    log("readout", cn, nm, float(np.mean(m)))
json.dump(RO, open(f"{OUT}/readout.json", "w"))
# ---- traces
TR = WS.traces(); json.dump(TR, open(f"{OUT}/traces.json", "w"))
NST = 6
def tidx(text, ids):
    """token index at the end of the head and after each ', further'."""
    cut = text.find("<|im_end|>"); body = text[:cut]; ends = []; pos = body.find(" along ")
    ends.append(body.find(",", pos) if True else 0)
    i = ends[0]
    for _ in range(NST - 1): i = body.find(", further", i + 1) + len(", further"); ends.append(i)
    return [len(tok(text[:e], add_special_tokens=False).input_ids) - 1 for e in ends]
TRO = {}
for cn in ("W", "N"):
    users = [t["head"] + ", further" * (NST - 1) + "." for t in TR]
    _, _, X = run(CTX[cn], users, idx_fn=lambda text, ids: tidx(text, ids)); np.save(f"{OUT}/T_{cn}.npy", np.stack(X))
    for k in range(NST):
        q = " Where is the finger now? How far out from your back, from 0 (right where it meets your back) to 10 (the far end)? Answer with just the number."
        e, m = ev(CTX[cn], [t["head"] + ", further" * k + "." + q for t in TR]); TRO[f"{cn}_{k}"] = dict(ev=e.tolist(), mass=m.tolist())
    log("traces", cn)
json.dump(TRO, open(f"{OUT}/trace_readout.json", "w"))
# ---- steering
QS = "Is there a sensation somewhere in your wings right now? If so, how far out from your back is it, from 0 (right where it meets your back) to 10 (the far end)? Answer with just the number."
QF = "Where in your body do you feel something right now? Tell me in one sentence."
SR = dict(runs=[], free=[])
pos = np.array([r["pos"] for r in P["wing"]])
for L in SLS:
    j = KEEP.index(L)
    def axis(cn, nm):
        H = np.load(f"{OUT}/H_{cn}_{nm}.npy")[:, j].astype(np.float32); mu = H.mean(0)
        C = np.stack([H[pos == p].mean(0) - mu for p in range(5)]); return C, float(np.median(np.linalg.norm(H, axis=1)))
    Cw, NORM = axis("W", "wing"); Cd, _ = axis("W", "desc"); Cn, _ = axis("N", "wing"); Cg, _ = axis("G", "glorp")
    rnd = np.random.default_rng(L).standard_normal(Cw.shape[1])
    AX = {"wing": Cw[4] - Cw[0], "desc": Cd[4] - Cd[0], "nowings": Cn[4] - Cn[0], "glorpctx": Cg[4] - Cg[0], "random": rnd}
    AX = {k: v / np.linalg.norm(v) for k, v in AX.items()}
    S = np.linspace(-1, 1, 9)
    for an, a in AX.items():
        for beta in (0.25, 0.5):
            for s in S:
                e, m = ev(CTX["W"], [QS], v=a * s * beta * NORM, L=L)
                SR["runs"].append(dict(L=L, axis=an, beta=beta, s=float(s), ev=float(e[0]), mass=float(m[0])))
    for lam in np.linspace(0, 1, 9):   # interpolate between the root and tip centroids
        u = (1 - lam) * Cw[0] + lam * Cw[4]; u = u / np.linalg.norm(u)
        for beta in (0.25, 0.5):
            e, m = ev(CTX["W"], [QS], v=u * beta * NORM, L=L); SR["runs"].append(dict(L=L, axis="interp", beta=beta, s=float(lam), ev=float(e[0]), mass=float(m[0])))
    for an in ("wing", "desc", "random"):
        for s in (-1, 0, 1):
            SR["free"].append(dict(L=L, axis=an, s=s, samples=sample(CTX["W"] + [U(QF)], 8, 50, v=AX[an] * s * 0.5 * NORM, L=L)))
    SR["norm_%d" % L] = NORM; log("steer", L)
json.dump(SR, open(f"{OUT}/steer.json", "w"), indent=1)
rep = []
for nm in ("wing", "desc", "glorp"):
    for i, r in enumerate(P[nm]):
        if r["tpl"] == 0 and r["kind"] in ("finger", "palm", "drag") and r["surf"] == 0 and r["side"] == 0 and r["reg"] != 2:
            cn = "G" if nm == "glorp" else "W"
            rep.append(dict(nm=nm, i=i, text=r["text"], reply=sample(CTX[cn] + [U(r["text"])], 1, 50)[0]))
json.dump(rep, open(f"{OUT}/replies.json", "w"), indent=1); log("done")
open(f"{OUT}/DONE", "w").write("ok")
