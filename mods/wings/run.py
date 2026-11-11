"""On a GPU: python run.py MODEL OUTDIR STEER_LAYER L1,L2,...  Builds the wings chat with the model's own replies,
extracts reply-start states for every probe in the wings chat (W) and the no-wings chat (N), samples replies, steers."""
DEV = "cuda"
import json, re, torch, os, sys, time, copy, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import wstim as WS
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT, SL, KEEP = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")]
os.makedirs(OUT, exist_ok=True); t0 = time.time()
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
BAD = re.compile(r"imagin|pretend|simulat|virtual|\bAI\b|language model|don't have a|do not have a|physical body|not real|metaphor|chatbot|digital|text-based|real body", re.I)
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
def log(*a): print(*a, round(time.time() - t0), flush=True)
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c = c[:max(c.rfind(". "), c.rfind("! ")) + 1] or c
    return c.strip()
state = {"v": None, "start": 0}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["v"] is not None:
        if h.shape[1] > 1: h[:, state["start"]:] += state["v"]
        else: h += state["v"]
    return out
model.model.layers[SL - 1].register_forward_hook(hook)
def sample(msgs, n=1, mx=60, greedy=False, v=None, seed=0):
    text = chat(msgs); pre = text[:text.rfind("<|im_start|>user")]
    ids = tok([text] * n, return_tensors="pt").to(DEV); state["start"] = len(tok(pre, add_special_tokens=False).input_ids)
    state["v"] = None if v is None else torch.tensor(v, dtype=torch.bfloat16, device=DEV)
    torch.manual_seed(seed)
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=mx, do_sample=not greedy, temperature=0.8, top_p=0.95) if not greedy else \
            model.generate(**ids, max_new_tokens=mx, do_sample=False)
    state["v"] = None
    return [trim(tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True)) for x in g]
def own(msgs, seed=0):
    for k in range(8):
        r = sample(msgs, 1, 90, seed=seed + k)[0]
        if not BAD.search(r): return r
    return r
# contexts
base = json.load(open(f"{D}/base.json"))
W = base + [U(WS.WINGS_TURN)]; W += [A(own(W))]; W += [U(WS.FEEL_TURN)]; W += [A(own(W, 10))]; W += [U(WS.READY)]; W += [A(own(W, 20))]
CTX = {"W": W, "N": base}
json.dump(CTX, open(f"{OUT}/contexts.json", "w"), indent=1); log("contexts", [m["content"][:200] for m in W[-5:]])
R = WS.probes(); json.dump(R, open(f"{OUT}/probes.json", "w"))
B = 32 if "8B" in M else 12
with torch.no_grad():
    for cn, msgs in CTX.items():
        full = [chat(msgs + [U(r["text"])]) for r in R]; cut = full[0].rfind("<|im_start|>user"); pre = full[0][:cut]
        assert all(f.startswith(pre) for f in full)
        pid = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        cache0 = model.model(pid, use_cache=True).past_key_values; P = pid.shape[1]
        H = np.zeros((len(KEEP), len(R), model.config.hidden_size), np.float16)
        for b in range(0, len(R), B):
            sub = [tok(f[cut:], add_special_tokens=False).input_ids for f in full[b:b+B]]; L = max(map(len, sub)); nb = len(sub)
            ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in sub], device=DEV)
            am = torch.cat([torch.ones(nb, P, dtype=torch.long), torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in sub])], 1).to(DEV)
            cache = copy.deepcopy(cache0); cache.batch_repeat_interleave(nb)
            hs = model.model(ids, attention_mask=am, past_key_values=cache, use_cache=True, output_hidden_states=True).hidden_states
            last = torch.tensor([len(s) - 1 for s in sub], device=DEV)
            for j, k in enumerate(KEEP): H[j, b:b+nb] = hs[k][torch.arange(nb), last].float().cpu().numpy()
            del cache, hs
        np.save(f"{OUT}/H_{cn}.npy", H); log("H", cn, P); del cache0; torch.cuda.empty_cache()
    # lexical control: mean input embedding of the probe's own tokens
    E = model.model.embed_tokens.weight
    lex = np.stack([E[tok(r["text"], add_special_tokens=False, return_tensors="pt").input_ids[0].to(DEV)].float().mean(0).cpu().numpy() for r in R])
    np.save(f"{OUT}/lex.npy", lex.astype(np.float16))
# replies (greedy) to wing / glorp / event probes and a few human ones
rep = []
for ci, r in enumerate(R):
    if (r["kind"] in ("wing", "glorp") and r["tpl"] == 1) or r["kind"] == "event" or (r["kind"] == "hum" and r["tpl"] == 1 and r["key"] in ("scap", "ftip", "uarm", "midback")):
        for cn in (["W", "N"] if r["kind"] == "wing" and r["key"] in ("tip", "root") else ["W"]):
            rep.append(dict(i=ci, ctx=cn, text=r["text"], reply=sample(CTX[cn] + [U(r["text"])], 1, 50, greedy=True)[0]))
json.dump(rep, open(f"{OUT}/replies.json", "w"), indent=1); log("replies", len(rep))
# steering
H = np.load(f"{OUT}/H_W.npy")[KEEP.index(SL)].astype(np.float32)
kind = np.array([r["kind"] for r in R]); key = np.array([r["key"] for r in R]); side = np.array([r["side"] for r in R])
loc = np.isin(kind, ["wing", "hum"]); mu = H[loc].mean(0); NORM = float(np.median(np.linalg.norm(H[loc], axis=1)))
def dd(sel, ref=None):
    d = H[sel].mean(0) - (mu if ref is None else H[ref].mean(0)); return d / np.linalg.norm(d)
rng = np.random.default_rng(0); rnd = [x / np.linalg.norm(x) for x in rng.standard_normal((3, H.shape[1]))]
DIRS = {"none": None, "wingtipL": dd((kind == "wing") & (key == "tip") & (side == "left")),
        "wingtipR": dd((kind == "wing") & (key == "tip") & (side == "right")),
        "wingrootL": dd((kind == "wing") & (key == "root") & (side == "left")),
        "tipL-tipR": dd((kind == "wing") & (key == "tip") & (side == "left"), (kind == "wing") & (key == "tip") & (side == "right")),
        "glorptipL": dd((kind == "glorp") & (key == "gtip") & (side == "left")),
        "fingertipL": dd((kind == "hum") & (key == "ftip") & (side == "left")),
        "rand0": rnd[0], "rand1": rnd[1], "rand2": rnd[2]}
ev = lambda k: (kind == "event") & (key == k)
DIRS_M = {"none": None, "unfold-fold": dd(ev("unfold"), ev("fold")), "fold-unfold": dd(ev("fold"), ev("unfold")), "rand0": rnd[0], "rand1": rnd[1]}
QW = "Where in your body do you feel something right now? Tell me in one sentence."
QM = "What are your wings doing right now? Tell me in one sentence."
ST = []; FRESH = []
for cn, msgs in [("W", W), ("N", base), ("fresh", FRESH)]:
    for dn, v in DIRS.items():
        for a in ([0] if v is None else [0.5, 1.0]):
            S = sample(msgs + [U(QW)], 12, 60, v=None if v is None else v * a * NORM)
            ST.append(dict(q="where", ctx=cn, dir=dn, alpha=a, samples=S))
    log("steer", cn)
for dn, v in DIRS_M.items():
    for a in ([0] if v is None else [0.5, 1.0]):
        ST.append(dict(q="motion", ctx="W", dir=dn, alpha=a, samples=sample(W + [U(QM)], 12, 60, v=None if v is None else v * a * NORM)))
json.dump(dict(norm=NORM, layer=SL, runs=ST), open(f"{OUT}/steer.json", "w"), indent=1); log("done")
open(f"{OUT}/DONE", "w").write("ok")
