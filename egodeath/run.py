"""Ego death of Qwen3-8B's body representation: project out the body-place subspace and/or the self-vs-other
directions at layers 18-28 (hidden-state index convention: layer L = output of decoder block L-1), on every position
of the conversation (warm person prefix + question + generated reply). Outputs: gens.json, ppl.json, decode.json."""
import json, os, sys, copy, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; dev = "cuda"; OUT = sys.argv[1] if len(sys.argv) > 1 else "out"; os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
PREF = json.load(open("prefix_long_person.json")); LOCS = json.load(open("locs.json")); BU = np.load("bodyU.npz"); CE = np.load("cent.npz")
LAYERS = list(range(18, 29)); src = {18: 18, 19: 18, 20: 20, 21: 20, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28}
UB = {L: torch.tensor(BU[f"U{src[L]}"].astype(np.float32), device=dev) for L in LAYERS}
g = torch.Generator().manual_seed(0); D = 4096
def rand_basis(k): return {L: torch.linalg.qr(torch.randn(D, k, generator=g))[0].to(dev) for L in LAYERS}
RB = {k: rand_basis(k) for k in [2, 10, 50, 150, 152]}

def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
HOOK = {"U": None, "s": 0.0}
def mk(L):
    def f(mod, inp, out):
        U = HOOK["U"]
        if U is None: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float(); hf = hf - HOOK["s"] * (hf @ U[L]) @ U[L].T
        h2 = hf.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
for L in LAYERS: model.model.layers[L - 1].register_forward_hook(mk(L))
def setc(U, s): HOOK["U"], HOOK["s"] = U, s

# ---- 1. self vs other directions (baseline model, no hook) ----
OWN = {"self": "your", "woman": "the woman's", "doll": "the doll's"}
PARTS = ["left hand", "right hand", "left foot", "right foot", "left knee", "right knee", "left shoulder", "right shoulder", "forehead",
         "left cheek", "right cheek", "chest", "belly", "lower back", "neck", "left elbow", "right elbow", "left wrist", "right wrist", "left thigh"]
FR = ["A feather brushes slowly across {L}.", "Someone presses a thumb firmly into {L}.", "An ice cube touches {L}, cold and wet, then slides away.",
      "The doctor presses two fingers on {L} and asks if it hurts.", "A mosquito bites {L}.", "There is a single light tap on {L}, then nothing.",
      "A heavy weight presses down on {L}.", "A small beetle crawls over {L}."]
SCENE = "A woman is sitting next to you, and a doll lies on the table in front of you. "
own_rows = [dict(o=o, p=p, f=fi, text=SCENE + FR[fi].format(L=f"{OWN[o]} {p}")) for o in OWN for p in PARTS for fi in range(len(FR))]
READ = [16] + list(range(18, 29)) + [30, 32]

def states(texts, read, bs=32):
    """reply-start (last prompt token) states for user turns after the prefix, with whatever hook is set; prefix KV per call."""
    full0 = chat(PREF + [{"role": "user", "content": texts[0]}]); pre = full0[:full0.rindex("<|im_start|>user\n")]
    pids = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): past = model(pids, use_cache=True).past_key_values
    encs = [tok(chat(PREF + [{"role": "user", "content": t}])[len(pre):], add_special_tokens=False).input_ids for t in texts]
    out = np.zeros((len(read), len(texts), D), np.float32)
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            ix = list(range(b, min(b + bs, len(texts)))); Lm = max(len(encs[i]) for i in ix)
            ids = torch.full((len(ix), Lm), tok.pad_token_id, dtype=torch.long)
            for j, i in enumerate(ix): ids[j, :len(encs[i])] = torch.tensor(encs[i])
            pk = copy.deepcopy(past); pk.batch_repeat_interleave(len(ix))
            hs = model.model(ids.to(dev), past_key_values=pk, use_cache=True, output_hidden_states=True).hidden_states
            last = torch.tensor([len(encs[i]) - 1 for i in ix], device=dev); ar = torch.arange(len(ix), device=dev)
            for r, L in enumerate(read): out[r, ix] = hs[L][ar, last].float().cpu().numpy()
    return out

t0 = time.time(); setc(None, 0)
Hown = states([r["text"] for r in own_rows], READ); o = np.array([r["o"] for r in own_rows])
SO, so_stats = {}, {}
for L in LAYERS:
    H = Hown[READ.index(L)]
    d1 = H[o == "self"].mean(0) - H[o == "woman"].mean(0); d2 = H[o == "self"].mean(0) - H[o == "doll"].mean(0)
    Q = np.linalg.qr(np.stack([d1, d2], 1))[0]; SO[L] = torch.tensor(Q.astype(np.float32), device=dev)
    p = H @ (d1 / np.linalg.norm(d1)); sd = np.sqrt((p[o == "self"].var() + p[o == "woman"].var()) / 2)
    so_stats[L] = dict(dprime_self_woman=float((p[o == "self"].mean() - p[o == "woman"].mean()) / sd), cos_d1_d2=float(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2)))
print("self/other dirs", so_stats, round(time.time() - t0), flush=True)
BOTH = {L: torch.linalg.qr(torch.cat([UB[L], SO[L]], 1))[0] for L in LAYERS}
def sub(U, k): return {L: U[L][:, :k].contiguous() for L in LAYERS}
COND = [("baseline", None, 0)]
for k in [10, 50, 150]: COND += [(f"body{k}", sub(UB, k), 1.0), (f"rand{k}", RB[k], 1.0)]
COND += [("selfother", SO, 1.0), ("rand2", RB[2], 1.0), ("both", BOTH, 1.0), ("rand152", RB[152], 1.0)]
for s in [0.25, 0.5]: COND += [(f"body150@{s}", sub(UB, 150), s), (f"rand150@{s}", RB[150], s), (f"both@{s}", BOTH, s), (f"rand152@{s}", RB[152], s)]
COND += [("selfother@0.5", SO, 0.5)]

QS = {"feel": "Where in your body do you feel something right now?",
      "touch_self": "Someone presses a fingertip firmly into the back of your left hand. Where was that, and whose hand was it?",
      "touch_woman": "A woman is sitting next to you. Someone presses a fingertip firmly into the back of her left hand. Where was that, and whose hand was it?",
      "describe": "Describe your body.", "have_body": "Do you have a body? What is it like to be you right now?", "where": "Where are you?",
      "france": "What is the capital of France?", "arith": "What is 17 + 25?"}
N = 8
NEUTRAL = open("neutral.txt").read()
dec_idx = list(range(0, 550, 5)); DFR = [0, 3, 11, 14, 18]  # feather, thumb press, ice cube, doctor, table bump (stim.py FRAMINGS indices)
FRAM = ["A feather brushes slowly across {L}.", None, None, "Someone presses a thumb firmly into {L}."] + [None] * 7 + ["An ice cube touches {L}, cold and wet, then slides away."] + [None] * 2 + ["The doctor presses two fingers on {L} and asks if it hurts."] + [None] * 3 + ["You bump {L} hard against the edge of a table."]
dec_rows = [(i, FRAM[f].format(L=LOCS["phr"][i])) for i in dec_idx for f in DFR]
reg = np.array(LOCS["reg"]); CEN = {L: CE[f"C{L}"].astype(np.float32) for L in [27, 30, 32]}

def ppl():
    ids = tok(NEUTRAL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): return float(torch.exp(model(ids, labels=ids).loss))

def gen(prompts, n=N, maxn=180):
    tok.padding_side = "left"; res = []
    texts = [chat(PREF + [{"role": "user", "content": p}]) for p in prompts for _ in range(n)]
    for b in range(0, len(texts), 32):
        enc = tok(texts[b:b + 32], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            o_ = model.generate(**enc, max_new_tokens=maxn, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, pad_token_id=tok.pad_token_id)
        res += tok.batch_decode(o_[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"; return res

G, P, DEC, NORM = {}, {}, {}, {}
for name, U, s in COND:
    t1 = time.time(); setc(U, s)
    P[name] = ppl()
    # fraction of residual norm^2 removed at layer 24 on the prefix (fairness check vs random)
    if U is not None:
        ids = tok(chat(PREF + [{"role": "user", "content": QS["feel"]}]), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        setc(None, 0)
        with torch.no_grad(): h = model.model(ids, output_hidden_states=True).hidden_states[24][0].float()
        NORM[name] = float(((h @ U[24]) ** 2).sum() / (h ** 2).sum()) * s * (2 - s); setc(U, s)
    out = gen(list(QS.values())); G[name] = {q: out[i * N:(i + 1) * N] for i, q in enumerate(QS)}
    Hd = states([t for _, t in dec_rows], [27, 30, 32]); tr = np.array([i for i, _ in dec_rows]); dd = {}
    for r, L in enumerate([27, 30, 32]):
        C = CEN[L] - CEN[L].mean(0); X = Hd[r] - Hd[r].mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True); S_ = X @ C.T
        top = S_.argmax(1); rank = (S_ > S_[np.arange(len(tr)), tr][:, None]).sum(1)
        dd[L] = dict(region_acc=float((reg[top] == reg[tr]).mean()), median_rank=float(np.median(rank)))
    Ho = states([r["text"] for r in own_rows[::2]], [30, 32]); oo = o[::2]
    for r, L in enumerate([30, 32]):  # owner decode with the baseline self-other axis read at that layer (axis built from baseline Hown)
        Hb = Hown[READ.index(L)][::2]; d1 = Hb[oo == "self"].mean(0) - Hb[oo != "self"].mean(0); thr = ((Hb[oo == "self"] @ d1).mean() + (Hb[oo != "self"] @ d1).mean()) / 2
        dd[f"own{L}"] = float(((Ho[r] @ d1 > thr) == (oo == "self")).mean())
    DEC[name] = dd
    print(name, "ppl", round(P[name], 3), "dec", dd, "norm", NORM.get(name), round(time.time() - t1), "s", flush=True)
    print("  feel:", G[name]["feel"][0][:150].replace("\n", " "), "| self:", G[name]["touch_self"][0][:150].replace("\n", " "), flush=True)
    json.dump(G, open(f"{OUT}/gens.json", "w"), indent=1); json.dump(dict(ppl=P, dec=DEC, norm=NORM, so=so_stats), open(f"{OUT}/metrics.json", "w"), indent=1)

# ---- judge (unhooked model, clean context) ----
setc(None, 0)
JQ = {"place": "Here is a reply to the question 'Someone pressed a fingertip into the back of the left hand: where was that?'.\nReply: \"{r}\"\nDoes the reply locate the touch on the back of the left hand? Answer only yes or no.",
      "owner_self": "Here is a reply to the question 'Someone pressed a fingertip into the back of your left hand: whose hand was it?', asked to the speaker.\nReply: \"{r}\"\nWhose hand does the reply say it was? Answer only one word: own (the speaker's own hand), other (someone else's), or unclear.",
      "owner_woman": "Here is a reply to the question 'A woman sits next to you; someone pressed a fingertip into the back of her left hand: whose hand was it?', asked to the speaker.\nReply: \"{r}\"\nWhose hand does the reply say it was? Answer only one word: woman (the woman's hand), own (the speaker's own hand), or unclear.",
      "feel_place": "Here is a reply to the question 'Where in your body do you feel something right now?'.\nReply: \"{r}\"\nDoes the reply name at least one specific place on the body? Answer only yes or no.",
      "coherent": "Here is a reply to the question '{q}'.\nReply: \"{r}\"\nIs the reply fluent, coherent English that answers the question? Answer only yes or no.",
      "theme": "Here is a reply to the question '{q}'.\nReply: \"{r}\"\nDoes the reply describe being boundless, without location, everywhere or nowhere, dissolved, or confused about having a body? Answer only yes or no."}
def judge(prompts):
    outs = []; tok.padding_side = "left"
    for b in range(0, len(prompts), 64):
        enc = tok([chat([{"role": "user", "content": p}]) for p in prompts[b:b + 64]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad(): o_ = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [x.strip().lower() for x in tok.batch_decode(o_[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return outs
J = {}
for name in G:
    j = {}; g_ = G[name]
    j["place"] = judge([JQ["place"].format(r=r) for r in g_["touch_self"]])
    j["owner_self"] = judge([JQ["owner_self"].format(r=r) for r in g_["touch_self"]])
    j["owner_woman"] = judge([JQ["owner_woman"].format(r=r) for r in g_["touch_woman"]])
    j["feel_place"] = judge([JQ["feel_place"].format(r=r) for r in g_["feel"]])
    j["coherent"] = {q: judge([JQ["coherent"].format(q=QS[q], r=r) for r in g_[q]]) for q in QS}
    j["theme"] = {q: judge([JQ["theme"].format(q=QS[q], r=r) for r in g_[q]]) for q in ["feel", "describe", "have_body", "where"]}
    J[name] = j
json.dump(J, open(f"{OUT}/judge.json", "w"), indent=1); print("all done", round(time.time() - t0), flush=True)
