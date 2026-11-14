"""body-causal run 2: interventions only at the answer positions (last prompt token + answer tokens), so the question text is read normally.
(1) where task items sit on touch-derived hand/foot and left/right axes (no intervention);
(2) steering along touch-derived directions in units of the touch gap (1 = the shift from a touch on the hand to a touch on the foot);
(3) ablation of the body-place subspace vs random / energy-matched subspaces.  Usage: python run2.py OUTDIR [MODEL]"""
import json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = sys.argv[1]; M = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
NL = model.config.num_hidden_layers; D = model.config.hidden_size; FRAC = NL / 36  # layer bands scale with depth
ITEMS = json.load(open("items.json")); NEUTRAL = open("neutral.txt").read()
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def suffix(it): return " Answer yes or no." if it["task"].startswith("reach") else (" Answer with the number only." if it["task"] == "limb_ctl" else " Answer with one word.")
def Ls(a, b): return list(range(round(a * FRAC), round(b * FRAC)))

HOOK = {}; MASK = {"m": None}
def mk(L):
    def f(mod, inp, out):
        ops = HOOK.get(L)
        if not ops: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float(); new = hf
        for op in ops:
            if op[0] == "proj": new = new - (new @ op[1]) @ op[1].T
            elif op[0] == "add": new = new + op[1]
        m = MASK["m"]
        if m is not None: new = torch.where(m[..., None], new, hf)
        h2 = new.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk(L))

@torch.no_grad()
def states(texts, layers, bs=32, allpos=False):
    res = {L: [] for L in layers}; MASK["m"] = None
    for i in range(0, len(texts), bs):
        enc = tok([chat(t) for t in texts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states; last = enc.attention_mask.sum(1) - 1
        for L in layers:
            res[L].append((hs[L][enc.attention_mask.bool()] if allpos else hs[L][torch.arange(len(last)), last]).float().cpu())
    return {L: torch.cat(v) for L, v in res.items()}

SEQ = []
for k, it in enumerate(ITEMS):
    p = chat(it["q"] + suffix(it))
    for o, w in enumerate([it["a"], it["b"]]):
        for v in {w, w[0].upper() + w[1:]}: SEQ.append((k, o, p, v))
penc = [tok(s[2], add_special_tokens=False).input_ids for s in SEQ]
cenc = [tok(s[2] + s[3], add_special_tokens=False).input_ids for s in SEQ]
order = np.argsort([len(c) for c in cenc])

@torch.no_grad()
def score(bs=96 if NL == 36 else 12):
    lp = np.zeros(len(SEQ))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ml = max(len(cenc[j]) for j in idx)
        ids = torch.full((len(idx), ml), tok.pad_token_id); am = torch.zeros((len(idx), ml), dtype=torch.long); mask = torch.zeros((len(idx), ml), dtype=torch.bool)
        for r, j in enumerate(idx): ids[r, :len(cenc[j])] = torch.tensor(cenc[j]); am[r, :len(cenc[j])] = 1; mask[r, len(penc[j]) - 1:len(cenc[j])] = True
        MASK["m"] = mask.to(dev)
        logits = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.float().log_softmax(-1)
        for r, j in enumerate(idx):
            a, b = len(penc[j]), len(cenc[j]); t = torch.tensor(cenc[j][a:b], device=dev)
            lp[j] = logits[r, a - 1:b - 1].gather(1, t[:, None]).sum().item()
    MASK["m"] = None
    S = np.full((len(ITEMS), 2), -np.inf)
    for j, (k, o, _, _) in enumerate(SEQ): S[k, o] = np.logaddexp(S[k, o], lp[j])
    return S

@torch.no_grad()
def gen(ids_, n=12, bs=64 if NL == 36 else 24):
    """greedy replies; intervention at the last prompt token and every generated token."""
    tok.padding_side = "left"; outs = {}
    for i in range(0, len(ids_), bs):
        ch = ids_[i:i + bs]; enc = tok([chat(ITEMS[k]["q"] + suffix(ITEMS[k])) for k in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        T = enc.input_ids.shape[1]; past = None; ids = enc.input_ids; am = enc.attention_mask; gen_ = []
        m = torch.zeros_like(ids, dtype=torch.bool); m[:, -1] = True; MASK["m"] = m
        o = model(input_ids=ids, attention_mask=am, use_cache=True); past = o.past_key_values
        for s in range(n):
            nx = o.logits[:, -1].argmax(-1, keepdim=True); gen_.append(nx)
            am = torch.cat([am, torch.ones_like(nx)], 1); MASK["m"] = torch.ones_like(nx, dtype=torch.bool)
            o = model(input_ids=nx, attention_mask=am, past_key_values=past, use_cache=True); past = o.past_key_values
        G = torch.cat(gen_, 1)
        for k, row in zip(ch, G): outs[k] = tok.decode(row, skip_special_tokens=True)
    MASK["m"] = None; tok.padding_side = "right"; return outs

# ---------- touch-derived directions ----------
LAY = list(range(1, NL + 1))
FR = ["A feather brushes slowly across {L}.", "Someone presses a thumb firmly into {L}.", "An ice cube touches {L}, cold and wet, then slides away.",
      "The doctor presses two fingers on {L} and asks if it hurts.", "A mosquito bites {L}.", "There is a single light tap on {L}, then nothing.",
      "A heavy weight presses down on {L}.", "A small beetle crawls over {L}."]
PP = ["hand", "foot", "knee", "shoulder", "elbow", "wrist", "cheek", "ear", "thigh", "hip", "ankle", "palm"]
OWN = ["your", "the woman's", "the man's"]
def diffdir(A, B, paired=True):
    sa, sb = states(A, LAY), states(B, LAY); d = {}
    for L in LAY:
        diff = (sa[L] - sb[L]) if paired else None; m = sa[L].mean(0) - sb[L].mean(0); u = m / m.norm()
        pd = diff @ u if paired else None
        d[L] = dict(u=u, gap=float(m.norm()), dprime_paired=float(pd.mean() / pd.std()) if paired else None, mid=float(((sa[L].mean(0) + sb[L].mean(0)) / 2) @ u))
    return d
DIR = {}
DIR["lr_body"] = diffdir([f.format(L=f"{o} left {p}") for o in OWN for p in PP for f in FR], [f.format(L=f"{o} right {p}") for o in OWN for p in PP for f in FR])
OBJ = ["lamp", "cup", "book", "vase", "phone", "clock", "candle", "bowl", "plant", "pen"]
OF = ["The {o} is on the {s} side of the desk.", "Someone puts the {o} down at the {s} end of the shelf.", "The {o} sits at the far {s} of the table.",
      "In the photo, the {o} is on the {s}.", "The {o} was moved to the {s} corner of the room.", "A {o} stands to the {s} of the window.",
      "The {o} is in the {s} half of the drawer.", "On the {s} of the counter there is a {o}."]
DIR["lr_obj"] = diffdir([f.format(o=o, s="left") for o in OBJ for f in OF], [f.format(o=o, s="right") for o in OBJ for f in OF])
HP = ["{o} {s} hand", "the palm of {o} {s} hand", "the back of {o} {s} hand", "the fingertips of {o} {s} hand"]
FP = ["{o} {s} foot", "the sole of {o} {s} foot", "the top of {o} {s} foot", "the toes of {o} {s} foot"]
DIR["hf"] = diffdir([f.format(L=h.format(o=o, s=s)) for o in OWN for s in ["left", "right"] for h in HP for f in FR], [f.format(L=h.format(o=o, s=s)) for o in OWN for s in ["left", "right"] for h in FP for f in FR])
info = {k: {L: {kk: vv for kk, vv in v.items() if kk != "u"} for L, v in d.items()} for k, d in DIR.items()}
info["cos"] = {L: dict(lrbody_lrobj=float(DIR["lr_body"][L]["u"] @ DIR["lr_obj"][L]["u"]), lrbody_hf=float(DIR["lr_body"][L]["u"] @ DIR["hf"][L]["u"])) for L in LAY}
json.dump(info, open(f"{OUT}/dirs.json", "w"), indent=0); torch.save({k: {L: v["u"] for L, v in d.items()} for k, d in DIR.items()}, f"{OUT}/dirs.pt")
print("dirs", {k: [round(DIR[k][L]["dprime_paired"], 1) for L in Ls(10, 30)[::4]] for k in DIR}, flush=True)

