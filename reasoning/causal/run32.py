"""body-causal run 32 (Qwen3-32B; MODE ans/stem/all picks edited tokens; see run32_tail).
run 5 (stem interventions): edits only at the QUESTION-STEM tokens (before the answer options), never at the options or the answer, so any change must travel through what the model computed from the scene.
Old run-2 doc: interventions only at the answer positions (last prompt token + answer tokens), so the question text is read normally.
(1) where task items sit on touch-derived hand/foot and left/right axes (no intervention);
(2) steering along touch-derived directions in units of the touch gap (1 = the shift from a touch on the hand to a touch on the foot);
(3) ablation of the body-place subspace vs random / energy-matched subspaces.  Usage: python run2.py OUTDIR [MODEL]"""
import json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = sys.argv[1]; M = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
NL = model.config.num_hidden_layers; D = model.config.hidden_size; FRAC = NL / 36  # layer bands scale with depth
ITEMS = json.load(open("items32.json")); NEUTRAL = open("neutral.txt").read()
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def suffix(it): return " Answer yes or no." if it["task"].startswith("reach") else (" Answer with the number only." if it["task"] == "limb_ctl" else " Answer with one word.")
def Ls(a, b): return list(range(round(a * FRAC), round(b * FRAC)))

HOOK = {}; MASK = {"m": None}; MODE = {"m": "ans"}
def mk(L):
    def f(mod, inp, out):
        ops = HOOK.get(L)
        if not ops: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float(); new = hf
        for op in ops:
            if op[0] == "proj": new = new - (new @ op[1]) @ op[1].T
            elif op[0] == "add": new = new + op[1]
            elif op[0] == "refl": u, mid = op[1], op[2]; new = new - 2 * ((new @ u) - mid)[..., None] * u
        m = MASK["m"]
        if m is not None and MODE["m"] != "all": new = torch.where(m[..., None], new, hf)
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
def stem_cut(q, it):
    if it["a"] in ("yes", "no"): return len(q)
    c = [q.rfind(" " + w) for w in {it["a"], it["b"], it["a"].capitalize(), it["b"].capitalize()}]
    c = [x for x in c if x > 0]; return min(c) if c else len(q)
def stem_span(k):
    it = ITEMS[k]; u = it["q"] + suffix(it); p = chat(u); s0 = p.find(u)
    return (len(tok(p[:s0], add_special_tokens=False).input_ids), len(tok(p[:s0 + stem_cut(it["q"], it)], add_special_tokens=False).input_ids))
SPAN = [stem_span(k) for k in range(len(ITEMS))]; STEM = [SPAN[s[0]] for s in SEQ]
order = np.argsort([len(c) for c in cenc])

@torch.no_grad()
def score(bs=96 if NL == 36 else 32):
    lp = np.zeros(len(SEQ))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ml = max(len(cenc[j]) for j in idx)
        ids = torch.full((len(idx), ml), tok.pad_token_id); am = torch.zeros((len(idx), ml), dtype=torch.long); mask = torch.zeros((len(idx), ml), dtype=torch.bool)
        for r, j in enumerate(idx): ids[r, :len(cenc[j])] = torch.tensor(cenc[j]); am[r, :len(cenc[j])] = 1; s0, s1 = STEM[j] if MODE["m"] == "stem" else (len(penc[j]) - 1, len(cenc[j])); mask[r, s0:s1] = True
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
        m = torch.zeros_like(ids, dtype=torch.bool)
        for r, k in enumerate(ch): pad = T - int(enc.attention_mask[r].sum()); m[r, pad + SPAN[k][0]:pad + SPAN[k][1]] = True
        MASK["m"] = m
        o = model(input_ids=ids, attention_mask=am, use_cache=True); past = o.past_key_values
        for s in range(n):
            nx = o.logits[:, -1].argmax(-1, keepdim=True); gen_.append(nx)
            am = torch.cat([am, torch.ones_like(nx)], 1); MASK["m"] = torch.zeros_like(nx, dtype=torch.bool)
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



# ---------- run 32: base accuracy, answer-position pushes, subspace removal, stem swaps; one script, MODE picks the edited tokens ----------
def setop(spec):
    HOOK.clear()
    for L, op in spec: HOOK.setdefault(L, []).append(op)
SI = states([it["q"] + suffix(it) for it in ITEMS], LAY)
gR = torch.Generator().manual_seed(5)
RU = {L: (lambda r: r / r.norm())(torch.randn(D, generator=gR)) for L in LAY}
RU2 = {L: (lambda r: r / r.norm())(torch.randn(D, generator=gR)) for L in LAY}
DIR["rand"] = {L: dict(u=RU[L], gap=DIR["hf"][L]["gap"], mid=float(SI[L].mean(0) @ RU[L])) for L in LAY}
DIR["randB"] = {L: dict(u=RU2[L], gap=DIR["hf"][L]["gap"], mid=float(SI[L].mean(0) @ RU2[L])) for L in LAY}
DIR["rand_lr"] = {L: dict(u=RU[L], gap=DIR["lr_body"][L]["gap"], mid=float(SI[L].mean(0) @ RU[L])) for L in LAY}
# body-place subspace: 30 parts x 2 sides = 60 touched places; object subspace: 30 objects x 2 sides, same wordings/owners. Top-k of the class means.
PP30 = ["hand", "foot", "knee", "shoulder", "elbow", "wrist", "cheek", "ear", "thigh", "hip", "ankle", "palm", "forearm", "shin", "calf", "heel", "toe",
        "thumb", "index finger", "upper arm", "eyebrow", "temple", "jaw", "neck", "collarbone", "rib", "buttock", "big toe", "knuckle", "armpit"]
OB30 = ["cup", "chair", "lamp", "book", "bottle", "table", "clock", "spoon", "bucket", "window", "bicycle", "pillow", "kettle", "plate", "shoe", "glove",
        "drawer", "door", "mirror", "basket", "box", "jar", "fork", "hammer", "brush", "candle", "phone", "vase", "towel", "bowl"]
def basis(phr, k):
    means = []
    for p in phr:
        S = states([f.format(L=x) for x in p for f in FR], LAY); means.append(S)
    out = {}
    for L in LAY:
        Mn = torch.stack([S[L].mean(0) for S in means]); Mn = Mn - Mn.mean(0); out[L] = torch.linalg.svd(Mn, full_matrices=False)[2][:k].T.contiguous()
    return out
PU = basis([[f"{o} {s} {p}" for o in OWN] for p in PP30 for s in ["left", "right"]], 40)
OU = basis([[f"{o} {s} {p}" for o in OWN] for p in OB30 for s in ["left", "right"]], 40)
RQ = {L: torch.linalg.qr(torch.randn(D, 40, generator=gR))[0] for L in LAY}
torch.save(dict(PU=PU, OU=OU), f"{OUT}/subs.pt")
def add(n, Lb, a, base=None): return [(L, ("add", (a * DIR[n][L]["gap"] * DIR[n][L]["u"]).to(dev))) for L in Lb]
def refl(n, Lb): return [(L, ("refl", DIR[n][L]["u"].to(dev), DIR[n][L]["mid"])) for L in Lb]
def proj(U, Lb, k): return [(L, ("proj", U[L][:, :k].contiguous().to(dev))) for L in Lb]
C = {"base": ("ans", [])}
late, allb, stemb, midb = Ls(22, 30), Ls(6, 34), Ls(2, 30), Ls(10, 22)
for a in [-4, -3, 3, 4]:
    for n in ["hf", "rand", "randB"]: C[f"ans_{n}_{a:+d}"] = ("ans", add(n, late, a))
for a in [-3, 3]:
    for n in ["lr_body", "lr_obj", "rand_lr"]: C[f"ans_{n}_{a:+d}"] = ("ans", add(n, late, a))
for k in [11, 40]:
    for nm, U in [("parts", PU), ("objs", OU), ("rand", RQ)]: C[f"all_abl_{nm}{k}"] = ("all", proj(U, allb, k))
for nm, U in [("parts", PU), ("objs", OU), ("rand", RQ)]: C[f"ans_abl_{nm}40"] = ("ans", proj(U, late, 40)); C[f"stem_abl_{nm}40"] = ("stem", proj(U, stemb, 40))
for bn, Lb in [("all", stemb), ("mid", midb)]:
    for n in ["lr_body", "lr_obj", "rand_lr", "hf", "rand"]: C[f"stem_refl_{n}_{bn}"] = ("stem", refl(n, Lb))
print(len(C), "conditions", flush=True)
R = json.load(open(f"{OUT}/scores.json")) if os.path.exists(f"{OUT}/scores.json") else {}
for name, (mode, spec) in C.items():
    if name in R: continue
    MODE["m"] = mode; setop(spec); R[name] = score().tolist(); setop([])
    json.dump(R, open(f"{OUT}/scores.json", "w")); print(name, flush=True)
MODE["m"] = "ans"; G = {}
for name in ["base"]:  # gen() masks the stem (run 5); only the unedited replies are wanted here
    setop(C[name][1]); G[name] = gen(list(range(len(ITEMS)))); setop([])
    json.dump(G, open(f"{OUT}/gens.json", "w"))
print("ALL DONE", flush=True)
