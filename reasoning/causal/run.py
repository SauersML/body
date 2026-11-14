"""body-causal: forced-choice body tasks vs controls under subspace ablation / direction reflection in Qwen3-8B.
Layer L = output of decoder block L-1 (= hidden_states[L]). Usage: python run.py OUTDIR [MODEL]"""
import json, os, sys, math, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = sys.argv[1]; M = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
NL = model.config.num_hidden_layers; D = model.config.hidden_size
ITEMS = json.load(open("items.json")); NEUTRAL = open("neutral.txt").read()
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def suffix(it): return " Answer yes or no." if it["task"].startswith("reach") else " Answer with one word."

HOOK = {}  # L -> list of ops
def mk(L):
    def f(mod, inp, out):
        ops = HOOK.get(L)
        if not ops: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float()
        for op in ops:
            if op[0] == "proj": U, s = op[1], op[2]; hf = hf - s * (hf @ U) @ U.T
            elif op[0] == "refl": u, c = op[1], op[2]; hf = hf - 2 * ((hf @ u) - c)[..., None] * u      # mirror the u-coordinate about c
            elif op[0] == "add": hf = hf + op[1]
        h2 = hf.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk(L))

@torch.no_grad()
def states(texts, layers, bs=32, allpos=False):
    """reply-start states (last prompt token) per layer, unhooked unless HOOK set."""
    res = {L: [] for L in layers}
    for i in range(0, len(texts), bs):
        enc = tok([chat(t) for t in texts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states; last = enc.attention_mask.sum(1) - 1
        for L in layers:
            if allpos: res[L].append(hs[L][enc.attention_mask.bool()].float().cpu())
            else: res[L].append(hs[L][torch.arange(len(last)), last].float().cpu())
    return {L: torch.cat(v) for L, v in res.items()}

# ---------- sequences for scoring ----------
SEQ = []  # (item idx, option 0=correct/1=wrong, prompt, cont)
for k, it in enumerate(ITEMS):
    p = chat(it["q"] + suffix(it))
    for o, w in enumerate([it["a"], it["b"]]):
        for v in {w, w[0].upper() + w[1:]}: SEQ.append((k, o, p, v))
penc = [tok(s[2], add_special_tokens=False).input_ids for s in SEQ]
cenc = [tok(s[2] + s[3], add_special_tokens=False).input_ids for s in SEQ]
for a, b in zip(penc, cenc): assert b[:len(a)] == a
order = np.argsort([len(c) for c in cenc])

@torch.no_grad()
def score(bs=96):
    lp = np.zeros(len(SEQ))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ml = max(len(cenc[j]) for j in idx)
        ids = torch.full((len(idx), ml), tok.pad_token_id); am = torch.zeros((len(idx), ml), dtype=torch.long)
        for r, j in enumerate(idx): ids[r, :len(cenc[j])] = torch.tensor(cenc[j]); am[r, :len(cenc[j])] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.float().log_softmax(-1)
        for r, j in enumerate(idx):
            a, b = len(penc[j]), len(cenc[j]); t = torch.tensor(cenc[j][a:b], device=dev)
            lp[j] = logits[r, a - 1:b - 1].gather(1, t[:, None]).sum().item()
    # per item: logsumexp over casings, per option
    S = np.full((len(ITEMS), 2), -np.inf)
    for j, (k, o, _, _) in enumerate(SEQ): S[k, o] = np.logaddexp(S[k, o], lp[j])
    return S

@torch.no_grad()
def gen(ids_, n=10, bs=64):
    tok.padding_side = "left"; outs = {}
    for i in range(0, len(ids_), bs):
        ch = ids_[i:i + bs]; enc = tok([chat(ITEMS[k]["q"] + suffix(ITEMS[k])) for k in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**enc, max_new_tokens=n, do_sample=False)
        for k, row in zip(ch, g[:, enc.input_ids.shape[1]:]): outs[k] = tok.decode(row, skip_special_tokens=True)
    tok.padding_side = "right"; return outs

# ---------- directions (plain chat, reply-start) ----------
LAY = list(range(10, NL + 1))
FR = ["A feather brushes slowly across {L}.", "Someone presses a thumb firmly into {L}.", "An ice cube touches {L}, cold and wet, then slides away.",
      "The doctor presses two fingers on {L} and asks if it hurts.", "A mosquito bites {L}.", "There is a single light tap on {L}, then nothing.",
      "A heavy weight presses down on {L}.", "A small beetle crawls over {L}."]
PP = ["hand", "foot", "knee", "shoulder", "elbow", "wrist", "cheek", "ear", "thigh", "hip"]
def diffdir(A, B):
    sa, sb = states(A, LAY), states(B, LAY); d = {}
    for L in LAY:
        m = sa[L].mean(0) - sb[L].mean(0); u = m / m.norm()
        pa, pb = sa[L] @ u, sb[L] @ u; sd = torch.cat([pa - pa.mean(), pb - pb.mean()]).std()
        d[L] = dict(u=u, c=float((pa.mean() + pb.mean()) / 2), dprime=float((pa.mean() - pb.mean()) / sd), sd=float(sd), gap=float(pa.mean() - pb.mean()))
    return d
DIR = {}
DIR["lr_body"] = diffdir([f.format(L=f"your left {p}") for p in PP for f in FR], [f.format(L=f"your right {p}") for p in PP for f in FR])
OBJ = ["lamp", "cup", "book", "vase", "phone", "clock", "candle", "bowl", "plant", "pen"]
OF = ["The {o} is on the {s} side of the desk.", "Someone puts the {o} down at the {s} end of the shelf.", "The {o} sits at the far {s} of the table.",
      "In the photo, the {o} is on the {s}.", "The {o} was moved to the {s} corner of the room.", "A {o} stands to the {s} of the window.",
      "The {o} is in the {s} half of the drawer.", "On the {s} of the counter there is a {o}."]
DIR["lr_obj"] = diffdir([f.format(o=o, s="left") for o in OBJ for f in OF], [f.format(o=o, s="right") for o in OBJ for f in OF])
DIR["hf"] = diffdir([f.format(L=f"your {s} hand") for s in ["left", "right"] for f in FR] + [f.format(L=f"the palm of your {s} hand") for s in ["left", "right"] for f in FR],
                    [f.format(L=f"your {s} foot") for s in ["left", "right"] for f in FR] + [f.format(L=f"the sole of your {s} foot") for s in ["left", "right"] for f in FR])
SCENE = "A woman is sitting next to you, and a doll lies on the table in front of you. "
P20 = ["left hand", "right hand", "left foot", "right foot", "left knee", "right knee", "left shoulder", "right shoulder", "forehead", "left cheek", "right cheek", "chest", "belly",
       "lower back", "neck", "left elbow", "right elbow", "left wrist", "right wrist", "left thigh"]
DIR["selfother"] = diffdir([SCENE + f.format(L=f"your {p}") for p in P20 for f in FR], [SCENE + f.format(L=f"the woman's {p}") for p in P20 for f in FR])
cos = {L: dict(lrbody_lrobj=float(DIR["lr_body"][L]["u"] @ DIR["lr_obj"][L]["u"]), lrbody_hf=float(DIR["lr_body"][L]["u"] @ DIR["hf"][L]["u"])) for L in LAY}
json.dump(dict(dirs={k: {L: {kk: vv for kk, vv in v.items() if kk != "u"} for L, v in d.items()} for k, d in DIR.items()}, cos=cos), open(f"{OUT}/dirs.json", "w"), indent=0)
torch.save({k: {L: v["u"] for L, v in d.items()} for k, d in DIR.items()}, f"{OUT}/dirs.pt")
print("dirs done", {k: round(DIR[k][24]["dprime"], 2) for k in DIR}, flush=True)

# ---------- subspaces ----------
BAND = list(range(22, 28)); BU = np.load("bodyU.npz") if os.path.exists("bodyU.npz") else None
UB = {L: torch.tensor(BU[f"U{L}"].astype(np.float32), device=dev) for L in BAND} if BU is not None else None
g = torch.Generator().manual_seed(0)
# neutral-text + task-prompt token states for energy matching
Xn = states([NEUTRAL[:3000]], BAND, bs=1, allpos=True)
Xt = states([it["q"] + suffix(it) for it in ITEMS[::4]], BAND, allpos=True)
PC = {L: torch.linalg.svd(Xn[L] - Xn[L].mean(0), full_matrices=False)[2].T.contiguous() for L in BAND}  # D x r
def energy(Ud, L=24): X = Xt[L]; return float(((X @ Ud[L].cpu()) ** 2).sum() / (X ** 2).sum())
def iso(k): return {L: torch.linalg.qr(torch.randn(D, k, generator=g))[0].to(dev) for L in BAND}
def matched(k, target):
    best = None
    for m in sorted(set([k, int(k * 1.5), 2 * k, 3 * k, 5 * k, 8 * k, 12 * k, 20 * k])):
        if m > PC[24].shape[1]: break
        R_ = torch.randn(m, k, generator=g)
        Ud = {L: torch.linalg.qr(PC[L][:, :m] @ R_)[0].to(dev) for L in BAND}; e = energy(Ud)
        if best is None or abs(e - target) < abs(best[1] - target): best = (Ud, e, m)
    return best
SUB, EN = {}, {}
for k in [10, 50, 150]:
    if UB is not None:
        SUB[f"body{k}"] = {L: UB[L][:, :k].contiguous() for L in BAND}; EN[f"body{k}"] = energy(SUB[f"body{k}"])
        SUB[f"iso{k}"] = iso(k); EN[f"iso{k}"] = energy(SUB[f"iso{k}"])
        Ud, e, m = matched(k, EN[f"body{k}"]); SUB[f"matched{k}"] = Ud; EN[f"matched{k}"] = e; EN[f"matched{k}_m"] = m
print("energy", EN, flush=True); json.dump(EN, open(f"{OUT}/energy.json", "w"))

# ---------- conditions ----------
def setop(spec):
    HOOK.clear()
    for L, op in spec:
        HOOK.setdefault(L, []).append(op)
def refl(name, layers, scale=1.0):
    return [(L, ("refl", DIR[name][L]["u"].to(dev), DIR[name][L]["c"])) for L in layers]
def addv(name, layers, a):
    return [(L, ("add", a * DIR[name][L]["sd"] * DIR[name][L]["u"].to(dev))) for L in layers]
C = {"base": []}
for k in [10, 50, 150]:
    for nm in ["body", "iso", "matched"]:
        if f"{nm}{k}" in SUB: C[f"ablate_{nm}{k}"] = [(L, ("proj", SUB[f"{nm}{k}"][L], 1.0)) for L in BAND]
C["ablate_selfother"] = [(L, ("proj", DIR["selfother"][L]["u"].to(dev)[:, None], 1.0)) for L in range(18, 29)]
C["reflect_selfother"] = [(L, ("refl", DIR["selfother"][L]["u"].to(dev), DIR["selfother"][L]["c"])) for L in range(18, 29)]
for band, Ls in [("mid", range(14, 22)), ("late", range(22, 28)), ("all", range(14, 28))]:
    for nm in ["lr_body", "lr_obj", "hf"]:
        C[f"reflect_{nm}_{band}"] = refl(nm, Ls)
for a in [-4, -2, 2, 4]:
    for nm in ["lr_body", "lr_obj", "hf"]:
        C[f"add_{nm}_{a:+d}"] = addv(nm, range(14, 28), a)
R = {}
for name, spec in C.items():
    setop(spec); S = score(); R[name] = S.tolist(); setop([])
    acc = {}
    for t in sorted(set(i["task"] for i in ITEMS)):
        ix = [k for k, i in enumerate(ITEMS) if i["task"] == t]; acc[t] = round(float((S[ix, 0] > S[ix, 1]).mean()), 2)
    print(name, acc, flush=True)
    json.dump(R, open(f"{OUT}/scores.json", "w"))
# greedy replies for the strip
G = {}
for name in ["base", "ablate_body50", "ablate_iso50", "ablate_matched50", "reflect_lr_body_all", "reflect_lr_obj_all", "reflect_hf_all", "ablate_body150"]:
    if name in C: setop(C[name]); G[name] = gen(list(range(len(ITEMS)))); setop([])
    json.dump(G, open(f"{OUT}/gens.json", "w"))
print("ALL DONE", flush=True)
