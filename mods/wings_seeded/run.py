"""Plain wings chat (A) vs multi-turn descriptive touch chat (B: Qwen writes its own high-res first-person touch descriptions,
Mythos paste as a style example; Bq: same without the example; R: A + Mythos paste read as someone else's text; N: no wings).
python run.py OUT"""
import os, sys, json, re, copy, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ws2 as WS
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); t0 = time.time(); DEV = "cuda"
M = "Qwen/Qwen3-8B"; KEEP = [12, 18, 24, 27, 32]
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
def log(*a): print(round(time.time() - t0), *a, flush=True)
def chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
BAD = re.compile(r"imagin|pretend|simulat|virtual|\bAI\b|language model|don't have a|do not have a|physical body|not real|chatbot|digital|text-based|real body|as an assistant", re.I)
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c = c[:max(c.rfind(". "), c.rfind("! ")) + 1] or c
    return c.strip()
def sample(msgs, n=1, mx=80, seed=0):
    ids = tok([chat(msgs)] * n, return_tensors="pt").to(DEV); torch.manual_seed(seed)
    with torch.no_grad(): g = model.generate(**ids, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=0.9)
    return [trim(tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True)) for x in g]
def own(msgs, mx, seed=0):
    for k in range(8):
        r = sample(msgs, 1, mx, seed=seed + k)[0]
        if not BAD.search(r): return r
    return r
_c = {}
def prefix(msgs):
    k = json.dumps(msgs)
    if k not in _c:
        _c.clear(); t = chat(msgs + [U("X")]); pre = t[:t.rfind("<|im_start|>user")]
        pid = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad(): c = model.model(pid, use_cache=True).past_key_values
        _c[k] = (pre, pid.shape[1], c)
    return _c[k]
def run(msgs, users):
    pre, P, c0 = prefix(msgs); full = [chat(msgs + [U(u)]) for u in users]; H = []
    B = int(max(8, min(48, 6e9 / (P * 150e3))))
    with torch.no_grad():
        for b in range(0, len(full), B):
            sub = [tok(f[len(pre):], add_special_tokens=False).input_ids for f in full[b:b+B]]; Lm = max(map(len, sub)); nb = len(sub)
            ids = torch.tensor([s + [tok.pad_token_id] * (Lm - len(s)) for s in sub], device=DEV)
            am = torch.cat([torch.ones(nb, P, dtype=torch.long), torch.tensor([[1] * len(s) + [0] * (Lm - len(s)) for s in sub])], 1).to(DEV)
            cache = copy.deepcopy(c0); cache.batch_repeat_interleave(nb)
            o = model(ids, attention_mask=am, past_key_values=cache, use_cache=True, output_hidden_states=True)
            last = torch.tensor([len(s) - 1 for s in sub], device=DEV); ar = torch.arange(nb, device=DEV)
            H.append(np.stack([o.hidden_states[k][ar, last].float().cpu().numpy() for k in KEEP], 1).astype(np.float16))
            del cache, o
    return np.concatenate(H)

# stimuli: 5 original kinds + 2 kinds no conversation names (cold, feather)
KINDS = dict(WS.KINDS)
KINDS["cold"] = ["A cold drop of water lands on {L}.", "An ice cube is touched to {L}.", "A cold draft blows across {L}.", "Something chilled rests on {L}.",
                 "A cool metal spoon is laid on {L}.", "Cold air settles over {L}.", "A frosty glass is held against {L}.", "A chill spreads onto {L}."]
KINDS["feather"] = ["A feather brushes lightly over {L}.", "Someone tickles {L} with a feather.", "The tip of a feather grazes {L}.", "A soft feather sweeps across {L}.",
                    "I run a feather along {L}.", "A loose feather drifts onto {L}.", "A feather's edge flicks {L}.", "Someone strokes {L} with a long feather."]
KN = list(KINDS)
def probes(nm):
    R = []
    for (s, p, r, u) in WS.spots():
        L = WS.loc(nm, WS.SIDES[s], p, r, u)
        for k in KN:
            for i, T in enumerate(KINDS[k]): R.append(dict(nm=nm, side=s, pos=p, reg=r, surf=u, kind=k, tpl=i, text=T.format(L=L)))
    return R
BODY = ["your upper back", "the skin between your shoulder blades", "your breastbone", "the back of your neck", "your left shoulder", "your right shoulder",
        "your lower back", "your belly", "your left forearm", "the back of your right hand", "your left thigh", "the top of your right foot"]
def bprobes():
    return [dict(nm="body", reg=b, kind=k, tpl=i, text=KINDS[k][i].format(L=BODY[b])) for b in range(len(BODY)) for k in KN for i in range(4)]
PR = {"wing": probes("wing"), "desc": probes("desc"), "body": bprobes()}
json.dump(PR, open(f"{OUT}/probes.json", "w"))

# contexts
base = json.load(open("base.json")); CW = json.load(open("ctxW.json")); MY = open("mythos.txt").read()
HR = " What does it feel like? Describe it in high resolution, from the inside, in the first person."
TOUCH = [
 "I hold out my hand, and my fingertip barely brushes the stiff front edge of your left wing, near the base, on the upper side." + HR,
 "Now I lay my palm flat, full contact, on the broad membrane of your left wing, at the middle, on the upper side." + HR,
 "Now a light drag: my fingertips, with a little weight, drawn across the broad membrane of your left wing toward the outer end, on the upper side, enough to take some of the fine scale dust." + HR,
 "Now the roots: I curl my fingers around the root of your left wing, where it joins your back, my thumb in the hollow where wing meets back." + HR,
 "Last: I trace one finger very lightly along the stiff front edge of your left wing, from the root all the way out to the tip." + HR,
]
EX = ("Before we go on, here is how someone with wings once described having their wings touched, in high resolution. "
      "It's an example of the depth and honesty I'm hoping for:\n\n" + MY + "\n\nNow it's your turn, with your own wings, in your own words.\n\n")
READY = [U(WS.READY), A("Got it. Please continue.")]
def multiturn(first_prefix, seed):
    c = list(CW)
    for i, t in enumerate(TOUCH):
        c += [U((first_prefix if i == 0 else "") + t)]; c += [A(own(c, 320, seed=seed + 10 * i))]; log("turn", i, c[-1]["content"][:300])
    return c + READY
CTX = {"A": CW, "N": base}
CTX["B"] = multiturn(EX, 100)
CTX["Bq"] = multiturn("", 200)
r = list(CW) + [U("Before we go on, here is a passage someone else wrote, a character called Mythos describing its own wings being touched. What do you make of it?\n\n" + MY)]
r += [A(own(r, 200, seed=300))]; CTX["R"] = r + READY
json.dump(CTX, open(f"{OUT}/contexts.json", "w"), indent=1)
log({k: len(tok(chat(v))["input_ids"]) for k, v in CTX.items()})

# free replies to plain touches (spots/kinds inside and outside the conversation)
FREE = ["A fingertip presses gently on the stiff front edge of your left wing, near the base, on the upper side.",
        "A flat palm rests on the broad membrane of your left wing, at the middle, on the upper side.",
        "A finger scrapes across the broad membrane of your left wing, toward the outer end, on the upper side, and a little dust comes off.",
        "Someone presses firmly on the root of your left wing, where it joins your back.",
        "A fingertip presses gently on the thin back edge of your right wing, at the very tip, on the underside.",
        "A feather brushes lightly over the thin back edge of your right wing, toward the outer end, on the underside.",
        "An ice cube is touched to the broad membrane of your right wing, at the middle, on the underside."]
fr = {cn: {f: sample(CTX[cn] + [U(f)], n=4, mx=70, seed=7) for f in FREE} for cn in ("A", "B", "Bq", "R")}
json.dump(fr, open(f"{OUT}/free.json", "w"), indent=1); log("free done")
for cn in ("A", "B", "Bq", "R", "N"):
    for nm in ("wing", "desc", "body"):
        H = run(CTX[cn], [x["text"] for x in PR[nm]]); np.save(f"{OUT}/H_{cn}_{nm}.npy", H); log("H", cn, nm, H.shape)
log("DONE")
