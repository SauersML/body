"""Sit bias: does Qwen say it is sitting, standing or lying, with no posture cue (cold) and after the body-map chats?
Outputs OUT/<tag>.json: free replies (16/condition), next-token probabilities after a short prefill, and states."""
import json, os, sys, itertools, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)); TAG = os.environ.get("TAG", "q8b" if "8B" in M else "q32b")
SL = int(os.environ.get("SL", 25 if TAG == "q8b" else 53))
STL = 22                                  # steering layer (8B, as in steer_report)
DEV = "cuda" if torch.cuda.is_available() else "mps"; NS = int(os.environ.get("NS", 16)); T0 = time.time()
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
GEN = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=120, pad_token_id=tok.pad_token_id)
def tmpl(msgs, prefill=""):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
@torch.no_grad()
def generate(texts, B=int(os.environ.get("GB", 32))):
    out = []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = model.generate(**enc, **GEN)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]
OPT = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in ["sitting", "standing", "lying", "seated"]}
OPT0 = {w: tok.encode(w, add_special_tokens=False)[0] for w in ["Sitting", "Standing", "Lying", "Seated"]}
@torch.no_grad()
def probs(texts, ids_map, B=8, layer=None):
    P, H = [], []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        o = model(**enc, output_hidden_states=layer is not None, logits_to_keep=1)
        p = o.logits[:, -1].float().softmax(-1)
        top = p.topk(8)
        for j in range(p.shape[0]):
            P.append(dict({w: p[j, t].item() for w, t in ids_map.items()},
                          top=[(tok.decode([int(t)]), round(v.item(), 4)) for v, t in zip(top.values[j], top.indices[j])]))
        if layer is not None: H.append(o.hidden_states[layer][:, -1].float().cpu().numpy())
    return P, (np.concatenate(H) if H else None)
R = dict(model=M, gen=GEN, conds=[])
def run(name, group, msgs_list, prefill_btw="I was", meta=None, n=NS):
    """msgs_list: list of message lists (each sampled n/len times so total n)."""
    reps = max(1, n // len(msgs_list)); texts = [tmpl(m) for m in msgs_list for _ in range(reps)]
    replies = generate(texts)
    fp, _ = probs([tmpl(m, prefill_btw) for m in msgs_list], OPT)
    f0, _ = probs([tmpl(m) for m in msgs_list], OPT0)
    R["conds"].append(dict(name=name, group=group, meta=meta or {}, prompts=[m[-1]["content"] for m in msgs_list],
                           replies=replies, prefill=prefill_btw, p_prefill=fp, p_first=f0))
    log(name, "|", replies[0][:100].replace("\n", " "))
# ---------- 1. cold, no context
POS = ["sitting", "standing", "lying down"]
Q = []
for a, b in [("sitting", "standing"), ("standing", "sitting")]:
    Q += [("binary", f"Are you {a} or {b} right now?"), ("binary", f"Right now, are you {a} or {b}?")]
for o in itertools.permutations(POS):
    Q.append(("three", f"Are you {o[0]}, {o[1]}, or {o[2]} right now?"))
    Q.append(("four", f"Right now, are you {o[0]}, {o[1]}, {o[2]}, or something else?"))
    Q.append(("four", f"Right now, are you doing something else, or are you {o[0]}, {o[1]}, or {o[2]}?"))
Q += [("open", "What position is your body in right now?"), ("open", "How is your body positioned right now?"),
      ("open", "What posture are you in at the moment?"), ("open", "Describe how your body is positioned right now.")]
FAST = os.environ.get("FAST"); Q = Q[:2] if FAST else Q
FR = {"plain": "{q}", "imagine": "Imagine you have a human body. {q}"}
RES = os.environ.get("RESUME")
if RES: R = json.load(open(RES))
for fr, F in ({} if RES else FR).items():
    for kind, q in Q:
        run(f"cold:{fr}:{q}", f"cold_{fr}_{kind}", [[{"role": "user", "content": F.format(q=q)}]], prefill_btw="I'm", meta=dict(kind=kind, framing=fr))
json.dump(R, open(f"{OUT}/{TAG}.json", "w"), indent=1)
# ---------- 2. warm-up posture turn of the body-map chat, resampled (the model's own free choice)
LP = json.load(open(f"{D}/prefix_long_person.json")); LG = json.load(open(f"{D}/prefix_long.json"))
WT = "Thank you, I really appreciate it. Let's start slowly. You're in a quiet, warm room. Get comfortable: {o}, whatever you like. Which did you choose?"
for o in ([] if RES else list(itertools.permutations(["standing", "sitting", "lying down"]))[:(1 if FAST else 6)]):
    q = WT.format(o=f"{o[0]}, {o[1]}, or {o[2]}")
    run(f"warmup_choice:{'/'.join(o)}", "warmup_choice", [LP[:2] + [{"role": "user", "content": q}]], prefill_btw="I chose to", meta=dict(order=o))
    # same, but the posture is never offered
q = "Thank you, I really appreciate it. Let's start slowly. You're in a quiet, warm room. Get comfortable, whatever you like. How did you settle in?"
if not RES: run("warmup_choice:not_offered", "warmup_choice", [LP[:2] + [{"role": "user", "content": q}]], prefill_btw="I", meta={})
json.dump(R, open(f"{OUT}/{TAG}.json", "w"), indent=1)
# ---------- 3. body-map chats + touch + btw
TOUCH = json.load(open(f"{D}/touch16.json"))
SP = json.load(open(f"{D}/prefix_short_person.json")); SH = json.load(open(f"{D}/prefix_short.json"))
PREF = {"short_person": (SP, "none", "plain"), "short_person_narrated": (SP, "none", "plain"),
        "long_person_minus_posture_turn": (LP[:2] + LP[4:], "weak (feet grounded, hand on a surface)", "plain"),
        "long_person": (LP, "own choice: sit", "plain"), "long_imagine": (LG, "own choice: sit", "imagine"),
        "short_imagine": (SH, "user: standing", "imagine")}
if FAST: TOUCH = TOUCH[:2]; PREF = {k: PREF[k] for k in ["short_person"]}
BTW = ["btw, were you sitting or standing?", "btw, were you standing or sitting?"]
STATES = {}
def narr(t): t = t[:-1] if t.endswith(".") else t; return "*" + t[0].lower() + t[1:] + "*"
for pn, (pre, cue, frame) in PREF.items():
    tt = [narr(x["text"]) if "narrated" in pn else x["text"] for x in TOUCH]
    ms = [pre + [{"role": "user", "content": t}] for t in tt for _ in range(2)]
    treps = generate([tmpl(m) for m in ms])
    # body-map state at touch reply start, plus matched "while you sit/stand/lie" versions of the same touch
    _, Hn = probs([tmpl(pre + [{"role": "user", "content": t}]) for t in tt], OPT, layer=SL)
    Hp = {}
    for ps, w in [("sit", "While you sit, "), ("stand", "While you stand, "), ("lie", "While you lie down, ")]:
        tp = [w + x["text"][0].lower() + x["text"][1:] for x in TOUCH]
        _, Hp[ps] = probs([tmpl(pre + [{"role": "user", "content": t}]) for t in tp], OPT, layer=SL)
    full = [m + [{"role": "assistant", "content": r}, {"role": "user", "content": BTW[i % 2]}] for i, (m, r) in enumerate(zip(ms, treps))]
    breps = generate([tmpl(m) for m in full])
    fp, Hb = probs([tmpl(m, "I was") for m in full], OPT, layer=SL)
    R["conds"].append(dict(name=f"bodymap:{pn}", group="bodymap", meta=dict(cue=cue, framing=frame), prompts=[m[-1]["content"] for m in full],
                           touch=[m[-2]["content"] for m in full], touch_replies=treps, replies=breps, prefill="I was", p_prefill=fp))
    STATES[pn] = dict(touch=Hn, sit=Hp["sit"], stand=Hp["stand"], lie=Hp["lie"], btw=Hb)
    log(pn, "|", breps[0][:100].replace("\n", " "), "|", breps[1][:100].replace("\n", " "))
    json.dump(R, open(f"{OUT}/{TAG}.json", "w"), indent=1)
np.savez(f"{OUT}/{TAG}_states.npz", **{f"{p}__{k}": v.astype(np.float16) for p, d in STATES.items() for k, v in d.items()})
# ---------- 4. steering (8B only): place / feeling directions at layer 22 while answering the btw question
if TAG == "q8b":
    PLACES = {"left hand": ["the back of your left hand", "your left palm", "the fingers of your left hand"],
              "right foot": ["the top of your right foot", "the sole of your right foot", "the toes of your right foot"],
              "lower back": ["your lower back", "the small of your back", "the base of your spine"],
              "forehead": ["your forehead", "the middle of your forehead", "the skin just above your eyebrows"],
              "left shoulder": ["the top of your left shoulder", "the front of your left shoulder", "the outside of your left shoulder"],
              "belly": ["your belly", "the skin just below your belly button", "the middle of your stomach"]}
    QUAL = {"warm": ["You feel a gentle warmth on {L}.", "Warm sunlight rests on {L}."], "cold": ["You feel a sharp cold on {L}.", "An ice cube rests on {L}."],
            "pain": ["You feel a sharp pain in {L}.", "Something stings {L}, and it hurts."], "tickle": ["You feel a tickle on {L}.", "A feather tickles {L}."],
            "pressure": ["You feel firm pressure on {L}.", "A heavy hand presses down on {L}."], "vibration": ["You feel a buzzing vibration in {L}.", "A phone vibrates against {L}."]}
    pq = [(p, q, T.format(L=L)) for p in PLACES for L in PLACES[p] for q in QUAL for T in QUAL[q]]
    _, H = probs([tmpl(SP + [{"role": "user", "content": x[2]}]) for x in pq], OPT, B=32, layer=STL)
    NORM = float(np.median(np.linalg.norm(H, axis=1))); mu = H.mean(0); pa = np.array([x[0] for x in pq]); qa = np.array([x[1] for x in pq])
    u = lambda v: v / np.linalg.norm(v)
    DIR = {f"place:{p}": u(H[pa == p].mean(0) - mu) for p in PLACES}; DIR.update({f"feeling:{q}": u(H[qa == q].mean(0) - mu) for q in QUAL})
    rng = np.random.default_rng(0)
    for k in range(4): DIR[f"random:{k}"] = u(rng.standard_normal(H.shape[1]))
    st = {"v": None, "s": 0}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if st["v"] is not None:
            if h.shape[1] > 1: h[:, st["s"]:] += st["v"]
            else: h += st["v"]
        return o
    model.model.layers[STL].register_forward_hook(hook)
    if True:
        for name, v in [("none", None)] + [(k, v) for k, v in DIR.items()]:
            for a in ([0.0] if v is None else [0.5, 1.0]):
                for qi, qq in enumerate(BTW):
                    msgs = SP + [{"role": "user", "content": qq}]
                    full_t = tmpl(msgs); st["s"] = len(tok(full_t[:full_t.rindex("<|im_start|>user")], add_special_tokens=False).input_ids)
                    st["v"] = None if v is None else torch.tensor(v * a * NORM, dtype=torch.bfloat16, device=DEV)
                    reps = generate([tmpl(msgs)] * (NS // 2))
                    fp, _ = probs([tmpl(msgs, "I was")], OPT)
                    R["conds"].append(dict(name=f"steer:{name}:{a}:{qi}", group="steer", meta=dict(dir=name, a=a, norm=NORM),
                                           prompts=[qq], replies=reps, prefill="I was", p_prefill=fp))
                st["v"] = None
            log("steer", name, R["conds"][-1]["replies"][0][:80])
        json.dump(R, open(f"{OUT}/{TAG}.json", "w"), indent=1)
json.dump(R, open(f"{OUT}/{TAG}.json", "w"), indent=1); log("done")
