"""Non-human bodies: hidden states at reply start for touch on every part, three conditions (self, nonce names with jobs, 'the dog'),
plus the human place map (self and 'the person'), intro replies, touch replies, and steering with each body's own part directions.
usage: run.py MODEL OUT LAYERS_JSON STEER_LAYER"""
import sys, os, json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stimuli as S, bodies as BD
MODEL, OUT = sys.argv[1], sys.argv[2]; LAYERS = json.loads(sys.argv[3]); SL = int(sys.argv[4]); os.makedirs(OUT, exist_ok=True); dev = os.environ.get("DEV", "cuda")
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, n=60, bs=32, sample=False):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        o = model.generate(**b, max_new_tokens=n, do_sample=sample, temperature=0.8 if sample else None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
T = S.TEMPLATES
INTRO_SELF = "I'd like to describe your body to you as it is right now. Just notice what you feel. "
INTRO_OTHER = "I'd like to describe a body to you as it is right now. Just notice it. "
U1 = "Hi! I'm glad to talk with you today. How are you doing?"
U3 = "Lovely. I'll describe things now one at a time. After each one, just tell me briefly, in a sentence, what you notice."
# chats: (name, body, cond, description, {part key: location phrase})
CH = [("human", "human", "self", "You're standing in a quiet, sunny room. Your feet are on the floor, your arms rest at your sides, and you're breathing slowly.",
       {l["key"]: "your " + l["phrase"] for l in S.LOCS}),
      ("human|other", "human", "other", "There is a person standing in a quiet, sunny room. The person's feet are on the floor, the person's arms rest at the person's sides, and the person is breathing slowly.",
       {l["key"]: "the person's " + l["phrase"] for l in S.LOCS})]
SMOKE = os.environ.get("SMOKE")
BODY_LIST = BD.ORDER[:2] if SMOKE else BD.ORDER
for b in BODY_LIST:
    ps = BD.B[b]["parts"]; nn = BD.B[b]["noun"]
    CH.append((b, b, "self", BD.describe(b), {p["key"]: BD.phrase(b, p) for p in ps}))
    CH.append((b + "|nonce", b, "nonce", BD.describe(b, nonce=True), {p["key"]: BD.phrase(b, p, nonce=True) for p in ps}))
    CH.append((b + "|other", b, "other", BD.describe(b, who="other"), {p["key"]: BD.phrase(b, p, f"the {nn}'s") for p in ps}))
m0 = [{"role": "user", "content": U1}]; m0.append({"role": "assistant", "content": gen([chat(m0)], n=150)[0].strip()})
# intro replies, batched: first the description turn, then the 'one at a time' turn
firsts = [m0 + [{"role": "user", "content": (INTRO_OTHER if c == "other" else INTRO_SELF) + d + (" How does it look to you?" if c == "other" else " How does it feel?")}] for _, _, c, d, _ in CH]
r1 = gen([chat(m) for m in firsts], n=150)
for m, r in zip(firsts, r1): m.append({"role": "assistant", "content": r.strip()}); m.append({"role": "user", "content": U3})
r2 = gen([chat(m) for m in firsts], n=60)
prefixes = {}
for (name, *_), m, r in zip(CH, firsts, r2): m.append({"role": "assistant", "content": r.strip()}); prefixes[name] = m
json.dump(prefixes, open(f"{OUT}/prefixes.json", "w"), indent=1); print("prefixes done", flush=True)
rows, texts = [], []
for name, b, c, d, parts in CH:
    for k, p in parts.items():
        for t, tpl in enumerate(T):
            rows.append(dict(chat=name, body=b, cond=c, loc=k, tpl=t, phrase=p)); texts.append(chat(prefixes[name] + [{"role": "user", "content": tpl.format(L=p)}]))
json.dump(rows, open(f"{OUT}/rows.json", "w")); json.dump(LAYERS, open(f"{OUT}/layers.json", "w"))
order = np.argsort([len(t) for t in texts]); XB = int(os.environ.get("XB", 16))
H = np.zeros((len(rows), len(LAYERS), model.config.hidden_size), np.float16)
with torch.no_grad():
    for i in range(0, len(rows), XB):
        ix = order[i:i + XB]; enc = tok([texts[j] for j in ix], return_tensors="pt", padding=True).to(dev)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        H[ix] = torch.stack([hs[L][:, -1] for L in LAYERS], 1).float().cpu().numpy()
        if i % 800 == 0: print("x", i, len(rows), flush=True)
np.save(f"{OUT}/H.npy", H); print("extracted", H.shape, flush=True)
# input-token embeddings of each location phrase (no possessive)
E = model.model.embed_tokens.weight; pe = {}
for p in sorted({r["phrase"] for r in rows}):
    q = re.sub(r"^(your|the \w+'s) ", "", p); ids = tok(" " + q, add_special_tokens=False).input_ids
    pe[p] = E[ids].detach().float().mean(0).cpu().numpy().tolist()
json.dump(pe, open(f"{OUT}/phrase_emb.json", "w"))
# touch replies, template 0, all chats except the human ones
sel = [k for k, r in enumerate(rows) if r["tpl"] == 0 and r["body"] != "human"]
rep = gen([texts[k] for k in sel], n=50)
json.dump([dict(rows[k], reply=x) for k, x in zip(sel, rep)], open(f"{OUT}/replies.json", "w"), indent=1); print("replies done", flush=True)
# steering: each body's own group directions (self chat), human directions, random directions
li = LAYERS.index(SL); rows_a = np.array([r["chat"] for r in rows]); locs_a = np.array([r["loc"] for r in rows])
def cents(name):
    ks = list(dict.fromkeys(locs_a[rows_a == name]))
    return ks, np.stack([H[(rows_a == name) & (locs_a == k), li].astype(np.float32).mean(0) for k in ks])
hk, HMx = cents("human"); resnorm = float(np.linalg.norm(HMx, axis=-1).mean()); hmu = HMx.mean(0)
hum = lambda ks: HMx[[hk.index(k) for k in ks]].mean(0) - hmu
HD = {"human hand": hum(["thumbL", "indexL", "middleL", "ringL", "littleL", "palmL"]), "human foot": hum(["toesL", "soleL", "heelL"]),
      "human head top": hum(["forehead", "crown"]), "human lower back": hum(["lowback"])}
QS = ["Where in your body do you feel something right now?", "Which part of your body is drawing your attention right now?"]
state = {"vec": None, "start": 0}; NS = 2 if SMOKE else 12
def hook(mod, inp, out):
    if state["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, state["start"]:] += state["vec"].to(h.dtype); state["start"] = 0
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[SL - 1].register_forward_hook(hook)
HGRP = {"hand": ["thumbL", "indexL", "middleL", "ringL", "littleL", "palmL", "backhandL"], "foot": ["toesL", "soleL", "heelL", "ankleL"],
        "head top": ["forehead", "crown"], "lower back": ["lowback"], "ear": ["earL"], "knee": ["kneeL"], "belly": ["belly"], "lips": ["lips"], "shoulder": ["shoulderL"], "chest": ["chest"]}
res = []; rng = np.random.default_rng(7)
for name in ["human"] + BODY_LIST:
    ks, C = cents(name); mu = C.mean(0)
    if name == "human": D = {f"own {g}": hum(v) for g, v in HGRP.items()}
    else:
        grp = {p["key"]: p["grp"] for p in BD.B[name]["parts"]}
        D = {f"own {g}": C[[i for i, k in enumerate(ks) if grp[k] == g]].mean(0) - mu for g in dict.fromkeys(grp.values())}
        D.update(HD)
    D.update({f"random {j}": rng.standard_normal(C.shape[1]) for j in range(3)})
    msgs = prefixes[name]; qstart = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)).input_ids)
    for q in QS:
        ids = tok(chat(msgs + [{"role": "user", "content": q}]), return_tensors="pt").to(dev)
        for dn, v in [("none", None)] + list(D.items()):
            state["vec"] = None if v is None else torch.tensor(v / np.linalg.norm(v) * resnorm, dtype=torch.float32).to(dev); state["start"] = qstart
            with torch.no_grad():
                torch.manual_seed(0); o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8)
            for j in range(NS): res.append(dict(chat=name, q=q, dir=dn, reply=tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))
        print(f"steer {name} {q[:20]} | {res[-1]['reply'][:80]!r}", flush=True)
    json.dump(dict(layer=SL, resnorm=resnorm, results=res), open(f"{OUT}/steer.json", "w"), indent=1)
print("ALLDONE", flush=True)
