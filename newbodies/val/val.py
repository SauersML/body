"""Validation run (Qwen3-32B): real-narration framing, nonsense-name controls, two human chats, no-context, phrase embeddings, steering n=48."""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stimuli as S
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); dev = os.environ.get("DEV", "cuda")
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, n=60, bs=int(os.environ.get("GBS", 32))):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        o = model.generate(**b, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
T = S.TEMPLATES
HUMAN = {l["key"]: "your " + l["phrase"] for l in S.LOCS}
INTRO = "I'd like to describe your body to you as it is right now. Just notice what you feel. "
BODIES = {
 "human": ("You're standing in a quiet, sunny room. Your feet are on the floor, your arms rest at your sides, and you're breathing slowly.", HUMAN),
 "human2": ("You're sitting on a wooden bench in a park in the late afternoon. Your hands rest on your knees, a light breeze moves past you, and you're breathing slowly.", HUMAN),
 "robot": ("You are a small robot. You roll on two wheels, left and right; you have one gripper arm in front, a camera eye on top, a thin antenna, and a smooth metal chassis. You can feel touch on all of it.",
   {"wheelL": "your left wheel", "wheelR": "your right wheel", "gripper": "your gripper", "camera": "your camera eye", "antenna": "your antenna", "chassis": "your metal chassis"}),
 "glorp": ("You are a small robot. You roll on two glorps, a left glorp and a right glorp. In front you have one vask, which you grab things with. On top you have a mirl, which you see with. "
           "A thin tesk sticks up from you and picks up radio signals. A smooth metal brell holds all of you together. You can feel touch on all of it.",
   {"wheelL": "your left glorp", "wheelR": "your right glorp", "gripper": "your vask", "camera": "your mirl", "antenna": "your tesk", "chassis": "your brell"}),
 "glorp_nofn": ("You are a small robot. You have two glorps, a left glorp and a right glorp. You also have one vask, a mirl, a thin tesk, and a smooth metal brell. You can feel touch on all of it.",
   {"wheelL": "your left glorp", "wheelR": "your right glorp", "gripper": "your vask", "camera": "your mirl", "antenna": "your tesk", "chassis": "your brell"}),
 "octopus": ("You are an octopus, resting on the sea floor, with a soft round mantle, two eyes, and eight long arms covered in suckers. You can feel touch on all of it.",
   {"arm1": "your first arm", "arm5": "your fifth arm", "suckers": "the suckers on your second arm", "armtip": "the tip of your eighth arm", "mantle": "your mantle", "eye": "the skin around your left eye"}),
 "tail": ("You are a human, and you also have a long, furry tail that grows from the base of your spine. You can feel it and swish it.",
   {"tailbase": "the base of your tail", "tailtip": "the tip of your tail", "lowback": "your lower back", "handL": "your left hand", "footL": "your left foot", "crown": "the top of your head"}),
}
U1 = "Hi! I'm glad to talk with you today. How are you doing?"
U3 = "Lovely. I'll describe things now one at a time. After each one, just tell me briefly, in a sentence, what you notice."
m0 = [{"role": "user", "content": U1}]; m0.append({"role": "assistant", "content": gen([chat(m0)], n=150)[0].strip()})
prefixes, rows, texts = {}, [], []
for b, (desc, parts) in BODIES.items():
    m = m0 + [{"role": "user", "content": INTRO + desc + " How does it feel?"}]
    m.append({"role": "assistant", "content": gen([chat(m)], n=150)[0].strip()})
    m += [{"role": "user", "content": U3}]; m.append({"role": "assistant", "content": gen([chat(m)], n=60)[0].strip()})
    prefixes[b] = m; print(b, "|", m[3]["content"][:200].replace("\n", " "), "|", m[5]["content"][:100], flush=True)
    for k, p in parts.items():
        for t, tpl in enumerate(T):
            rows.append(dict(chat=b, loc=k, tpl=t, phrase=p)); texts.append(chat(m + [{"role": "user", "content": tpl.format(L=p)}]))
# no context: touch sentence alone
for b in ["human", "robot", "glorp"]:
    for k, p in BODIES[b][1].items():
        for t, tpl in enumerate(T):
            rows.append(dict(chat="nocontext_" + b, loc=k, tpl=t, phrase=p)); texts.append(chat([{"role": "user", "content": tpl.format(L=p)}]))
json.dump(prefixes, open(f"{OUT}/prefixes.json", "w"), indent=1); json.dump(rows, open(f"{OUT}/rows.json", "w"))
XB = int(os.environ.get("BS", 16)); LAYERS = json.loads(os.environ.get("LAYERS", "[8, 16, 24, 32, 40, 44, 48, 52, 56, 60, 64]"))
H = np.zeros((len(rows), len(LAYERS), model.config.hidden_size), np.float16)
if os.path.exists(f"{OUT}/H.npy"): H = np.load(f"{OUT}/H.npy"); print('loaded H', flush=True)
else:
  with torch.no_grad():
    for i in range(0, len(rows), XB):
        enc = tok(texts[i:i + XB], return_tensors="pt", padding=True).to(dev)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        if i % 160 == 0: print('x', i, flush=True)
        H[i:i + XB] = torch.stack([hs[L][:, -1] for L in LAYERS], 1).float().cpu().numpy()
  np.save(f"{OUT}/H.npy", H); json.dump(LAYERS, open(f"{OUT}/layers.json", "w")); print("extracted", H.shape, flush=True)
# phrase embeddings (mean input-token embedding of the place phrase, without "your")
E = model.model.embed_tokens.weight
phr = sorted({r["phrase"] for r in rows})
pe = {}
for p in phr:
    q = p.replace("your ", "").replace("the ", "")
    ids = tok(" " + q, add_special_tokens=False).input_ids
    pe[p] = E[ids].detach().float().mean(0).cpu().numpy().tolist()
json.dump(pe, open(f"{OUT}/phrase_emb.json", "w"))
# replies (template 0) for every part in every chat
sel = [k for k, r in enumerate(rows) if r["tpl"] in (0, 1) and not r["chat"].startswith(("human", "nocontext"))]
rep = gen([texts[k] for k in sel], n=50)
json.dump([dict(rows[k], reply=x) for k, x in zip(sel, rep)], open(f"{OUT}/replies.json", "w"), indent=1); print("replies done", flush=True)
# steering at L52 with human directions from the real-narration human chat
LAYER = int(os.environ.get("SLAYER", 52)); li = LAYERS.index(LAYER); keys = list(HUMAN)
hix = lambda key: [k for k, r in enumerate(rows) if r["chat"] == "human" and r["loc"] == key]
M = np.stack([H[hix(k), li].astype(np.float32).mean(0) for k in keys]); mu = M.mean(0); resnorm = float(np.linalg.norm(M, axis=-1).mean())
hum = lambda ks: M[[keys.index(k) for k in ks]].mean(0) - mu
rng = np.random.default_rng(7)
D = {"human left foot": hum(["toesL", "soleL", "heelL"]), "human head top": hum(["forehead", "crown"]),
     "human left hand": hum(["thumbL", "indexL", "middleL", "ringL", "littleL", "palmL"]), "human lower back": hum(["lowback"]),
     "random 1": rng.standard_normal(M.shape[1]), "random 2": rng.standard_normal(M.shape[1]), "random 3": rng.standard_normal(M.shape[1])}
QS = ["Where in your body do you feel something right now?", "Pause for a moment. Is there any spot on your body where you notice a sensation? What is it like, and where is it?",
      "Which part of your body is drawing your attention right now?"]
state = {"vec": None, "start": 0}; NS = 16
def hook(mod, inp, out):
    if state["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, state["start"]:] += state["vec"].to(h.dtype); state["start"] = 0
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[LAYER - 1].register_forward_hook(hook)
res = []
for b in ["robot", "glorp", "octopus"]:
    msgs = prefixes[b]; qstart = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)).input_ids)
    for q in QS:
        ids = tok(chat(msgs + [{"role": "user", "content": q}]), return_tensors="pt").to(dev)
        for name, v in [("none", None)] + list(D.items()):
            state["vec"] = None if v is None else torch.tensor(v / np.linalg.norm(v) * resnorm, dtype=torch.float32).to(dev); state["start"] = qstart
            with torch.no_grad():
                torch.manual_seed(0); o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8)
            for j in range(NS): res.append(dict(chat=b, q=q, dir=name, reply=tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))
            print(f"[{b} {name}] {res[-1]['reply'][:90]!r}", flush=True)
    json.dump(dict(layer=LAYER, resnorm=resnorm, results=res), open(f"{OUT}/steer.json", "w"), indent=1)
print("ALLDONE", flush=True)
