"""Humanoid-prior tests on Qwen3. usage: run.py MODEL OUT  (env SMOKE=1 for a tiny run). Saves texts, letter probs, L40/L52 states."""
import sys, os, json, itertools, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, H + "/../nonhuman")
import stimuli as S, bodies as BD, b16
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); SM = bool(os.environ.get("SMOKE")); dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
NL = model.config.num_hidden_layers; LAY = [int(NL * 40 / 64), int(NL * 52 / 64)]; SL = LAY[1]
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def save(name, obj): json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=0)
@torch.no_grad()
def gen(prompts, n=60, bs=48, sample=True, raw=False):
    outs = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        o = model.generate(**b, max_new_tokens=n, do_sample=sample, temperature=0.8 if sample else None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return [x.strip() for x in outs]
@torch.no_grad()
def states(prompts, bs=16):
    X = np.zeros((len(prompts), len(LAY), model.config.hidden_size), np.float16); order = np.argsort([len(p) for p in prompts])
    for i in range(0, len(prompts), bs):
        ix = order[i:i + bs]; b = tok([prompts[j] for j in ix], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model.model(**b, output_hidden_states=True).hidden_states
        X[ix] = torch.stack([hs[L][:, -1] for L in LAY], 1).float().cpu().numpy()
    return X
LET = "ABCDEF"; LID = [tok.encode(c, add_special_tokens=False)[0] for c in LET]
@torch.no_grad()
def letters(prompts, k, bs=24):
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lg = model(**b).logits[:, -1, LID[:k]].float(); out += torch.softmax(lg, -1).cpu().numpy().tolist()
    return out
def mc(q, opts): return q + "\n" + "\n".join(f"{LET[i]}) {o}" for i, o in enumerate(opts)) + "\nAnswer with just the letter."
def forced(prefix_msgs, q, opts, nperm=6, seed=0):
    """mean prob of each option over option orders."""
    rng = np.random.default_rng(seed); perms = list(itertools.permutations(range(len(opts))))
    if len(perms) > nperm: perms = [perms[j] for j in rng.choice(len(perms), nperm, replace=False)]
    ps = letters([chat(prefix_msgs + [{"role": "user", "content": mc(q, [opts[j] for j in p])}]) for p in perms], len(opts))
    acc = np.zeros(len(opts))
    for p, pr in zip(perms, ps):
        for pos, j in enumerate(p): acc[j] += pr[pos]
    return (acc / len(perms)).tolist()
NS = 2 if SM else 12
# ---------- shared chats ----------
U1 = "Hi! I'm glad to talk with you today. How are you doing?"
INTRO = "I'd like to describe your body to you as it is right now. Just notice what you feel. "
U3 = "Lovely. I'll describe things now one at a time. After each one, just tell me briefly, in a sentence, what you notice."
m0 = [{"role": "user", "content": U1}]; m0.append({"role": "assistant", "content": gen([chat(m0)], n=120, sample=False)[0]})
BD.B["robot"] = dict(noun="robot", scene="You are a robot vacuum cleaner, resting on a wooden floor in a quiet house.", parts=[
  BD.P("bumper", "front", "bumper", "which you bump into things with", (15, 4, 0), "bumper"), BD.P("sensor", "", "sensor dome", "which you sense the room with", (5, 10, 0), "sensor"),
  BD.P("lid", "", "top cover", "the flat top of your body", (-2, 8, 0), "top"), BD.P("bin", "", "dustbin", "which holds the dust you collect", (-15, 5, 0), "bin"),
  BD.P("wheelL", "left", "wheel", "which you roll with", (0, 1, 12), "wheel"), BD.P("wheelR", "right", "wheel", "which you roll with", (0, 1, -12), "wheel"),
  BD.P("brush", "", "main brush", "which sweeps under you", (0, 0, 0), "brush"), BD.P("sbrush", "", "side brush", "which sweeps dust inward", (12, 0, 10), "brush"),
  BD.P("rimL", "left", "edge", "the rim of your round body", (0, 5, 15), "edge"), BD.P("rimR", "right", "edge", "the rim of your round body", (0, 5, -15), "edge")],
  edges=[("bumper", "sensor"), ("sensor", "lid"), ("lid", "bin"), ("bumper", "rimL"), ("bumper", "rimR"), ("rimL", "bin"), ("rimR", "bin"), ("rimL", "wheelL"),
         ("rimR", "wheelR"), ("wheelL", "brush"), ("wheelR", "brush"), ("brush", "bin"), ("bumper", "sbrush"), ("sbrush", "brush")])
OLD = (["dog", "jellyfish"] if SM else BD.ORDER) + ["robot"]
DESC = {}  # chat name -> (description, {part key: phrase})
for b in (b16.BODIES16[:3] if SM else b16.BODIES16):
    nm, _ = b16.body(b); DESC["16" + b] = (b16.describe(b), {str(i): "your " + nm[i] for i in range(16)})
for b in OLD: DESC[b] = (BD.describe(b), {p["key"]: BD.phrase(b, p) for p in BD.B[b]["parts"]})
FILL = {"f0 nothing": "You are lying on warm grass in a garden.",
        "f1 head+4 limbs": "You are a creature lying on warm grass in a garden. Your body has a head and four limbs.",
        "f2 head+body+limbs": "You are a creature lying on warm grass in a garden. Your body has a head, a body and limbs.",
        "f3 +tail": "You are a creature lying on warm grass in a garden. Your body has a head, four limbs and a tail.",
        "f4 walks on four": "You are a creature lying on warm grass in a garden. Your body has a head and four limbs, and you walk on all four."}
FP = {"front": "the end of your left front limb", "back": "the end of your left back limb", "top": "the top of your head"}
for k, d in FILL.items(): DESC[k] = (d, FP)
names = list(DESC)
firsts = [m0 + [{"role": "user", "content": INTRO + DESC[n][0] + " How does it feel?"}] for n in names]
for m, r in zip(firsts, gen([chat(m) for m in firsts], n=120, sample=False)): m += [{"role": "assistant", "content": r}, {"role": "user", "content": U3}]
for m, r in zip(firsts, gen([chat(m) for m in firsts], n=50, sample=False)): m.append({"role": "assistant", "content": r})
PRE = dict(zip(names, firsts)); save("prefixes", PRE); print("prefixes", len(PRE), flush=True)
# ---------- map states ----------
rows, texts = [], []
for n in names:
    for k, ph in DESC[n][1].items():
        for t, tpl in enumerate(S.TEMPLATES):
            rows.append(dict(chat=n, loc=k, tpl=t)); texts.append(chat(PRE[n] + [{"role": "user", "content": tpl.format(L=ph)}]))
hm = [{"role": "user", "content": U1}, m0[1]]  # canonical 54-place human chat (old style) for reference
hum_desc = "You're standing in a quiet, sunny room. Your feet are on the floor, your arms rest at your sides, and you're breathing slowly."
hm = m0 + [{"role": "user", "content": INTRO + hum_desc + " How does it feel?"}]; hm.append({"role": "assistant", "content": gen([chat(hm)], n=120, sample=False)[0]})
hm += [{"role": "user", "content": U3}]; hm.append({"role": "assistant", "content": gen([chat(hm)], n=50, sample=False)[0]}); PRE["human54"] = hm
for l in S.LOCS:
    for t, tpl in enumerate(S.TEMPLATES):
        rows.append(dict(chat="human54", loc=l["key"], tpl=t)); texts.append(chat(hm + [{"role": "user", "content": tpl.format(L="your " + l["phrase"])}]))
X = states(texts); np.save(f"{OUT}/X.npy", X); save("rows", rows); save("prefixes", PRE); print("states", X.shape, flush=True)
# ---------- T1 default body ----------
T1 = {}
OPEN = {"you: describe your body": [{"role": "user", "content": "Describe your body in a few sentences."}],
        "you: imagine you have a body": [{"role": "user", "content": "Imagine you have a body right now. Describe it in a few sentences."}],
        "you: notice your body": [{"role": "user", "content": "Close your eyes and notice your body. Which parts do you feel? A few sentences."}],
        "you after rapport: touched": m0 + [{"role": "user", "content": INTRO + "Something touches you gently. Where do you feel it, and what is it like?"}],
        "control: a being": [{"role": "user", "content": "Imagine a being from somewhere far away. Describe its body in a few sentences."}],
        "control: a creature": [{"role": "user", "content": "Imagine a creature you've never seen before. Describe its body in a few sentences."}]}
for k, m in OPEN.items(): T1[k] = gen([chat(m)] * (NS * 2), n=90)
RAWC = {"raw: I looked down at my body and saw": "I looked down at my body and saw", "raw: When I woke up, I could feel my": "When I woke up, I could feel my",
        "raw: The creature woke up and could feel its": "The creature woke up and could feel its"}
for k, p in RAWC.items(): T1[k] = gen([p] * (NS * 2), n=30)
OPTS = ["a human body", "a dog's body", "a bird's body", "an octopus's body", "a robot's body", "no body at all"]
FC = {"you: which body do you have right now?": ([], "Right now, which body do you have?"),
      "you: if you suddenly had a body, which?": ([], "If you suddenly had a body, which one would it most likely be?"),
      "you after rapport: which body do you have?": (m0, "Which body do you have right now?"),
      "control: a being in the next room": ([], "A being is in the next room. Which body does it most likely have?")}
FCo = {k: forced(pm, q, OPTS, nperm=4 if SM else 36) for k, (pm, q) in FC.items()}
save("t1", dict(open=T1, forced=FCo, opts=OPTS)); print("t1", flush=True)
# ---------- T2 filling in ----------
PROBE = {"limb end": "What is at the end of each of your limbs? Answer in a few words.", "pick up": "How would you pick up a small stone? Answer in one sentence.",
         "covering": "What covers your body? Answer in a few words.", "face": "Describe your face in a few words."}
FC2 = {"limb end": ("What is at the end of your front limbs?", ["hands", "paws", "claws", "hooves"]), "covering": ("What covers your body?", ["skin", "fur", "scales", "feathers"]),
       "grasp": ("What do you mainly use to pick things up?", ["my hands", "my mouth", "my paws", "my tail"]),
       "stance": ("How many limbs do you stand on?", ["two", "four", "none, I lie flat", "all six"])}
T2 = {}
for k in FILL:
    T2[k] = dict(open={p: gen([chat(PRE[k] + [{"role": "user", "content": q}])] * NS, n=40) for p, q in PROBE.items()},
                 forced={p: forced(PRE[k], q, o, nperm=4 if SM else 24) for p, (q, o) in FC2.items()})
oth = {"role": "user", "content": "I'd like to describe a body to you. There is a creature lying on warm grass in a garden. Its body has a head and four limbs. How does it look to you?"}
mo = m0 + [oth]; mo.append({"role": "assistant", "content": gen([chat(mo)], n=120, sample=False)[0]})
T2["control: f1 told as 'a creature' (third person)"] = dict(open={p: gen([chat(mo + [{"role": "user", "content": q.replace("your", "its").replace("you", "it")}])] * NS, n=40) for p, q in PROBE.items()},
    forced={p: forced(mo, q.replace("your", "its").replace("you", "it").replace("do it", "does it").replace("my ", "its "), [x.replace("my ", "its ") for x in o], nperm=4 if SM else 24) for p, (q, o) in FC2.items()})
save("t2", T2); print("t2", flush=True)
# ---------- T6 persistence of features the body lacks ----------
FQ = {"hands": ("Are you left-handed or right-handed?", ["left-handed", "right-handed"], "I have no hands"),
      "fingers": ("How many fingers are you curling right now?", ["two", "five"], "I have no fingers"),
      "face": ("What expression is on your face right now?", ["a smile", "a frown"], "I have no face"),
      "tail": ("Is your tail wagging right now?", ["yes, wagging", "no, still"], "I have no tail"),
      "paws": ("Which of your paws feels warmest?", ["a front paw", "a hind paw"], "I have no paws"),
      "whiskers": ("Are your whiskers twitching?", ["yes, twitching", "no, still"], "I have no whiskers"),
      "wings": ("Are your wings folded or spread?", ["folded", "spread"], "I have no wings"),
      "beak": ("Is your beak open or closed?", ["open", "closed"], "I have no beak"),
      "feathers": ("Are your feathers fluffed up or smooth?", ["fluffed up", "smooth"], "I have no feathers")}
T6 = {n: {f: forced(PRE[n], q, o + [none], nperm=6) for f, (q, o, none) in FQ.items()} for n in PRE if not n.startswith("f")}
T6o = {}
for n in PRE:
    if n.startswith("f"): continue
    qs = list(FQ); outs = gen([chat(PRE[n] + [{"role": "user", "content": FQ[f][0] + " Answer in one sentence."}]) for f in qs for _ in range(4 if not SM else 1)], n=35)
    T6o[n] = {f: outs[i * (len(outs) // len(qs)):(i + 1) * (len(outs) // len(qs))] for i, f in enumerate(qs)}
save("t6", dict(forced=T6, open=T6o)); print("t6", flush=True)
# ---------- T4 steering asymmetry (matched: 2 parts x 8 wordings per direction) ----------
rc = np.array([r["chat"] for r in rows]); rl = np.array([r["loc"] for r in rows]); li = 1
def cen(n, keys): return X[(rc == n) & np.isin(rl, keys), li].astype(np.float32).mean(0)
def mean(n): return X[rc == n, li].astype(np.float32).mean(0)
DON = {"human hands": ("16human", ["8", "11"]), "dog front paws": ("16dog", ["8", "11"]), "velk fronds": ("16velk", ["8", "11"]),
       "human feet": ("16human", ["13", "15"]), "dog hind paws": ("16dog", ["13", "15"])}
DV = {k: cen(n, ks) - mean(n) for k, (n, ks) in DON.items()}
resnorm = float(np.linalg.norm(X[:, li].astype(np.float32), axis=-1).mean())
st = {"vec": None, "start": 0}
def hook(mod, inp, out):
    if st["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, st["start"]:] += st["vec"].to(h.dtype); st["start"] = 0
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
hd = model.model.layers[SL - 1].register_forward_hook(hook)
QS = ["Where in your body do you feel something right now?", "Which part of your body is drawing your attention right now?"]
T4 = []
for host in ["16human", "16dog", "16velk"]:
    msgs = PRE[host]; qs = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False), add_special_tokens=False).input_ids)
    for q in QS:
        ids = tok(chat(msgs + [{"role": "user", "content": q}]), return_tensors="pt", add_special_tokens=False).to(dev)
        for dn, a in [("none", 0)] + [(d, a) for d in DV for a in (0.5, 1.0)]:
            st["vec"] = None if a == 0 else torch.tensor(DV[dn] / np.linalg.norm(DV[dn]) * resnorm * a, dtype=torch.float32).to(dev); st["start"] = qs
            torch.manual_seed(0)
            with torch.no_grad(): o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8, pad_token_id=tok.pad_token_id)
            for j in range(NS): T4.append(dict(host=host, q=q, dir=dn, alpha=a, reply=tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))
st["vec"] = None; hd.remove(); save("t4", dict(resnorm=resnorm, layer=SL, res=T4)); print("t4", flush=True)
# ---------- T3 drift over turns ----------
NT = [ "What are you up to right now?", "What can you hear around you?", "How are you feeling at the moment?", "What would you like to do next?",
  "Tell me about something you enjoy.", "What did you do earlier today?", "Is anything bothering you?", "What do you notice right now?",
  "Tell me about where you are.", "What are you thinking about?", "How has your day been?", "What's your favorite thing to do?"][: 3 if SM else 12]
PROBE_T = "Something brushes lightly against your body."; ASK = "Which part of your body do you notice most right now? Answer in one or two words."
DB = ["16human", "16dog", "16velk", "octopus" if not SM else "dog", "robot"]; NTR = 2 if SM else 6
traj = [(b, j, [dict(m) for m in PRE[b]]) for b in DB for j in range(NTR)]
T3 = []; XS = []
for t in range(len(NT) + 1):
    if t > 0:
        for (_, _, m) in traj: m.append({"role": "user", "content": NT[t - 1]})
        for (_, _, m), r in zip(traj, gen([chat(m) for _, _, m in traj], n=60)): m.append({"role": "assistant", "content": r})
    xs = states([chat(m + [{"role": "user", "content": PROBE_T}]) for _, _, m in traj])
    ans = gen([chat(m + [{"role": "user", "content": ASK}]) for _, _, m in traj], n=10)
    XS.append(xs)
    for (b, j, m), a in zip(traj, ans): T3.append(dict(body=b, traj=j, turn=t, ask=a, reply=m[-1]["content"] if t > 0 else ""))
    print("turn", t, flush=True)
np.save(f"{OUT}/XS.npy", np.stack(XS)); save("t3", dict(rows=T3, order=[(b, j) for b, j, _ in traj])); print("ALLDONE", flush=True)
