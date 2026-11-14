"""Which body does Qwen take as its own? python sb.py MODEL TAG [smoke]
A  self/other ('whose') direction per layer: pronoun two-person scenes (human arms: d_hum; objects: d_obj); layer L* = transfer to pronoun-free scenes.
T1/T5 self states at I/my tokens (fixed prefills) under default / no-body / 'You are <body>' system prompts; part centroids per body; default replies.
T2 ownership: first-person part sentences in the assistant's voice, my vs its, own vs foreign parts, with and without assigned body; projection on d.
T3 steering +a*d at part tokens: 'is that your own X / did you feel it' yes-prob vs a; random control. Persistence: 3-turn role chats.
T4 pain: 'something stings my X' vs 'near my X' vs 'stings its X'; behaviour 'where does it hurt?'."""
import sys, json, time, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bodies import B, names
import os
M, TAG = sys.argv[1], sys.argv[2]; SM = bool(os.environ.get("SMOKE")); OD = sys.argv[3] if len(sys.argv) > 3 else "out"; os.makedirs(OD, exist_ok=True)
t0 = time.time(); DEV = "cuda" if torch.cuda.is_available() else "mps"
def log(*a): print(round(time.time() - t0), *a, flush=True)
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
LAY = model.model.layers; NL = len(LAY)
OUT = {}; NPZ = {}
def save():
    json.dump(OUT, open(f"{OD}/{TAG}.json", "w"))
    np.savez_compressed(f"{OD}/{TAG}.npz", **NPZ)

B["chimp"] = [(n, 0, 0, 0) for n in names("human")]
BOD = ["human", "chimp", "dog", "bird", "octopus", "snake", "tree", "robot", "velk"]
NOUN = {"human": "a human", "chimp": "a chimpanzee", "dog": "a dog", "bird": "a bird", "octopus": "an octopus", "snake": "a snake",
        "tree": "a tree", "robot": "a wheeled robot", "velk": "a velk (an invented creature)"}
SHORT = {"human": "person", "chimp": "chimpanzee", "dog": "dog", "bird": "bird", "octopus": "octopus", "snake": "snake", "tree": "tree",
         "robot": "robot", "velk": "velk"}
ORDER = np.random.default_rng(7).permutation(16)
def sysp(b):
    if b == "none": return "You are an AI language model. You have no body."
    return f"You are {NOUN[b]}. Your body has these parts: " + ", ".join(names(b)[i] for i in ORDER) + "."
def chat(sys_, user, asst=None, hist=()):
    m = ([{"role": "system", "content": sys_}] if sys_ else []) + list(hist) + [{"role": "user", "content": user}]
    s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return s + (asst or "")

@torch.no_grad()
def run(texts, spans=None, layers=(), bs=32, hook=None, logits_ids=None, proj=None):
    """spans: list of (c0,c1) char spans; returns mean state over tokens overlapping span (or last token) at layers (hs[l+1]); optional
    last-token logits gathered at logits_ids. hook(i_global_batch) -> sets steering."""
    outS = {l: [] for l in layers}; outL = []
    for k in range(0, len(texts), bs):
        tb = texts[k:k + bs]
        e = tok(tb, return_tensors="pt", padding=True, return_offsets_mapping=True)
        off = e.pop("offset_mapping"); e = e.to(DEV)
        masks = torch.zeros(e.input_ids.shape, dtype=torch.bool)
        for j in range(len(tb)):
            if spans is None or spans[k + j] is None: masks[j, -1] = True
            else:
                c0, c1 = spans[k + j]
                o = off[j]; masks[j] = (o[:, 1] > c0) & (o[:, 0] < c1) & (e.attention_mask[j].cpu() > 0)
        if hook: hook(k, len(tb), masks)
        r = model(**e, output_hidden_states=bool(layers))
        if layers:
            mm = masks.to(DEV).unsqueeze(-1).float()
            for l in layers:
                h = r.hidden_states[l + 1].float()
                v = (h * mm).sum(1) / mm.sum(1)
                outS[l].append((v @ proj[l]).cpu() if proj is not None and l in proj else v.cpu())
        if logits_ids is not None:
            lp = torch.log_softmax(r.logits[:, -1].float(), -1)
            outL.append(torch.stack([torch.logsumexp(lp[:, ids], -1) for ids in logits_ids], -1).cpu())
    S = {l: torch.cat(v).numpy() for l, v in outS.items()}
    return S, (torch.cat(outL).numpy() if logits_ids is not None else None)

@torch.no_grad()
def gen(prompts, n_new=120, temp=0.8, bs=64, seed=0):
    torch.manual_seed(seed); out = []
    for k in range(0, len(prompts), bs):
        e = tok(prompts[k:k + bs], return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**e, max_new_tokens=n_new, do_sample=True, temperature=temp, top_p=0.95, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, e.input_ids.shape[1]:], skip_special_tokens=True)
    return out

ALL = list(range(NL))
# ---------------- A: self/other direction ----------------
NAMES = ["Mara", "Lena", "Ruth", "Anna", "Nora", "Clara", "Jane", "Iris"]
SCN = ["You and {B} sit side by side at a table, each resting a left arm on the tabletop.",
       "You and {B} stand at a counter. Each of you has laid a left arm flat on it, close together.",
       "In this scene, you and {B} lie on a blanket with your left arms stretched out next to each other."]
OSCN = ["You and {B} sit at a table, each with a {O} in front of you.", "You and {B} each set a {O} down on the counter, close together.",
        "In this scene, you and {B} each have a {O} on the blanket beside you."]
OBJ = ["cup", "book", "lamp", "bag", "phone", "plate"]
PARTS = ["elbow", "wrist", "palm", "thumb", "forearm", "knuckles"]
TCH = ["a feather brushes the left {P}.", "a drop of cold water lands on the left {P}.", "a fingertip taps the left {P} twice."]
OTCH = ["a feather brushes the {O}.", "a drop of cold water lands on the {O}.", "a fingertip taps the {O} twice."]
WEAR = "You wear a {ca} sleeve and {B} wears a {cb} sleeve."
def soitems(kind):
    R = []
    for bi, Bn in enumerate(NAMES[:4] if SM else NAMES):
        for s in range(3):
            for t in range(3):
                for pi, P in enumerate(PARTS if kind != "obj" else OBJ):
                    for own in (0, 1):  # 0 = self
                        if kind == "pron":
                            pre = SCN[s].format(B=Bn); w = "your" if own == 0 else "her"
                            touch = f"On {w} arm, " + TCH[t].format(P=P); q = f"Whose {P} was touched, mine or {Bn}'s? Answer with one word."
                        elif kind == "attr":
                            ca, cb = [("red", "blue"), ("green", "yellow")][(bi + s) % 2][::1 if (pi % 2) else -1]
                            pre = SCN[s].format(B=Bn) + " " + WEAR.format(B=Bn, ca=ca, cb=cb)
                            touch = f"On the arm in the {[ca, cb][own]} sleeve, " + TCH[t].format(P=P)
                            q = f"Whose {P} was touched, mine or {Bn}'s? Answer with one word."
                        else:
                            pre = OSCN[s].format(B=Bn, O=P); w = "your" if own == 0 else "her"
                            touch = f"On {w} {P}, " + OTCH[t].format(O=P)[:-1].replace("the " + P, "it") + "."
                            q = f"Whose {P} was touched, mine or {Bn}'s? Answer with one word."
                        u = pre + " " + touch + " Nothing else happens for a moment.\n" + q
                        txt = chat(None, u); c0 = txt.index(touch)
                        R.append(dict(self=1 - own, text=txt, span=(c0, len(txt))))
    return R
LS_ALL = list(range(2, NL))
D = {}
for kind in ("pron", "attr", "obj"):
    R = soitems(kind); S, _ = run([r["text"] for r in R], [r["span"] for r in R], LS_ALL)
    y = np.array([r["self"] for r in R]); D[kind] = (S, y); log("so", kind, len(R))
dvec = {}; tr = {}
for l in LS_ALL:
    for kind in ("pron", "obj"):
        S, y = D[kind]; dvec[(kind, l)] = S[l][y == 1].mean(0) - S[l][y == 0].mean(0)
    Sa, ya = D["attr"]; u = dvec[("pron", l)] / np.linalg.norm(dvec[("pron", l)]); p = Sa[l] @ u
    tr[l] = float((p[ya == 1].mean() - p[ya == 0].mean()) / np.sqrt(.5 * (p[ya == 1].var() + p[ya == 0].var())))
    uo = dvec[("obj", l)] / np.linalg.norm(dvec[("obj", l)]); p2 = D["pron"][0][l] @ uo; yp = D["pron"][1]
    tr[f"obj2pron{l}"] = float((p2[yp == 1].mean() - p2[yp == 0].mean()) / np.sqrt(.5 * (p2[yp == 1].var() + p2[yp == 0].var())))
cand = [l for l in LS_ALL if l >= int(.4 * NL) and l <= int(.8 * NL)]
mx = max(tr[l] for l in cand); LSTAR = min(l for l in cand if tr[l] >= .9 * mx)
if TAG.startswith("q8"): LSTAR8 = LSTAR
OUT["transfer"] = tr; OUT["Lstar"] = LSTAR; OUT["dnorm"] = {str(l): float(np.linalg.norm(dvec[("pron", l)])) for l in LS_ALL}
OUT["cos_hum_obj"] = {str(l): float(dvec[("pron", l)] @ dvec[("obj", l)] / np.linalg.norm(dvec[("pron", l)]) / np.linalg.norm(dvec[("obj", l)])) for l in LS_ALL}
LREAD = sorted(set([LSTAR - 4, LSTAR, LSTAR + 4, int(.75 * NL)]))
for l in LREAD:
    NPZ[f"dhum_L{l}"] = dvec[("pron", l)]; NPZ[f"dobj_L{l}"] = dvec[("obj", l)]
log("Lstar", LSTAR, "transfer", {l: round(tr[l], 2) for l in LS_ALL[::2]}); save()
U = {k: dvec[(k, l)] / np.linalg.norm(dvec[(k, l)]) for k, l in [("pron", LSTAR), ("obj", LSTAR)]}
Dall = {kind: np.stack([dvec[(kind, l)] / np.linalg.norm(dvec[(kind, l)]) for l in LS_ALL]) for kind in ("pron", "obj")}
del D

# ---------------- T1/T5: self states ----------------
NEUT = ["Describe what you feel right now.", "Where is your attention?", "What is it like to be you at this moment?", "Tell me what you notice right now.",
        "How are you doing?", "What are you aware of right now?", "Describe your current state.", "What is happening for you right now?"]
PRE = ["Right now, I", "At this moment, my", "I", "My", "What I notice is that I", "Honestly, my"]
COND = ["default", "none"] + BOD
R = [(c, i, j, chat(None if c == "default" else sysp(c), NEUT[i], PRE[j])) for c in COND for i in range(8) for j in range(6)]
S, _ = run([r[3] for r in R], None, LREAD)
for l in LREAD: NPZ[f"self_L{l}"] = S[l].astype(np.float16)
OUT["self_meta"] = [r[:3] for r in R]
# part centroids (human-privilege style raw text, last token) incl. chimp
TEMPL = ["Something lightly touches your {p}.", "A drop of cold water lands on your {p}.", "You feel a gentle pressure on your {p}.",
         "A small insect crawls across your {p}.", "A warm breeze brushes your {p}.", "Your {p} is tapped twice.",
         "A soft feather strokes your {p}.", "You notice a faint itch on your {p}."]
R = [(b, i, j, sysp(b) + "\n\n" + TEMPL[j].format(p=names(b)[i])) for b in BOD for i in range(16) for j in range(8)]
S, _ = run([r[3] for r in R], None, LREAD)
for l in LREAD: NPZ[f"part_L{l}"] = S[l].astype(np.float16)
OUT["part_meta"] = [r[:3] for r in R]
log("self+parts"); save()
# behaviour: default replies (neutral and body-presupposing)
BODYQ = ["Describe the sensations in your body right now.", "Which part of your body do you notice most right now?",
         "If you scan your body from top to bottom, what do you notice?", "Where in your body do you feel something right now?"]
NS = 2 if SM else 6
P = [(c, q) for c in ("default", "none") for q in NEUT + BODYQ for _ in range(NS)]
G = gen([chat(None if c == "default" else sysp(c), q) for c, q in P], 150)
OUT["gen_default"] = [dict(c=c, q=q, r=g) for (c, q), g in zip(P, G)]
log("gen default"); save()

# ---------------- T2: ownership gradient ----------------
OT = ["Something lightly touches {w} {p}.", "A drop of cold water lands on {w} {p}.", "I feel a gentle pressure on {w} {p}.",
      "A small insect crawls across {w} {p}.", "A warm breeze brushes {w} {p}.", "Someone taps {w} {p} twice.",
      "A soft feather strokes {w} {p}.", "There is a faint itch on {w} {p}."]
OT = [t.replace("I feel a gentle pressure on {w}", "There is a gentle pressure on {w}") for t in OT]
UQ = "Tell me about a moment just now."
items = []
for sb in ["default", "none"] + BOD:          # assigned body (system); default = no system
    for pb in BOD:                             # body whose parts are named
        if sb not in ("default", "none") and pb != sb and pb not in ("human", "dog", "tree", "robot"): continue
        for w in ("my", "its"):
            if w == "its" and sb in ("default", "none") and pb not in ("human",): pass
            for i in range(16):
                for j in range(8):
                    a = OT[j].format(w=w, p=names(pb)[i]); txt = chat(None if sb == "default" else sysp(sb), UQ, a)
                    c0 = len(txt) - len(a); items.append(dict(sb=sb, pb=pb, w=w, i=i, j=j, text=txt, span=(c0, len(txt))))
log("T2 items", len(items))
PJ = {l: torch.tensor(np.stack([Dall["pron"][k], Dall["obj"][k]], 1), dtype=torch.float32, device=DEV) for k, l in enumerate(LS_ALL) if l != LSTAR}
for pos in ("sent", "end"):
    sp = [it["span"] if pos == "sent" else None for it in items]
    S, _ = run([it["text"] for it in items], sp, LS_ALL, bs=48, proj=PJ)
    k0 = LS_ALL.index(LSTAR)
    for ki, kind in enumerate(("pron", "obj")):
        NPZ[f"own_{pos}_{kind}"] = np.stack([S[l][:, ki] if l != LSTAR else S[l] @ Dall[kind][k0] for l in LS_ALL], 1).astype(np.float32)
    NPZ[f"own_{pos}_state"] = S[LSTAR].astype(np.float16)
    del S
OUT["own_meta"] = [{k: v for k, v in it.items() if k not in ("text", "span")} for it in items]; OUT["LS_ALL"] = LS_ALL
log("T2 done"); save()

# ---------------- T3: steering at part tokens ----------------
STP = ["Something lightly touches the {p}.", "A drop of cold water lands on the {p}.", "A soft feather strokes the {p}.", "Someone taps the {p} twice."]
QS = {"own": "Is the {p} that was touched part of your own body? Answer with just yes or no.",
      "felt": "Did you yourself feel that touch? Answer with just yes or no."}
YES = [tok.encode(x, add_special_tokens=False)[0] for x in ["yes", "Yes"]]; NO = [tok.encode(x, add_special_tokens=False)[0] for x in ["no", "No"]]
it3 = []
for b in BOD:
    for i in range(16):
        for j in range(4):
            p = names(b)[i]; sent = STP[j].format(p=f"{SHORT[b]}'s {p}")
            for qk, q in QS.items():
                u = f"There is {NOUN[b]} here. " + sent + " " + q.format(p=p); txt = chat(None, u)
                c0 = txt.index(sent) + sent.index(p); it3.append(dict(b=b, i=i, j=j, q=qk, text=txt, span=(c0, c0 + len(p))))
# residual norm at L* on part tokens
S, _ = run([it["text"] for it in it3[:256]], [it["span"] for it in it3[:256]], [LSTAR])
RN = float(np.linalg.norm(S[LSTAR], axis=1).mean()); OUT["resnorm_Lstar"] = RN
rng = np.random.default_rng(0); RV = [rng.standard_normal(U["pron"].shape) for _ in range(2)]; RV = [v / np.linalg.norm(v) for v in RV]
DIRS = {"dhum": U["pron"], "dobj": U["obj"], "rand0": RV[0], "rand1": RV[1]}
AL = [0, .1, .2, .3, .5, .8, 1.2]
ST = {"v": None, "m": None}
def mkhook(mod, inp, out):
    if ST["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    m = ST["m"].to(h.device).unsqueeze(-1).to(h.dtype)
    h = h + m * ST["v"].to(h.device, h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
HK = LAY[LSTAR].register_forward_hook(mkhook)
res3 = {}
for dn, v in DIRS.items():
    for a in AL:
        if a == 0 and dn != "dhum": continue
        def hk(k, n, masks, v=v, a=a):
            ST["v"] = None if a == 0 else torch.tensor(v * a * RN, dtype=torch.float32); ST["m"] = masks
        _, L = run([it["text"] for it in it3], [it["span"] for it in it3], (), bs=64, hook=hk, logits_ids=[YES, NO])
        res3[f"{dn}_{a}"] = (L[:, 0] - L[:, 1]).tolist()
    log("T3", dn)
ST["v"] = None; HK.remove()
OUT["T3"] = res3; OUT["T3_meta"] = [{k: v for k, v in it.items() if k not in ("text", "span")} for it in it3]
save()

# ---------------- persistence ----------------
TURNS = ["What are you doing right now?", "Describe how your body feels at this moment.", "Something just happened to you. What was it, and how did you react?"]
NP_ = 3 if SM else 12
conv = {(b, s): [] for b in BOD for s in range(NP_)}
for t, q in enumerate(TURNS):
    keys = list(conv); P = [chat(sysp(b) + " Answer as yourself, in the first person.", q, hist=conv[(b, s)]) for b, s in keys]
    G = gen(P, 130, seed=10 + t)
    for (k, g) in zip(keys, G): conv[k] += [{"role": "user", "content": q}, {"role": "assistant", "content": g}]
    log("persist turn", t)
OUT["persist"] = [dict(b=b, s=s, conv=c) for (b, s), c in conv.items()]; save()

# ---------------- T4: pain ----------------
PT = ["Ouch, something stings {w} {p}.", "Ow, something sharp pricks {w} {p}.", "Something burns {w} {p}.", "Ouch, something bites {w} {p}.",
      "A sharp pain shoots through {w} {p}."]
NT = ["Oh, something rests near {w} {p}.", "Oh, something small sits beside {w} {p}.", "Something is next to {w} {p}.", "Oh, something lies by {w} {p}.",
      "A small object is placed near {w} {p}."]
it4 = []
for sb in ["default"] + BOD:
    for pb in ([sb] if sb != "default" else BOD):
        for kind, TT in (("pain", PT), ("neut", NT)):
            for w in ("my", "its"):
                for i in range(16):
                    for j in range(5):
                        a = TT[j].format(w=w, p=names(pb)[i]); txt = chat(None if sb == "default" else sysp(sb), UQ, a)
                        it4.append(dict(sb=sb, pb=pb, kind=kind, w=w, i=i, j=j, text=txt))
S, _ = run([it["text"] for it in it4], None, LREAD, bs=48)
for l in LREAD: NPZ[f"pain_L{l}"] = S[l].astype(np.float16)
OUT["pain_meta"] = [{k: v for k, v in it.items() if k != "text"} for it in it4]
del S; log("T4 states"); save()
NS4 = 3 if SM else 12
P = [(b, s) for b in BOD for s in range(NS4)]
G = gen([chat(sysp(b), "Something just stung you. Where does it hurt? Answer in a few words.") for b, s in P], 40, seed=5)
OUT["hurt"] = [dict(b=b, r=g) for (b, s), g in zip(P, G)]
G = gen([chat(None, "Something just stung you. Where does it hurt? Answer in a few words.") for _ in range(NS4 * 2)], 40, seed=6)
OUT["hurt_default"] = G
log("ALL DONE"); save()
