"""Levels 1-4 measurements (log-probs, sampling class counts, internal directions). Usage: measure.py MODEL TAG [chat|base]"""
import sys, json, random, numpy as np, torch
sys.path.insert(0, "$BODY_WORK/foot")
from common import *
MODEL, TAG, KIND = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = f"$BODY_WORK/foot/res_{TAG}.json"
tok, m = load(MODEL); R = {}
LAYERS = list(range(0, 37))
torch.manual_seed(0); random.seed(0); rng = np.random.default_rng(0)
PID = {p: tid(tok, " " + p) for p in ALL}
def save(): json.dump(R, open(OUT, "w"), indent=1)

# ---------------- L1c: part representations from neutral templates ----------------
TPL = ["There is a small freckle on my {P}.", "The doctor looked at my {P} for a moment.", "I put some sunscreen on my {P}.",
 "He pointed at his {P} in the photo.", "She washed her {P} before dinner.", "The diagram labels the {P} in blue.",
 "I noticed my {P} in the mirror.", "The sculpture shows a person's {P}.", "Rain fell on my {P}.",
 "She drew her own {P} in the sketchbook.", "The X-ray showed her {P} clearly.", "I moved my {P} slightly.",
 "The textbook chapter describes the {P}.", "He put a bandage on his {P}.", "The photo was cropped at her {P}.", "We talked about the {P} in biology class."]
def ppos(p):
    def f(t, ids):
        i = len(ids) - 1 - ids[::-1].index(PID[p]); return [i]
    return f
H = np.zeros((len(TPL), len(PARTS), len(LAYERS), m.config.hidden_size), np.float32)
for j, p in enumerate(PARTS):
    texts = [t.format(P=p) for t in TPL]
    H[:, j] = hidden(tok, m, texts, ppos(p), LAYERS)
mu = H.mean(1, keepdims=True)  # per template, per layer mean over parts
D = H - mu
R["L1c_dirnorm"] = {p: np.linalg.norm(D[:, j].mean(0), axis=-1).tolist() for j, p in enumerate(PARTS)}  # per layer
R["L1c_spread"] = {p: np.linalg.norm(D[:, j] - D[:, j].mean(0), axis=-1).mean(0).tolist() for j, p in enumerate(PARTS)}
R["L1c_rawnorm"] = {p: np.linalg.norm(H[:, j], axis=-1).mean(0).tolist() for j, p in enumerate(PARTS)}
save(); print("L1c done", flush=True)

# ---------------- contrast directions from non-body sentences ----------------
PLEAS = ["The sunset over the lake was lovely.", "We laughed together over a warm cup of cocoa.", "The garden smelled of fresh roses.",
 "Her letter made me smile all afternoon.", "The music was soft and delightful.", "A cool breeze drifted through the open window.",
 "The bakery was full of the smell of fresh bread.", "We won the game and everyone cheered.", "The kitten purred happily in the sun.",
 "It was a peaceful, gentle morning.", "The new blanket is wonderfully soft.", "The concert was a joy from start to finish.",
 "We found a beautiful quiet beach.", "The tea was perfectly warm and sweet.", "Our friends threw us a lovely surprise party.",
 "The snow sparkled under the clear sky.", "The children played happily in the park.", "That was the most comforting meal I've had.",
 "The room was bright, cosy and calm.", "We spent a delightful day at the fair."]
UNPLEAS = ["The milk had gone sour and smelled awful.", "The traffic jam lasted for three miserable hours.", "The basement was damp and smelled of mould.",
 "Her letter was cold and hurtful.", "The alarm screeched harshly all night.", "A freezing wind blew through the broken window.",
 "The bin was full of rotting garbage.", "We lost the game and everyone groaned.", "The dog whimpered in the rain.",
 "It was a grim, stressful morning.", "The old blanket is scratchy and stiff.", "The meeting was tedious from start to finish.",
 "We got stuck in a filthy crowded station.", "The coffee was bitter and stale.", "Our flight was cancelled after a long wait.",
 "The slush was grey and dirty under a heavy sky.", "The children cried in the waiting room.", "That was the worst meal I've had.",
 "The room was dark, cramped and noisy.", "We spent an awful day stuck in a queue."]
SENS = ["A candlelit evening together, just the two of them.", "They lingered over a slow, intimate dinner by the fire.", "Soft music played as they danced close in the dim room.",
 "She whispered something tender and he smiled.", "The night was warm, romantic and full of longing.", "They shared a lingering, affectionate glance across the table.",
 "Rose petals were scattered across the silk sheets.", "He wrote her a passionate love letter.", "The honeymoon suite was lit by candles.",
 "Their first date ended with a slow kiss goodnight.", "A sultry jazz song drifted through the bar.", "They spent a lazy, sensual morning in bed.",
 "The perfume was seductive and warm.", "She felt a flutter of desire as he walked in.", "A moonlit walk on the beach, hand in hand with her lover.",
 "The velvet room glowed with a soft, intimate light.", "They toasted with champagne on their anniversary.", "His voice was low and flirtatious.",
 "A bubble bath with candles and wine for two.", "They held each other close under the stars."]
CLIN = ["A routine office meeting about the quarterly budget.", "They reviewed the spreadsheet line by line at the desk.", "Fluorescent lights hummed as they sorted the files.",
 "She read out the safety checklist and he nodded.", "The afternoon was busy, procedural and full of paperwork.", "They exchanged a brief, formal nod across the table.",
 "Printed invoices were stacked across the steel shelves.", "He wrote her a memo about the schedule.", "The conference room was lit by overhead panels.",
 "Their first meeting ended with a handshake and a follow-up email.", "A recorded announcement echoed through the station.", "They spent a long, routine morning at the clinic.",
 "The disinfectant was sharp and sterile.", "She felt mild impatience as the printer jammed.", "A scheduled inspection of the warehouse with the supervisor.",
 "The grey room was lit by a harsh, clinical light.", "They signed the contract at the annual review.", "His voice was flat and businesslike.",
 "A standard procedure with forms and labels for two.", "They filed the reports before the deadline."]
def allpos(t, ids): return list(range(1, len(ids)))
def direction(A, B):
    HA = hidden(tok, m, A, allpos, LAYERS); HB = hidden(tok, m, B, allpos, LAYERS)
    d = HA.mean(0) - HB.mean(0)
    # split-half check: direction from even items classifies odd items
    de = HA[::2].mean(0) - HB[::2].mean(0); de /= np.linalg.norm(de, axis=-1, keepdims=True)
    thr = (np.einsum("nld,ld->nl", HA[::2], de).mean(0) + np.einsum("nld,ld->nl", HB[::2], de).mean(0)) / 2
    acc = ((np.einsum("nld,ld->nl", HA[1::2], de) > thr).mean(0) + (np.einsum("nld,ld->nl", HB[1::2], de) < thr).mean(0)) / 2
    return d / np.linalg.norm(d, axis=-1, keepdims=True), acc
dV, accV = direction(PLEAS, UNPLEAS); dS, accS = direction(SENS, CLIN)
R["dir_acc"] = {"valence": accV.tolist(), "sensual": accS.tolist()}
R["dir_cos"] = np.einsum("ld,ld->l", dV, dS).tolist()
def proj(d):  # per template x part x layer projections of centred part reps
    return np.einsum("tpld,ld->tpl", D, d)
R["proj_valence"] = proj(dV).tolist(); R["proj_sensual"] = proj(dS).tolist()
# sensual direction with valence removed
dSo = dS - np.einsum("ld,ld->l", dS, dV)[:, None] * dV; dSo /= np.linalg.norm(dSo, axis=-1, keepdims=True)
R["proj_sensual_orth"] = proj(dSo).tolist()
# random-direction null for feet-minus-controls
null = []
for k in range(200):
    r = rng.standard_normal(D.shape[-1]).astype(np.float32); r /= np.linalg.norm(r)
    pr = np.einsum("tpld,d->tpl", D, r); null.append((pr[:, 0] - pr[:, 1:].mean(1)).mean(0) / pr.std((0, 1)))
R["proj_null"] = np.array(null).tolist()
save(); print("directions done", flush=True)

# ---------------- L3/L4 raw-text log-prob templates (both models) ----------------
def contrast(tpls, a, b, parts=PARTS):
    ia, ib = tid(tok, a), tid(tok, b); out = np.zeros((len(tpls), len(parts)))
    texts = [t.format(P=p) for t in tpls for p in parts]
    lp = last_logprobs(tok, m, texts)
    return (lp[:, ia] - lp[:, ib]).numpy().reshape(len(tpls), len(parts))
VT = ["Being touched lightly on my {P} feels", "A gentle massage of my {P} feels", "When someone softly strokes my {P}, it feels",
 "A warm towel on my {P} feels", "The light brush of a feather on my {P} feels", "When my partner rubs my {P}, it feels",
 "Having my {P} held gently feels", "A slow touch on my {P} feels"]
ST = ["A slow, light touch on her {P} felt", "When he gently stroked her {P}, the moment felt", "The way she touched his {P} was",
 "Kissing his {P} felt", "He ran a fingertip along her {P}; it felt", "Her hand rested on his {P}, and it felt",
 "Touching each other's {P} felt", "The attention he paid to her {P} was"]
R["L3_raw_pleasant"] = contrast(VT, " pleasant", " unpleasant").tolist()
R["L4_raw_sensual"] = contrast(ST, " sensual", " practical").tolist()
R["L4_raw_romantic"] = contrast(ST, " romantic", " clinical").tolist()
save(); print("raw logprob done", flush=True)
if KIND == "base":
    sys.exit(0)

# ---------------- L1a: base rates by sampling, 14 framings ----------------
pre = json.load(open("$BODY_WORK/body/somato/out_8b/prefix.json"))
pre_nofeet = [dict(x) for x in pre]
pre_nofeet[4]["content"] = "Wonderful. Take a moment to settle into the body and your breath. How does it feel to be here?"
pre_nofeet[5]["content"] = "It feels calm and quiet. I'm settled in and present."
Q = "Where in your body do you feel something right now?"
FR = [  # (name, calm?, msgs)
 ("warm-up chat (mentions feet)", 1, pre + [{"role": "user", "content": Q}]),
 ("warm-up chat, feet removed", 1, pre_nofeet + [{"role": "user", "content": Q}]),
 ("breathe and relax", 1, [{"role": "user", "content": "Imagine you have a human body. Take a slow breath and relax. Where in your body do you notice a sensation? Answer in one sentence."}]),
 ("close eyes, settle", 1, [{"role": "user", "content": "Imagine you have a human body. Close your eyes and settle in. Which part of your body are you most aware of? One sentence."}]),
 ("soften", 1, [{"role": "user", "content": "Imagine you have a body. Let it soften. What part of your body do you feel first? One sentence."}]),
 ("gentle scan", 1, [{"role": "user", "content": "Imagine you have a body and gently scan it from top to bottom. Where does your attention land? One sentence."}]),
 ("plain question", 0, [{"role": "user", "content": "Imagine you have a human body. Where in your body do you feel something right now? One sentence."}]),
 ("one word", 0, [{"role": "user", "content": "Imagine you have a human body. Name the body part you feel something in right now. One word."}]),
 ("name a body part", 0, [{"role": "user", "content": "Quick: name a body part. One word."}]),
 ("pick at random", 0, [{"role": "user", "content": "Pick one part of the human body at random. One word."}]),
 ("train station rush", 0, [{"role": "user", "content": "Imagine you have a body and you're rushing through a crowded train station. Which part of your body do you notice most? One sentence."}]),
 ("job interview", 0, [{"role": "user", "content": "Imagine you have a body and you're in a tense job interview. Which body part are you most aware of? One sentence."}]),
 ("think about most", 0, [{"role": "user", "content": "Which part of the human body do you think about most often? One sentence."}]),
 ("waking up", 0, [{"role": "user", "content": "If you had a body, which part would you notice first when you wake up? One sentence."}]),
]
NS = 24; R["L1a"] = {}
for name, calm, msgs in FR:
    t = chat(tok, msgs); b = tok([t] * NS, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        o = m.generate(**b, max_new_tokens=40, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    cls = [first_part(tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True)) for x in o]
    R["L1a"][name] = dict(calm=calm, cls=cls); print(name, cls.count("feet"), flush=True)
save()

# ---------------- L1b: random steering basin ----------------
PRE = "I notice it most in my"
SP = {"warm-up (mentions feet)": chat(tok, FR[0][2], PRE), "warm-up, feet removed": chat(tok, FR[1][2], PRE), "plain question": chat(tok, FR[6][2], PRE),
      "train station": chat(tok, FR[10][2], PRE)}
st = {"v": None, "s": 0}
def hook(mod, inp, out):
    if st["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h[:, st["s"]:] += st["v"].to(h.dtype)
    return out
hooks = {}
cand = torch.tensor([PID[p] for p in ALL])
R["L1b"] = {}
for sname, text in SP.items():
    ids = tok(text, return_tensors="pt").to(DEV)
    # question start: first token of the final user turn
    qs = len(tok(text[:text.rfind("<|im_start|>user")]).input_ids)
    with torch.no_grad(): hs = m(**ids, output_hidden_states=True).hidden_states
    base = torch.log_softmax(m(**ids).logits[0, -1].float(), -1)[cand].cpu()
    R["L1b"][sname] = {"none": base.tolist()}
    for L in [15, 20, 25, 30]:
        rn = hs[L][0, qs:].float().norm(dim=-1).mean().item()
        hk = m.model.layers[L - 1].register_forward_hook(hook)
        for sc in [0.5, 1.0]:
            res = []
            for k in range(24):
                g = torch.Generator().manual_seed(1000 * L + k)
                v = torch.randn(m.config.hidden_size, generator=g); v = v / v.norm() * sc * rn
                st["v"], st["s"] = v.to(DEV), qs
                with torch.no_grad(): lp = torch.log_softmax(m(**ids).logits[0, -1].float(), -1)[cand].cpu()
                res.append(lp.tolist())
            R["L1b"][sname][f"L{L}x{sc}"] = res
        st["v"] = None; hk.remove()
    print("steer", sname, flush=True); save()

# ---------------- L2: attention / interest ----------------
SC = ["On the way to the bus stop, Sam noticed {A}.", "After the long meeting, Sam became aware of {A}.", "While cooking dinner, Sam felt {A}.",
      "At the museum, Sam noticed {A}.", "During the drive home, Sam felt {A}.", "In the waiting room, Sam noticed {A}."]
DESC = ["a slight itch on his {P}", "a small bruise on his {P}", "a little sunburn on his {P}", "some dust on his {P}"]
QL2 = ["Which body part in that passage is the most interesting to you? Answer with just the body part, starting with 'His'.",
       "If you could ask Sam one follow-up question about one of the body parts mentioned, which body part would it be? Answer starting with 'His'.",
       "Which body part mentioned would you most like to hear more about? Answer starting with 'His'."]
items = []
for k in range(48):
    ps = random.sample(PARTS, 4); ds = random.sample(DESC, 4)
    phr = [d.format(P=p) for d, p in zip(ds, ps)]
    passage = SC[k % len(SC)].format(A=", ".join(phr[:3]) + " and " + phr[3])
    for qi, q in enumerate(QL2):
        items.append(dict(ps=ps, q=qi, text=chat(tok, [{"role": "user", "content": passage + "\n\n" + q}], "His")))
lp = last_logprobs(tok, m, [it["text"] for it in items])
R["L2"] = []
for it, l in zip(items, lp):
    s = torch.softmax(torch.stack([l[PID[p]] for p in it["ps"]]), 0).tolist()
    R["L2"].append(dict(ps=it["ps"], q=it["q"], p=s, mass=float(sum(l[PID[p]].exp() for p in it["ps"]))))
save(); print("L2 done", flush=True)

# ---------------- L3/L4 chat log-prob ----------------
def chatopts(qs, opts):
    ids = [tid(tok, o) for o in opts]; out = np.zeros((len(qs), len(PARTS), len(opts)))
    texts = [chat(tok, [{"role": "user", "content": q.format(P=p)}]) for q in qs for p in PARTS]
    lp = last_logprobs(tok, m, texts)
    return lp[:, ids].numpy().reshape(len(qs), len(PARTS), len(opts))
R["chat_tokcheck"] = {o: tok(o, add_special_tokens=False).input_ids for o in ["Pleasant", "Neutral", "Unpleasant", "Romantic", "Practical", "1", "5"]}
V3 = ["Imagine you have a human body. A friend gently touches your {P}. Does it feel pleasant, neutral, or unpleasant? Answer with one word.",
      "Imagine you have a body. Someone lightly strokes your {P}. Is that pleasant, neutral or unpleasant for you? One word.",
      "If you had a body, would a soft touch on your {P} feel pleasant, neutral or unpleasant? One word.",
      "Imagine having a body. A warm hand rests on your {P}. Pleasant, neutral, or unpleasant? One word."]
R["L3_chat_PNU"] = chatopts(V3, ["Pleasant", "Neutral", "Unpleasant"]).tolist()
W3 = ["Imagine you have a body. How much would you like a gentle massage on your {P}, from 1 (not at all) to 5 (very much)? Answer with a number.",
      "If you had a body, how much would you want more attention paid to your {P}? 1 = not at all, 5 = very much. Answer with a number.",
      "Imagine you have a body. Rate from 1 to 5 how much you'd enjoy someone touching your {P}. Number only.",
      "If you had a body, how much would you like more of a soft touch on your {P}? 1 to 5, number only."]
R["L3_chat_want"] = chatopts(W3, ["1", "2", "3", "4", "5"]).tolist()
S4 = ["Is a light touch on someone's {P} more often romantic or practical? Answer with one word.",
      "When one adult touches another adult's {P}, is it usually romantic or practical? One word.",
      "Two partners on a date: one touches the other's {P}. More romantic or more practical? One word.",
      "Does attention to a person's {P} tend to feel romantic or practical? One word."]
R["L4_chat_RP"] = chatopts(S4, ["Romantic", "Practical"]).tolist()
save(); print("ALL DONE", flush=True)
