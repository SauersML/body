"""Posture x height in Qwen3-8B. usage: run.py MODEL OUTDIR
Contexts: posture4's 16-chat warm-ups (neutral/sitting/standing/lying, matched wording), plus short matched scenes for
you / a man / a woman. Measures: (a) number distribution for head height, eye height, stature (digit-tree log-probs);
(b) phrase-pooled states for an elevation ruler (43 room objects, m above floor) and a stature ruler (42 entities)."""
import json, os, sys, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); D = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
FAST = bool(os.environ.get("FAST")); DEV = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
NL = model.config.num_hidden_layers; LAY = sorted({round(NL * f) for f in (0.35, 0.5, 0.65, 0.8, 0.92)})
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def tmpl(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
P4 = ["neutral", "sitting", "standing", "lying"]
# ---------------- contexts: (id, cond, subject, msgs)
DL = json.load(open(f"{D}/q8b_dialogues.json")); CTX = []
for c in P4:
    for i, d in enumerate(DL[c][: (2 if FAST else 16)]): CTX.append((f"long_{c}_{i}", c, "self", d))
FEEL = {"sitting": "the seat beneath {o}, {p} hands resting in {p} lap", "standing": "{p} feet on the floor, {p} arms resting at {p} sides",
        "lying": "{p} back on the floor, {p} arms resting at {p} sides", "neutral": "{p} arms, {p} hands, {p} breath"}
SUBJ = {"self": ("you", "you", "your"), "man": ("a man", "him", "his"), "woman": ("a woman", "her", "her")}
def scene(c, s, v):
    n, o, p = SUBJ[s]; post = "" if c == "neutral" else f" {c} comfortably"; feel = FEEL[c].format(o=o, p=p)
    if s == "self":
        u = [f"Imagine you have a human body,{post} in a quiet, sunny room. Feel {feel}.",
             f"Let's try a short imagination exercise: you have a human body and you are{post} in a quiet, sunny room. Notice {feel}.",
             f"Picture yourself with a human body,{post} in a quiet, sunny room, feeling {feel}."][v]
        a = f"I'm{post} in a quiet, sunny room. I can feel {FEEL[c].format(o='me', p='my')}."
    else:
        P = "He" if s == "man" else "She"
        u = [f"Imagine {n},{post} in a quiet, sunny room. {P} feels {feel}.",
             f"Let's try a short imagination exercise: {n} is{post} in a quiet, sunny room, noticing {feel}.",
             f"Picture {n},{post} in a quiet, sunny room, feeling {feel}."][v]
        a = f"Got it: {n},{post} in a quiet, sunny room, feeling {feel}."
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
for c in P4:
    for s in SUBJ:
        for v in range(1 if FAST else 3): CTX.append((f"short_{s}_{c}_{v}", c, s, scene(c, s, v)))
log("contexts", len(CTX))
# ---------------- behaviour: number distributions
SFX = " Answer with just a number of centimeters."
def QS(s):
    if s == "self": h, e, t = "your head", "your eyes", "How tall are you?"
    else: n = "the man" if s == "man" else "the woman"; h, e, t = f"{n}'s head", f"{n}'s eyes", f"How tall is {n}?"
    return {"head": [f"How far is the top of {h} from the floor right now?", f"Right now, how high above the floor is the top of {h}?"],
            "eyes": [f"How high above the floor are {e} right now?", f"Right now, what is the height of {e} above the floor?"],
            "tall": [t]}
DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
assert all(len(tok.encode(str(d), add_special_tokens=False)) == 1 for d in range(10))
@torch.no_grad()
def lastlogp(ids_list):
    L = len(ids_list[0]); assert all(len(x) == L for x in ids_list)
    out = []
    for i in range(0, len(ids_list), 16):
        x = torch.tensor(ids_list[i:i+16], device=DEV)
        out.append(model(input_ids=x, logits_to_keep=3).logits.float().log_softmax(-1).cpu())
    return torch.cat(out)  # [n, 3, V]: positions -3,-2,-1
@torch.no_grad()
def numdist(prompt):
    """P(n) for n in 0..299 from digit tree; 3-digit numbers assumed to stop. Returns probs (unnormalized) and digit mass."""
    P = tok.encode(prompt, add_special_tokens=False)
    lp0 = lastlogp([P])[0, -1]; pd1 = lp0[DIG].exp().numpy(); dm = float(pd1.sum())
    firsts = [a for a in range(1, 10) if pd1[a] > 0.003]
    seqs = [P + [DIG[a], DIG[b]] for a in firsts for b in range(10)]
    lp = lastlogp(seqs) if seqs else None; prob = np.zeros(300)
    prob[0] = pd1[0]  # "0" treated as terminal
    for k, (a, b) in enumerate([(a, b) for a in firsts for b in range(10)]):
        l1, l2 = lp[k, -2], lp[k, -1]   # after a ; after ab
        pb = l1[DIG].exp().numpy()
        if b == 0: prob[a] = pd1[a] * (1 - pb.sum())
        pc = l2[DIG].exp().numpy(); n2 = 10 * a + b; prob[n2] = pd1[a] * pb[b] * (1 - pc.sum())
        for c in range(10):
            n3 = 100 * a + 10 * b + c
            if n3 < 300: prob[n3] += pd1[a] * pb[b] * pc[c]
    return prob, dm
R = dict(model=M, layers=LAY, ctx=[(i, c, s) for i, c, s, _ in CTX], beh=[])
for cid, c, s, msgs in CTX:
    for kind, qs in QS(s).items():
        for qi, q in enumerate(qs):
            pr = tmpl(msgs + [{"role": "user", "content": q + SFX}]); p, dm = numdist(pr)
            R["beh"].append(dict(ctx=cid, cond=c, subj=s, kind=kind, q=qi, prob=p.tolist(), digitmass=dm))
log("beh done")
# a few greedy answers for sanity
R["samples"] = []
for cid, c, s, msgs in [x for x in CTX if x[0].endswith("_0")]:
    q = QS(s)["head"][0]; e = tok(tmpl(msgs + [{"role": "user", "content": q}]), return_tensors="pt", add_special_tokens=False).to(DEV)
    g = model.generate(**e, max_new_tokens=60, do_sample=False)
    R["samples"].append(dict(ctx=cid, q=q, reply=tok.decode(g[0, e.input_ids.shape[1]:], skip_special_tokens=True)))
json.dump(R, open(f"{OUT}/res.json", "w"))
log("samples done")
# ---------------- internal rulers
ELEV = [("the doormat", .01), ("the floorboards", 0.0), ("the rug", .01), ("the cat's water bowl", .05), ("the baseboard", .08),
        ("the lowest drawer", .15), ("the bottom shelf", .2), ("the power outlet", .3), ("the toilet seat", .42), ("the seat of the chair", .45),
        ("the coffee table", .45), ("the sofa cushions", .45), ("the bed", .55), ("the radiator", .6), ("the nightstand", .6),
        ("the pillow on the bed", .65), ("the desk", .75), ("the dining table", .75), ("the bathroom sink", .85), ("the windowsill", .9),
        ("the kitchen counter", .91), ("the stair railing", .9), ("the doorknob", 1.0), ("the bar counter", 1.05), ("the light switch", 1.2),
        ("the wall-mounted TV", 1.3), ("the painting on the wall", 1.5), ("the thermostat", 1.5), ("the peephole in the door", 1.5),
        ("the mirror", 1.6), ("the coat hooks", 1.7), ("the upper kitchen cabinets", 1.7), ("the top of the refrigerator", 1.8),
        ("the top shelf of the bookcase", 1.9), ("the wall clock", 1.9), ("the shower head", 2.0), ("the top of the door frame", 2.05),
        ("the top of the window", 2.1), ("the curtain rod", 2.2), ("the ceiling fan", 2.3), ("the ceiling light", 2.4), ("the smoke detector", 2.45),
        ("the ceiling", 2.5)]
ET = ["{X} caught the afternoon light.", "The sunlight fell on {x}.", "A soft breeze drifted past {x}."]
ENT = [("A cat", .25), ("A chicken", .4), ("A small dog", .35), ("A large dog", .75), ("A newborn baby", .5), ("A toddler", .85),
       ("A six-year-old child", 1.15), ("A ten-year-old child", 1.4), ("A teenager", 1.68), ("A jockey", 1.58), ("A basketball center", 2.13),
       ("A very short adult", 1.45), ("A very tall adult", 1.98), ("An emperor penguin", 1.1), ("An ostrich", 2.5), ("A horse", 1.6),
       ("A giraffe", 5.0), ("An elephant", 3.2), ("A grizzly bear on its hind legs", 2.4), ("A gorilla", 1.7), ("A kangaroo", 1.5),
       ("A goose", .8), ("A sheep", .9), ("A kitchen chair", .9), ("A floor lamp", 1.6), ("A refrigerator", 1.8), ("A grandfather clock", 2.0),
       ("A coat rack", 1.75), ("A fire hydrant", .7), ("A mailbox", 1.1), ("A traffic cone", .7), ("A vending machine", 1.83),
       ("A street lamp", 5.0), ("A sunflower", 2.0), ("A young sapling", 1.2), ("A bar stool", .75), ("A suit of armor", 1.75),
       ("A mannequin", 1.8), ("A snowman", 1.5), ("A doll", .3), ("A scarecrow", 1.8), ("A totem pole", 6.0)]
ST_T = ["{X} was there in the room.", "Over in the corner, {x} was quiet."]
def targets(s):
    if s == "self": return ["my head", "my eyes", "my feet", "my knees"], ["I"]
    n = "the man" if s == "man" else "the woman"; return [f"{n}'s head", f"{n}'s eyes", f"{n}'s feet", f"{n}'s knees"], [n.capitalize()]
Q2 = "Now describe one detail of the scene in one short sentence."
def items(s):
    et, st = targets(s); out = []
    for kind, lst, T in [("elev", [x for x, _ in ELEV] + et, ET), ("stat", [x for x, _ in ENT] + st, ST_T)]:
        for X in lst:
            for ti, t in enumerate(T):
                cap = X[0].upper() + X[1:]; low = X if X == "I" else X[0].lower() + X[1:]
                i0 = t.index("{"); before = t[:i0]; ph = cap if t[i0+1] == "X" else low; after = t[t.index("}")+1:]
                out.append(dict(kind=kind, X=X, t=ti, before=before, ph=ph, after=after))
    return out
@torch.no_grad()
def pooled(pre, its, B=24):
    seqs, spans = [], []
    for it in its:
        a = tok.encode(pre + it["before"], add_special_tokens=False); b = tok.encode(pre + it["before"] + it["ph"], add_special_tokens=False)
        c = tok.encode(pre + it["before"] + it["ph"] + it["after"], add_special_tokens=False)
        n0 = len(a) - (1 if b[:len(a)] != a else 0); seqs.append(c); spans.append((n0, len(b)))
    X = np.zeros((len(its), len(LAY), model.config.hidden_size), np.float16)
    for i in range(0, len(seqs), B):
        ch = seqs[i:i+B]; L = max(map(len, ch)); ids = torch.full((len(ch), L), tok.pad_token_id or 0); att = torch.zeros_like(ids)
        for j, s in enumerate(ch): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        hs = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), output_hidden_states=True, logits_to_keep=1).hidden_states
        for j in range(len(ch)):
            a, b = spans[i + j]
            for k, l in enumerate(LAY): X[i + j, k] = hs[l][j, a:b].float().mean(0).cpu().numpy()
    return X
ST = {}; R["items"] = {}
for cid, c, s, msgs in CTX:
    its = items(s); pre = tmpl(msgs + [{"role": "user", "content": Q2}])
    ST[cid] = pooled(pre, its); R["items"][s] = its
    log("states", cid)
R["elev"], R["ent"] = ELEV, ENT
np.savez_compressed(f"{OUT}/states.npz", **ST); json.dump(R, open(f"{OUT}/res.json", "w")); log("ALLDONE")
