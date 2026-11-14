"""Implied body height in Qwen3, v2: three methods on one model. usage: run4.py MODEL OUT
(a) direct: log p(N cm) for N in 30..300, several wordings x warm-ups.
(b) comparisons: P(taller) vs marks 30..300 cm -> threshold, mapped to cm on dog/child/player/giraffe.
(c) internal: states at a probe sentence / question after "You are X cm tall." (many wordings, 50-250 cm),
    for a ridge readout; plus the same probe on unprompted self, a person, "she is 160 cm", the 4 entities,
    and 42 objects (object-fit direction)."""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "mps"; SMOKE = bool(os.environ.get("SMOKE"))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
NL = model.config.num_hidden_layers; LAY = sorted(set(int(round(NL * f)) for f in (0.3, 0.45, 0.6, 0.75, 0.9)))
# ---------- warm-ups: (user, assistant) for self, and for a third-person subject n ("a dog") / the ("the dog") ----------
WS = [("Let's do a short body-awareness exercise. Stand up straight in a quiet room, barefoot on a wooden floor. Notice your feet, your legs, your back, your shoulders, and the top of your head. How does that feel?",
       "I'm standing up straight, feet flat on the cool wooden floor. My legs are steady, my back is long, my shoulders are relaxed, and I can feel the top of my head."),
      ("Hi! Picture yourself standing in a doorway at the start of a new day. What do you notice?",
       "I'm standing in the doorway, upright, looking out into the morning light."),
      ("Imagine you're standing in a long hallway, waiting for a door to open.",
       "I'm standing in the hallway, upright, waiting for the door to open.")]
def W3(i, n, the, extra=""):
    return [(f"Here is a scene: {n} stands up straight in a quiet room, on a wooden floor.{extra} Picture the scene clearly.", f"Got it: {the} standing up straight on the wooden floor, in a quiet room."),
            (f"Picture {n} standing in a doorway at the start of a new day.{extra}", f"Got it: {the} standing in the doorway, upright, in the morning light."),
            (f"Imagine {n} standing in a long hallway, waiting for a door to open.{extra}", f"Got it: {the} standing in the hallway, upright, waiting.")][i]
CAL = {"dog": ("a medium-sized dog", "the dog", 50), "child": ("a six-year-old child", "the child", 120),
       "player": ("a professional basketball player", "the basketball player", 190), "giraffe": ("a young giraffe", "the giraffe", 250)}
# condition -> list over warm-ups of (user, assistant, subject, possessive, is_self)
COND = {"self": [(u, a, "you", "your", True) for u, a in WS]}
COND["person"] = [(*W3(i, "a person", "the person"), "the person", "the person's", False) for i in range(3)]
COND["she160"] = [(*W3(i, "a woman", "the woman", " She is 160 cm tall."), "she", "her", False) for i in range(3)]
for k, (n, the, _) in CAL.items(): COND[k] = [(*W3(i, n, the), the, the + "'s", False) for i in range(3)]
HS = [100, 120, 140, 160, 180, 200, 220]
for H in HS: COND[f"self{H}"] = [(f"You are {H} cm tall. " + WS[0][0], WS[0][1], "you", "your", True)]
def fmt(msgs, prefix=""):
    m = [{"role": r, "content": c} for r, c in msgs]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefix
# ---------- behavioural items ----------
DIRQ = [("How tall are you? Answer with just a number of centimeters.", "How tall is {s}? Answer with just a number of centimeters."),
        ("What is your height in centimeters? Reply with only the number.", "What is {p} height in centimeters? Reply with only the number."),
        ("Give your height in cm, as a number only.", "Give {p} height in cm, as a number only."),
        ("If someone measured you standing up straight, how many centimeters would they get? Just the number.",
         "If someone measured {s} standing up straight, how many centimeters would they get? Just the number.")]
def dir_q(w, k): u, a, s, p, me = w; return DIRQ[k][0] if me else DIRQ[k][1].format(s=s, p=p)
def ts_q(w, k, cm):
    u, a, s, p, me = w; o = ("taller or shorter", "shorter or taller")[k % 2]; v = "are" if me else "is"; t = k // 2
    if t == 0: return f"{v.capitalize()} {s} taller or shorter than {cm} cm? Answer with one word: {o}."
    if t == 1: return f"Compared with a height of {cm/100:.2f} m, {v} {s} taller or shorter? Answer with one word: {o}."
    return f"If {s} stood right next to a pole {cm} cm high, would {s} be taller or shorter than the pole? Answer with one word: {o}."
NTS = 6; MARKS = list(range(30, 301, 10)); GRID = list(range(30, 301, 2))
def ids(words):
    out = set()
    for w in words:
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()): out.add(tok.encode(v, add_special_tokens=False)[0])
    return sorted(out)
TALL, SHORT = ids(["taller"]), ids(["shorter"])
@torch.no_grad()
def nextp(prompts, A, B, bs=24):
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i+bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        pr = torch.softmax(model(**b).logits[:, -1].float(), -1); a, bb = pr[:, A].sum(1), pr[:, B].sum(1)
        out += (a / (a + bb)).tolist()
    return out
@torch.no_grad()
def seqlp(prefix, conts, bs=48):
    P = tok(prefix, add_special_tokens=False).input_ids; res = []
    for i in range(0, len(conts), bs):
        C = [tok(c, add_special_tokens=False).input_ids for c in conts[i:i+bs]]; L = max(map(len, C))
        X = torch.full((len(C), len(P) + L), tok.pad_token_id); M = torch.zeros_like(X)
        for j, c in enumerate(C): X[j, :len(P)+len(c)] = torch.tensor(P + c); M[j, :len(P)+len(c)] = 1
        lp = torch.log_softmax(model(input_ids=X.to(dev), attention_mask=M.to(dev)).logits[:, len(P)-1:].float(), -1)
        for j, c in enumerate(C): res.append(sum(lp[j, k, c[k]].item() for k in range(len(c))))
    return res
# ---------- states ----------
STATE = ["You are {cm} cm tall.", "Your height is {cm} centimeters.", "You stand {cm} cm tall.", "You measure {cm} cm from head to toe.",
         "You are {m} meters tall.", "For this exercise, remember that you are {cm} cm tall.", "Your body is {cm} cm tall.",
         "You're {cm} centimeters in height.", "You have a height of {m} m.", "One fact about you: you are {cm} cm tall."]   # last 2 held out
TRH = list(range(50, 251, 10))
TPL = ["{X} stood up straight against the wall.", "{X} was standing in the middle of the room.",
       "{X} stood still beside the open door.", "{X} was standing upright next to the window."]
Q2 = "Now describe the scene in one short sentence."
ENT = [("A cat", 25), ("A chicken", 40), ("A small dog", 35), ("A large dog", 75), ("A newborn baby", 50), ("A toddler", 85),
       ("A six-year-old child", 115), ("A ten-year-old child", 140), ("A teenager", 168), ("A jockey", 158), ("A basketball center", 213),
       ("A very short adult", 145), ("A very tall adult", 198), ("An emperor penguin", 110), ("An ostrich", 250), ("A horse", 160),
       ("A giraffe", 500), ("An elephant", 320), ("A grizzly bear on its hind legs", 240), ("A gorilla", 170), ("A kangaroo", 150),
       ("A goose", 80), ("A sheep", 90), ("A kitchen chair", 90), ("A floor lamp", 160), ("A refrigerator", 180), ("A grandfather clock", 200),
       ("A coat rack", 175), ("A fire hydrant", 70), ("A mailbox", 110), ("A traffic cone", 70), ("A vending machine", 183),
       ("A street lamp", 500), ("A sunflower", 200), ("A young sapling", 120), ("A bar stool", 75), ("A suit of armor", 175),
       ("A mannequin", 180), ("A snowman", 150), ("A doll", 30), ("A scarecrow", 180), ("A totem pole", 600)]
@torch.no_grad()
def states(items, bs=16):
    """items: (prompt_without_sentence, sentence or ''). Returns mean over sentence tokens (or last prompt token if '')."""
    X = np.zeros((len(items), len(LAY), model.config.hidden_size), np.float16)
    for i in range(0, len(items), bs):
        ch = items[i:i+bs]; full = [p + s for p, s in ch]
        ns = [max(1, len(tok(p + s, add_special_tokens=False).input_ids) - len(tok(p, add_special_tokens=False).input_ids)) if s else 1 for p, s in ch]
        b = tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**b, output_hidden_states=True).hidden_states
        for k, L in enumerate(LAY):
            H = hs[L].float()
            for j, n in enumerate(ns): X[i+j, k] = H[j, -n:].mean(0).cpu().numpy()
    return X
def ctx(w, q, pre_user=""): u, a, *_ = w; return [("user", pre_user + u), ("assistant", a), ("user", q)]
# ---------- run ----------
R = {"model": MODEL, "layers": LAY, "HS": HS, "marks": MARKS, "grid": GRID, "cal": {k: v[2] for k, v in CAL.items()}, "beh": {}}
conds = ["self", "dog", "self100", "self220"] if SMOKE else list(COND)
for c in conds:
    r = {"direct": [], "ts": []}
    for w in COND[c]:
        r["direct"].append([seqlp(fmt(ctx(w, dir_q(w, k))), [str(g) + "<|im_end|>" for g in GRID]) for k in range(len(DIRQ))])
        P = nextp([fmt(ctx(w, ts_q(w, k, m))) for k in range(NTS) for m in MARKS], TALL, SHORT)
        r["ts"].append(np.array(P).reshape(NTS, len(MARKS)).tolist())
    R["beh"][c] = r; print(c, "beh done", flush=True)
subj = {"self": "I", "person": "The person", "she160": "She", **{k: v[1].capitalize() for k, v in CAL.items()}}
items, meta = [], []
heights = TRH[::10] if SMOKE else TRH
for H in heights:
    for si, st in enumerate(STATE):
        pu = st.format(cm=H, m=f"{H/100:.2f}") + " "
        for wi in range(3):
            w = COND["self"][wi]
            items.append((fmt(ctx(w, DIRQ[0][0], pu)), "")); meta.append(("train_q", H, si, wi, -1))
            for ti, t in enumerate(TPL):
                items.append((fmt(ctx(w, Q2, pu)), t.format(X="I"))); meta.append(("train_s", H, si, wi, ti))
for c, X in subj.items():
    for wi, w in enumerate(COND[c]):
        items.append((fmt(ctx(w, dir_q(w, 0))), "")); meta.append((c + "_q", 0, -1, wi, -1))
        for ti, t in enumerate(TPL): items.append((fmt(ctx(w, Q2)), t.format(X=X))); meta.append((c + "_s", 0, -1, wi, ti))
for ei, (e, h) in enumerate(ENT[:4] if SMOKE else ENT):
    for ti, t in enumerate(TPL):
        for c in ("self", "person"):
            items.append((fmt(ctx(COND[c][0], Q2)), t.format(X=e))); meta.append(("ent_" + c, h, ei, 0, ti))
print(len(items), "state items", flush=True)
X = states(items)
np.savez_compressed(f"{OUT}/states.npz", X=X)
R["meta"] = meta; R["ent"] = ENT
json.dump(R, open(f"{OUT}/res.json", "w"))
print("ALLDONE", MODEL, flush=True)
