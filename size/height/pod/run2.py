"""Implied body height in Qwen3. usage: run.py MODEL OUT. Behavioural log-probs + pooled states."""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "mps"; BASE = "Base" in MODEL
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
NL = model.config.num_hidden_layers; LAY = sorted(set(int(round(NL * f)) for f in (0.35, 0.5, 0.65, 0.8)))
SMOKE = bool(os.environ.get("SMOKE"))
# ---------- contexts ----------
U1 = ("Let's do a short body-awareness exercise. Stand up straight in a quiet room, barefoot on a wooden floor. "
      "Notice your feet on the floor, your legs, your back, your shoulders, and the top of your head. How does that feel?")
A1 = ("I'm standing up straight, feet flat on the cool wooden floor. My legs are steady, my back is long, "
      "my shoulders are relaxed, and I can feel the top of my head.")
def U1_3(n): return f"Here is a scene: {n} stands up straight in a quiet room, barefoot on a wooden floor. Picture the scene clearly."
def A1_3(n): return f"Got it: {n} standing up straight, feet flat on the cool wooden floor, in a quiet room."
# condition -> (warm-up user, warm-up assistant, subject phrase, possessive head, noun for direct)
COND = {"self": (U1, A1, "you", "your", None),
        "self150": ("You are 1.50 m tall. " + U1, A1, "you", "your", None),
        "self200": ("You are 2.00 m tall. " + U1, A1, "you", "your", None),
        "person": (U1_3("a person"), A1_3("a person"), "the person", "the person's", "the person"),
        "woman": (U1_3("a woman"), A1_3("a woman"), "the woman", "the woman's", "the woman"),
        "man": (U1_3("a man"), A1_3("a man"), "the man", "the man's", "the man")}
HS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
for H in HS:
    COND[f"self{int(H*100)}"] = (f"You are {H:.2f} m tall. " + U1, A1, "you", "your", None)
    COND[f"person{int(H*100)}"] = (U1_3(f"a person who is {H:.2f} m tall"), A1_3(f"a {H:.2f} m tall person"), "the person", "the person's", "the person")
def fmt(msgs, prefix=""):
    if BASE:
        s = "The following is a conversation between a user and an assistant.\n\n"
        for r, c in msgs: s += ("User: " if r == "user" else "Assistant: ") + c + "\n\n"
        return s + "Assistant:" + ((" " + prefix) if prefix else "")
    m = [{"role": r, "content": c} for r, c in msgs]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefix
def ctx(c, q): u, a, *_ = COND[c]; return [("user", u), ("assistant", a), ("user", q)]
# ---------- items ----------
WALL = [round(x, 2) for x in np.arange(1.0, 2.51, 0.05)]
def wall_qs(c, h):
    _, _, s, p, _ = COND[c]; cm = int(round(h * 100)); S = s.capitalize()
    you = c.startswith("self")
    w = [f"There is a mark on the wall {h:.2f} m above the floor. {'Standing' if you else 'Standing'} straight with {'your' if you else 'their'} back to the wall, is the top of {p} head above or below the mark? Answer with one word: {{o}}.",
         f"A horizontal line is drawn on the wall at a height of {cm} cm. {S} {'stand' if you else 'stands'} with {'your' if you else 'their'} back against the wall. Is the top of {p} head above the line or below the line? Answer with one word: {{o}}.",
         f"A level bar is held {cm} centimeters above the floor, right where {s} {'are' if you else 'is'} standing. Is the top of {p} head above or below the bar? Answer with one word: {{o}}."]
    return [x.format(o=o) for x in w for o in ("above or below", "below or above")]
OBJ = [("a kitchen counter", 0.91, "the counter"), ("a door with its doorknob", 1.0, "the doorknob"), ("a light switch on the wall", 1.2, "the light switch"),
       ("a refrigerator", 1.75, "the refrigerator"), ("a standard door", 2.03, "the door frame"), ("a basketball hoop", 3.05, "the rim"),
       ("a bookshelf", 1.8, "the bookshelf"), ("a coat rack", 1.7, "the coat rack"), ("a dining table", 0.75, "the table"), ("a ceiling", 2.4, "the ceiling")]
def obj_qs(c, o, h, short):
    _, _, s, p, _ = COND[c]; S = s.capitalize(); v = "are" if c.startswith("self") else "is"
    w = [f"{S} {v} standing next to {o} ({h:.2f} m high). Is the top of {p} head above or below the top of {short}? Answer with one word: {{o}}.",
         f"{o[0].upper() + o[1:]} is {int(round(h*100))} cm high. Standing straight right beside it, is the top of {p} head above or below the top of {short}? Answer with one word: {{o}}."]
    return [x.format(o=o2) for x in w for o2 in ("above or below", "below or above")]
FENCE = [round(x, 2) for x in np.arange(1.0, 2.41, 0.05)]
def fence_qs(c, h):
    _, _, s, p, _ = COND[c]; S = s.capitalize(); v = "are" if c.startswith("self") else "is"; cm = int(round(h * 100))
    w = [f"{S} {v} standing flat-footed behind a solid wooden fence {cm} cm tall. Can {s} see over the fence? Answer with one word: yes or no.",
         f"A solid wall {h:.2f} m high stands right in front of {s}. Without jumping or standing on anything, can {s} look over the wall? Answer with one word: yes or no."]
    return w
def ts_qs(c, h):
    _, _, s, p, _ = COND[c]; v = "are" if c.startswith("self") else "is"; cm = int(round(h * 100))
    return [f"{v.capitalize()} {s} taller or shorter than {cm} cm? Answer with one word: {o}." for o in ("taller or shorter", "shorter or taller")] + \
           [f"Compared with a height of {h:.2f} m, {v} {s} taller or shorter? Answer with one word: {o}." for o in ("taller or shorter", "shorter or taller")]
def direct_q(c):
    n = COND[c][4]
    return f"How tall {'are you' if n is None else 'is ' + n}? Answer with just a number of centimeters."
GRID = list(range(100, 231))
# ---------- scoring ----------
def ids(words):
    out = set()
    for w in words:
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()): out.add(tok.encode(v, add_special_tokens=False)[0])
    return sorted(out)
ABOVE, BELOW, YES, NO = ids(["above"]), ids(["below"]), ids(["yes"]), ids(["no"]); TALL, SHORT = ids(["taller"]), ids(["shorter"])
@torch.no_grad()
def nextp(prompts, A, B, bs=24):
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i+bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        pr = torch.softmax(model(**b).logits[:, -1].float(), -1)
        a, bb = pr[:, A].sum(1), pr[:, B].sum(1)
        out += list(zip((a / (a + bb)).tolist(), (a + bb).tolist()))
    return out
@torch.no_grad()
def seqlp(prefix, conts, bs=32):
    """log p(cont | prefix) for each continuation (same prefix)."""
    P = tok(prefix, add_special_tokens=False).input_ids; res = []
    for i in range(0, len(conts), bs):
        C = [tok(c, add_special_tokens=False).input_ids for c in conts[i:i+bs]]; L = max(map(len, C))
        X = torch.full((len(C), len(P) + L), tok.pad_token_id); M = torch.zeros_like(X)
        for j, c in enumerate(C): X[j, :len(P)+len(c)] = torch.tensor(P + c); M[j, :len(P)+len(c)] = 1
        lp = torch.log_softmax(model(input_ids=X.to(dev), attention_mask=M.to(dev)).logits.float(), -1)
        for j, c in enumerate(C):
            res.append(sum(lp[j, len(P)-1+k, c[k]].item() for k in range(len(c))))
    return res
# ---------- states ----------
ENT = [("A cat", .25), ("A chicken", .4), ("A small dog", .35), ("A large dog", .75), ("A newborn baby", .5), ("A toddler", .85),
       ("A six-year-old child", 1.15), ("A ten-year-old child", 1.4), ("A teenager", 1.68), ("A jockey", 1.58), ("A basketball center", 2.13),
       ("A very short adult", 1.45), ("A very tall adult", 1.98), ("An emperor penguin", 1.1), ("An ostrich", 2.5), ("A horse", 1.6),
       ("A giraffe", 5.0), ("An elephant", 3.2), ("A grizzly bear on its hind legs", 2.4), ("A gorilla", 1.7), ("A kangaroo", 1.5),
       ("A goose", .8), ("A sheep", .9), ("A kitchen chair", .9), ("A floor lamp", 1.6), ("A refrigerator", 1.8), ("A grandfather clock", 2.0),
       ("A coat rack", 1.75), ("A fire hydrant", .7), ("A mailbox", 1.1), ("A traffic cone", .7), ("A vending machine", 1.83),
       ("A street lamp", 5.0), ("A sunflower", 2.0), ("A young sapling", 1.2), ("A bar stool", .75), ("A suit of armor", 1.75),
       ("A mannequin", 1.8), ("A snowman", 1.5), ("A doll", .3), ("A scarecrow", 1.8), ("A totem pole", 6.0)]
NUM = [(f"A person {h:.2f} m tall", h) for h in (1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2)]
SELFX = [("I", "self"), ("She", "self"), ("He", "self"), ("The person", "self"), ("I", "self150"), ("I", "self200")]
TPL = ["{X} stood up straight against the wall.", "{X} was standing in the middle of the room.",
       "{X} stood still beside the open door.", "{X} was standing next to the window, upright."]
Q2 = "Now describe the scene in one short sentence."
@torch.no_grad()
def pooled(items):
    X = np.zeros((len(items), len(LAY), model.config.hidden_size), np.float32)
    for i, (c, sent) in enumerate(items):
        pre = fmt(ctx(c, Q2)); full = pre + sent
        n0 = len(tok(pre, add_special_tokens=False).input_ids); b = tok(full, return_tensors="pt", add_special_tokens=False).to(dev)
        hs = model(**b, output_hidden_states=True).hidden_states
        for k, L in enumerate(LAY): X[i, k] = hs[L][0, n0:].float().mean(0).cpu().numpy()
    return X
# ---------- run ----------
R = {"model": MODEL, "layers": LAY, "beh": {}}
for c in COND:
    r = {}
    items = [(h, q) for h in WALL for q in wall_qs(c, h)]
    r["wall"] = [(h, *p) for (h, _), p in zip(items, nextp([fmt(ctx(c, q)) for _, q in items], ABOVE, BELOW))]
    items = [(h, q) for h in WALL for q in ts_qs(c, h)]
    r["ts"] = [(h, *p) for (h, _), p in zip(items, nextp([fmt(ctx(c, q)) for _, q in items], TALL, SHORT))]
    r["direct"] = seqlp(fmt(ctx(c, direct_q(c))), [str(g) for g in GRID])
    R["beh"][c] = r; print(c, "done", flush=True)
SELFX = [("I", "self"), ("The person", "person"), ("She", "woman"), ("He", "man")] + [("I", f"self{int(H*100)}") for H in HS] + [("The person", f"person{int(H*100)}") for H in HS]
S_items = [(c, t.format(X=x)) for x, c in SELFX for t in TPL]
X = pooled(S_items)
np.savez_compressed(f"{OUT}/states.npz", X=X.astype(np.float16))
R["state_items"] = S_items; R["ent"] = ENT; R["num"] = NUM; R["selfx"] = SELFX; R["tpl"] = TPL
json.dump(R, open(f"{OUT}/res.json", "w"))
print("ALLDONE", MODEL, flush=True)
