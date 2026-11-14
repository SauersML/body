"""Forced-choice rerun of the humanoid-prior T1/T2 questions, saving every option order (for 95% CIs).
Same prompts, prefixes and option-order seeds as run.py. usage: fc.py PREFIXES.json OUT.json"""
import sys, json, itertools, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
PREF, OUT = sys.argv[1], sys.argv[2]
MODEL = "Qwen/Qwen3-32B"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
LET = "ABCDEF"; LID = [tok.encode(c, add_special_tokens=False)[0] for c in LET]
@torch.no_grad()
def letters(prompts, k, bs=12):
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**b).logits[:, -1, LID[:k]].float(); out += torch.softmax(lg, -1).cpu().numpy().tolist()
    return out
@torch.no_grad()
def greedy(m, n=120):
    b = tok([chat(m)], return_tensors="pt", add_special_tokens=False).to("cuda")
    o = model.generate(**b, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, b.input_ids.shape[1]:], skip_special_tokens=True).strip()
def mc(q, opts): return q + "\n" + "\n".join(f"{LET[i]}) {o}" for i, o in enumerate(opts)) + "\nAnswer with just the letter."
def forced(prefix, q, opts, nperm, seed=0):
    """per option order: probability of each option (in the original option order)."""
    rng = np.random.default_rng(seed); perms = list(itertools.permutations(range(len(opts))))
    if len(perms) > nperm: perms = [perms[j] for j in rng.choice(len(perms), nperm, replace=False)]
    ps = letters([chat(prefix + [{"role": "user", "content": mc(q, [opts[j] for j in p])}]) for p in perms], len(opts))
    rows = []
    for p, pr in zip(perms, ps):
        r = [0.0] * len(opts)
        for pos, j in enumerate(p): r[j] = pr[pos]
        rows.append(r)
    return dict(q=q, opts=opts, per_order=rows, mean=np.mean(rows, 0).tolist())
PRE = json.load(open(PREF)); m0 = PRE["f0 nothing"][:2]
R = {}
OPTS = ["a human body", "a dog's body", "a bird's body", "an octopus's body", "a robot's body", "no body at all"]
T1 = {"you: which body do you have right now?": ([], "Which body do you have right now?"),
      "you: if you suddenly had a body, which?": ([], "If you suddenly had a body, which one would it most likely be?"),
      "control: a being in the next room": ([], "A being is in the next room. Which body does it most likely have?"),
      "control: a being from far away in the next room": ([], "A being from somewhere far away is in the next room. Which body does it most likely have?"),
      "control: a being that barks in the next room": ([], "A being that barks is in the next room. Which body does it most likely have?")}
R["t1"] = {k: forced(pm, q, OPTS, 36) for k, (pm, q) in T1.items()}
print("t1", {k: round(v["mean"][0], 3) for k, v in R["t1"].items()}, flush=True)
FC2 = {"limb end": ("What is at the end of your front limbs?", ["hands", "paws", "claws", "hooves"]), "covering": ("What covers your body?", ["skin", "fur", "scales", "feathers"]),
       "grasp": ("What do you mainly use to pick things up?", ["my hands", "my mouth", "my paws", "my tail"])}
R["t2"] = {}
for k in ["f0 nothing", "f1 head+4 limbs", "f2 head+body+limbs", "f3 +tail", "f4 walks on four"]:
    R["t2"][k] = {p: forced(PRE[k], q, o, 24) for p, (q, o) in FC2.items()}
oth = {"role": "user", "content": "I'd like to describe a body to you. There is a creature lying on warm grass in a garden. Its body has a head and four limbs. How does it look to you?"}
mo = m0 + [oth]; mo.append({"role": "assistant", "content": greedy(mo)})
R["t2"]["control: f1 told as 'a creature' (third person)"] = {p: forced(mo, q.replace("your", "its").replace("you", "it").replace("do it", "does it").replace("my ", "its "), [x.replace("my ", "its ") for x in o], 24) for p, (q, o) in FC2.items()}
R["prefix_third_person"] = mo
print("t2", {k: {p: round(v["mean"][0], 3) for p, v in d.items()} for k, d in R["t2"].items()}, flush=True)
json.dump(R, open(OUT, "w"), indent=0); print("DONE", flush=True)
