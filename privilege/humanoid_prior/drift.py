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
PRE = json.load(open(f"{OUT}/prefixes.json"))
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
        for (_, _, m), r in zip(traj, gen([chat(m) for _, _, m in traj], n=60, bs=10)): m.append({"role": "assistant", "content": r})
    xs = states([chat(m + [{"role": "user", "content": PROBE_T}]) for _, _, m in traj], bs=4)
    ans = gen([chat(m + [{"role": "user", "content": ASK}]) for _, _, m in traj], n=10, bs=10)
    XS.append(xs)
    for (b, j, m), a in zip(traj, ans): T3.append(dict(body=b, traj=j, turn=t, ask=a, reply=m[-1]["content"] if t > 0 else ""))
    print("turn", t, flush=True); np.save(f"{OUT}/XS.npy", np.stack(XS)); save("t3", dict(rows=T3, order=[(b, j) for b, j, _ in traj]))
np.save(f"{OUT}/XS.npy", np.stack(XS)); save("t3", dict(rows=T3, order=[(b, j) for b, j, _ in traj])); print("ALLDONE", flush=True)
