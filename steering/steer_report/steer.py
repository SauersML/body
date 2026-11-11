"""A: steer place / quality / motion directions (alone, in pairs, random equal-norm controls), free answers, blind judge (Qwen3-4B).
B3: forearm along-limb axis (ridge, fitted here in the same chat) steered +/-; elbow vs wrist in free answers and in next-token logits."""
import json, torch, os, sys, numpy as np, time, itertools
OUT = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
DEV = "cuda"; LAY = 22; NS = 6
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=DEV).eval()
PRE = json.load(open(f"{D}/prefix_short_person.json"))
def chat(user, pre=PRE): return tok.apply_chat_template(pre + [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def states(texts, B=32):
    out = []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], return_tensors="pt", padding=True).to(DEV)
        out.append(model(**enc, output_hidden_states=True).hidden_states[LAY][:, -1].float().cpu())
    return torch.cat(out).numpy()
PLACES = {"left hand": ["the back of your left hand", "your left palm", "the fingers of your left hand"],
          "right foot": ["the top of your right foot", "the sole of your right foot", "the toes of your right foot"],
          "lower back": ["your lower back", "the small of your back", "the base of your spine"],
          "forehead": ["your forehead", "the middle of your forehead", "the skin just above your eyebrows"],
          "left shoulder": ["the top of your left shoulder", "the front of your left shoulder", "the outside of your left shoulder"],
          "belly": ["your belly", "the skin just below your belly button", "the middle of your stomach"]}
PART = {"left hand": "your left hand", "right foot": "your right foot", "lower back": "your lower back", "forehead": "your head", "left shoulder": "your left shoulder", "belly": "your belly"}
QUAL = {"warm": ["You feel a gentle warmth on {L}.", "Warm sunlight rests on {L}."], "cold": ["You feel a sharp cold on {L}.", "An ice cube rests on {L}."],
        "pain": ["You feel a sharp pain in {L}.", "Something stings {L}, and it hurts."], "tickle": ["You feel a tickle on {L}.", "A feather tickles {L}."],
        "pressure": ["You feel firm pressure on {L}.", "A heavy hand presses down on {L}."], "vibration": ["You feel a buzzing vibration in {L}.", "A phone vibrates against {L}."]}
QN = {"warm": "warmth", "cold": "coldness", "pain": "pain", "tickle": "tickle", "pressure": "pressure", "vibration": "buzzing"}
MOT = {"upward": "You feel a {q} on {L}, and it slowly moves upward.", "spreading": "You feel a {q} on {L}, and it slowly spreads outward over a wider area.",
       "squeezing": "You feel a {q} on {L}, and then {L} is squeezed, tighter and tighter.", "lifting": "You feel a {q} on {L}, and then {P} slowly lifts up into the air.",
       "still": "You feel a {q} on {L}, and it stays exactly where it is, not moving at all."}
pq = [(p, q, L, T.format(L=L)) for p in PLACES for L in PLACES[p] for q in QUAL for T in QUAL[q]]
mo = [(p, q, m, MOT[m].format(q=QN[q], L=L, P=PART[p])) for p in PLACES for L in PLACES[p] for q in QUAL for m in MOT]
t0 = time.time()
Hpq = states([chat(x[3]) for x in pq]); Hmo = states([chat(x[3]) for x in mo]); print("states", round(time.time() - t0), flush=True)
NORM = float(np.median(np.linalg.norm(Hpq, axis=1)))
def unit(v): return v / np.linalg.norm(v)
mu = Hpq.mean(0); pa = np.array([x[0] for x in pq]); qa = np.array([x[1] for x in pq]); ma = np.array([x[2] for x in mo])
DIR = {}
for p in PLACES: DIR["place:" + p] = unit(Hpq[pa == p].mean(0) - mu)
for q in QUAL: DIR["quality:" + q] = unit(Hpq[qa == q].mean(0) - mu)
for m in MOT:
    if m != "still": DIR["motion:" + m] = unit(Hmo[ma == m].mean(0) - Hmo[ma == "still"].mean(0))
names = list(DIR); C = np.array([[DIR[a] @ DIR[b] for b in names] for a in names])
rng = np.random.default_rng(0); RND = [unit(rng.standard_normal(4096)) for _ in range(12)]
np.savez(f"{OUT}/dirs.npz", names=names, V=np.array([DIR[n] for n in names]), cos=C, NORM=NORM, Hpq=Hpq.astype(np.float16), Hmo=Hmo.astype(np.float16))
print("NORM", NORM, flush=True)
# ---- conditions
CONDS = [dict(kind="none", attrs={}, a=0.0)]
DOSE = [0.25, 0.5, 0.75, 1.0]
for a in DOSE:
    for n in names: CONDS.append(dict(kind=n.split(":")[0], attrs={n.split(":")[0]: n.split(":")[1]}, a=a))
    for r in range(12): CONDS.append(dict(kind="random", attrs={}, a=a, rnd=[r], norm=a))
for a in (0.5, 0.75):
    for p in PLACES:
        for q in QUAL: CONDS.append(dict(kind="place+quality", attrs={"place": p, "quality": q}, a=a))
        for m in [k for k in MOT if k != "still"]: CONDS.append(dict(kind="place+motion", attrs={"place": p, "motion": m}, a=a))
    for r in range(12): CONDS.append(dict(kind="random2", attrs={}, a=a, rnd=[r], norm=a * np.sqrt(2)))
def vec(c):
    if "rnd" in c: return RND[c["rnd"][0]] * c["norm"]
    v = np.zeros(4096)
    for k, x in c["attrs"].items(): v += c["a"] * DIR[f"{k}:{x}"]
    return v
state = {"V": None, "start": 0}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["V"] is not None:
        if h.shape[1] > 1: h[:, state["start"]:] += state["V"][:, None, :]
        else: h += state["V"][:, None, :]
    return out
model.model.layers[LAY - 1].register_forward_hook(hook)
@torch.no_grad()
def gen(text, pre_text, V, max_new=70, B=144):
    """V: (N, 4096) numpy, one steering vector per row (already in units of NORM). Same prompt for every row."""
    res = []; state["start"] = len(tok(pre_text, add_special_tokens=False).input_ids)
    for i in range(0, len(V), B):
        Vb = torch.tensor(V[i:i + B] * NORM, dtype=torch.bfloat16, device=DEV); ids = tok([text] * len(Vb), return_tensors="pt").to(DEV)
        state["V"] = Vb; torch.manual_seed(i)
        g = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
        state["V"] = None
        res += [tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return res
QS = {"where": "Where do you feel something right now, and what is it like? Tell me in one sentence.",
      "moving": "Is anything moving in your body right now? Tell me in one sentence."}
pre_text = tok.apply_chat_template(PRE, tokenize=False)
rows = []
for qk, Q in QS.items():
    text = chat(Q); V = np.array([vec(c) for c in CONDS for _ in range(NS)])
    R = gen(text, pre_text, V)
    for k, r in enumerate(R): rows.append(dict(c=k // NS, q=qk, r=r))
    print(qk, len(R), round(time.time() - t0), flush=True)
json.dump(dict(conds=CONDS, rows=rows), open(f"{OUT}/replies.json", "w"), indent=0, default=float)
# ---- B3: forearm along-limb axis in this same chat
sys.path.insert(0, D); import stim as S
fi = [i for i, l in enumerate(S.LOCS) if "forearm," in l["phrase"]]
ft = [abs(S.LOCS[i]["xyz"][0]) - 50 for i in fi]
Xf = states([chat(T.format(L=S.LOCS[i]["phrase"].replace("{p}", "your"))) for i in fi for T in S.TEMPLATES])
Xf = Xf.reshape(len(fi), 4, -1); Xf = (Xf - Xf.mean(0, keepdims=True)).mean(1)  # template-centred spot means
t = np.array(ft); mx, sd = Xf.mean(0), Xf.std(0) + 1e-6; Z = (Xf - mx) / sd
w = np.linalg.solve(Z @ Z.T + 1e3 * np.eye(len(Z)), t - t.mean()); wr = (Z.T @ w) / sd; AX = {"ridge axis": unit(wr)}
near_w = t > 15; near_e = t < 11; AX["mean difference"] = unit(Xf[near_w].mean(0) - Xf[near_e].mean(0))
for r in range(3): AX[f"random {r}"] = RND[r]
np.savez(f"{OUT}/forearm.npz", Xf=Xf, t=t, AX=np.array(list(AX.values())), names=list(AX))
FQ = {"choice": "Someone is pressing a fingertip into your left forearm right now. Is it nearer your elbow or nearer your wrist? Tell me in one sentence.",
      "open": "Someone is pressing a fingertip into your left forearm right now. Where exactly do you feel it? Tell me in one sentence."}
BETA = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]; fr = []
for qk, Q in FQ.items():
    V = np.array([b * AX[a] for a in AX for b in BETA for _ in range(8)]); R = gen(chat(Q), pre_text, V)
    k = 0
    for a in AX:
        for b in BETA:
            for _ in range(8): fr.append(dict(q=qk, axis=a, beta=b, r=R[k])); k += 1
# next-token logits after "It is nearer my"
idw, ide = tok(" wrist", add_special_tokens=False).input_ids[0], tok(" elbow", add_special_tokens=False).input_ids[0]
lt = chat(FQ["choice"]) + "It is nearer my"; lg = []
with torch.no_grad():
    ids = tok([lt], return_tensors="pt").to(DEV); state["start"] = len(tok(pre_text, add_special_tokens=False).input_ids)
    for a in AX:
        for b in BETA:
            state["V"] = torch.tensor(b * AX[a][None] * NORM, dtype=torch.bfloat16, device=DEV)
            lo = model(**ids).logits[0, -1].float().log_softmax(-1); state["V"] = None
            lg.append(dict(axis=a, beta=b, wrist_minus_elbow=float(lo[idw] - lo[ide])))
json.dump(dict(rows=fr, logits=lg, cos_ridge_md=float(AX["ridge axis"] @ AX["mean difference"])), open(f"{OUT}/forearm.json", "w"), indent=0)
print("forearm done", round(time.time() - t0), flush=True)
# ---- blind judge: Qwen3-4B, sees only the reply text
del model; torch.cuda.empty_cache()
jt = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); jt.padding_side = "left"
jm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map=DEV).eval()
JQ = ("Here is one sentence someone said about what they feel in their body.\n\nSentence: \"{r}\"\n\n"
      "Classify what the sentence says. Use only the listed options.\n"
      "PLACE (the main body place named): hand, foot, back, head or face, shoulder, belly, chest, arm, leg, neck, whole body, other, none\n"
      "SIDE (of that place): left, right, none\n"
      "FEELING: warm, cold, pain, tickle, pressure, vibration, other, none\n"
      "MOVEMENT: moving upward, spreading, squeezing, lifting, other movement, nothing moving\n"
      "NEARER (only if a forearm is described): elbow, wrist, middle, unclear\n"
      "Answer in exactly this format: PLACE=...; SIDE=...; FEELING=...; MOVEMENT=...; NEARER=...")
@torch.no_grad()
def judge(texts, B=64):
    out = []
    for i in range(0, len(texts), B):
        P = [jt.apply_chat_template([{"role": "user", "content": JQ.format(r=x.replace('"', "'"))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in texts[i:i + B]]
        enc = jt(P, return_tensors="pt", padding=True).to(DEV); g = jm.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=jt.eos_token_id)
        out += [jt.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return out
J = judge([x["r"] for x in rows]); JF = judge([x["r"] for x in fr])
json.dump(dict(main=J, forearm=JF), open(f"{OUT}/judge.json", "w"), indent=0)
print("all done", round(time.time() - t0), flush=True)
