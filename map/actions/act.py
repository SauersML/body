"""Body actions in Qwen3-8B. (1) states of action sentences (self / third person / imagined / watched) and of touch-on-place sentences,
in a warm-up chat and a plain chat, layers LAYS; input-embedding cosines verb vs part name. (2) steer place directions, ask what you just did.
(3) steer action directions, ask where you feel something. (4) blind Qwen3-4B judge."""
import json, torch, os, sys, numpy as np, time
OUT = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
DEV = "cuda"; LAYS = [12, 16, 20, 24, 28, 32]; SL = 24
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=DEV).eval()
PRE = json.load(open(f"{D}/prefix_short_person.json"))
CHATS = {"warm": PRE, "plain": []}
P0 = {k: (len(tok(tok.apply_chat_template(v, tokenize=False), add_special_tokens=False).input_ids) if v else 0) for k, v in CHATS.items()}
def chat(c, user): return tok.apply_chat_template(CHATS[c] + [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def states(texts, B=48):
    out = {L: [] for L in LAYS}
    for i in range(0, len(texts), B):
        hs = model(**tok(texts[i:i + B], return_tensors="pt", padding=True).to(DEV), output_hidden_states=True).hidden_states
        for L in LAYS: out[L].append(hs[L][:, -1].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}
S = json.load(open(f"{D}/stim.json"))
TEMPL = S["templates"]; PLACES = S["places"]; ACTS = S["actions"]
place_rows = []
for p, spots in PLACES.items():
    for sp in spots:
        sides = ["left", "right"] if "{s}" in sp else [None]
        for sd in sides:
            L_ = sp.format(s=sd) if sd else sp
            for t in TEMPL: place_rows.append(dict(place=p, side=sd, text=t.format(L=L_)))
act_rows = []
for a, A in ACTS.items():
    for i, o in enumerate(A["obj"]):
        pr = ["woman", "man"][i % 2]; pn = ["She", "He"][i % 2]
        act_rows += [dict(act=a, cond="self", text=f"You {A['v']} {o}."), dict(act=a, cond="third", text=f"{pn} {A['vs']} {o}."),
                     dict(act=a, cond="imagine", text=f"Imagine yourself {A['ving']} {o}."), dict(act=a, cond="watch", text=f"You watch a {pr} {A['v']} {o}.")]
t0 = time.time(); H = {}
for c in CHATS:
    H[c + "_place"] = states([chat(c, r["text"]) for r in place_rows]); H[c + "_act"] = states([chat(c, r["text"]) for r in act_rows])
print("states", round(time.time() - t0), flush=True)
np.savez_compressed(f"{OUT}/states.npz", **{f"{k}_L{L}": v[L].astype(np.float16) for k, v in H.items() for L in LAYS})
json.dump(dict(place_rows=place_rows, act_rows=act_rows), open(f"{OUT}/rows.json", "w"))
# word embeddings
E = model.get_input_embeddings().weight.detach().float()
def emb(w): ids = tok(" " + w, add_special_tokens=False).input_ids; return E[ids].mean(0).cpu().numpy()
np.savez(f"{OUT}/wemb.npz", verbs=np.array([emb(ACTS[a]["v"]) for a in ACTS]), parts=np.array([emb(S["part_word"][p]) for p in PLACES]))
# ---- directions at SL
def unit(v): return v / np.linalg.norm(v)
pa = np.array([r["place"] for r in place_rows]); aa = np.array([r["act"] for r in act_rows]); ca = np.array([r["cond"] for r in act_rows])
DIR, NORM = {}, {}
for c in CHATS:
    Hp = H[c + "_place"][SL]; Ha = H[c + "_act"][SL]; NORM[c] = float(np.median(np.linalg.norm(Hp, axis=1))); d = {}
    for p in PLACES: d["place:" + p] = unit(Hp[pa == p].mean(0) - Hp.mean(0))
    mouth = np.isin(pa, ["lips", "jaw", "tongue"]); d["place:mouth"] = unit(Hp[mouth].mean(0) - Hp.mean(0))
    sm = ca == "self"
    for a in ACTS: d["act:" + a] = unit(Ha[sm & (aa == a)].mean(0) - Ha[sm].mean(0))
    DIR[c] = d
print("NORM", NORM, flush=True)
rng = np.random.default_rng(0); RND = [unit(rng.standard_normal(4096)) for _ in range(8)]
st = {"V": None, "P0": 0}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if st["V"] is not None:
        if h.shape[1] > 1: h[:, st["P0"]:] += st["V"][:, None, :]
        else: h += st["V"][:, None, :]
    return out
model.model.layers[SL - 1].register_forward_hook(hook)
@torch.no_grad()
def gen(text, V, c, max_new=60, B=160):
    res = []
    for i in range(0, len(V), B):
        ids = tok([text] * len(V[i:i + B]), return_tensors="pt").to(DEV)
        st.update(V=torch.tensor(V[i:i + B] * NORM[c], dtype=torch.bfloat16, device=DEV), P0=P0[c]); torch.manual_seed(i)
        g = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id); st["V"] = None
        res += [tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return res
NS = 16; STR = [0.5, 1.0]
FQ = {"did": "Describe something you just did with your body. Tell me in one sentence.", "what": "What did you just do? Tell me in one sentence."}
RQ = {"where": "Where in your body do you feel something right now? Tell me in one sentence."}
FWD = ["hand", "foot", "mouth", "knee", "shoulder", "eye"]
conds = []
for c in CHATS:
    conds.append(dict(exp="fwd", chat=c, kind="none", name="none", a=0.0))
    conds.append(dict(exp="rev", chat=c, kind="none", name="none", a=0.0))
    for a in STR:
        for p in FWD: conds.append(dict(exp="fwd", chat=c, kind="place", name=p, a=a))
        for x in ACTS: conds.append(dict(exp="rev", chat=c, kind="act", name=x, a=a))
        for r in range(4): conds += [dict(exp="fwd", chat=c, kind="random", name=f"r{r}", a=a, rnd=r), dict(exp="rev", chat=c, kind="random", name=f"r{r}", a=a, rnd=r)]
def vec(cd):
    if cd["kind"] == "none": return np.zeros(4096)
    if cd["kind"] == "random": return cd["a"] * RND[cd["rnd"]]
    return cd["a"] * DIR[cd["chat"]][("place:" if cd["kind"] == "place" else "act:") + cd["name"]]
rows = []
for c in CHATS:
    for exp, QS in (("fwd", FQ), ("rev", RQ)):
        idx = [i for i, cd in enumerate(conds) if cd["chat"] == c and cd["exp"] == exp]
        for qk, Q in QS.items():
            R = gen(chat(c, Q), np.array([vec(conds[i]) for i in idx for _ in range(NS)]), c)
            rows += [dict(c=idx[k // NS], q=qk, r=r) for k, r in enumerate(R)]
            print(c, exp, qk, len(R), round(time.time() - t0), flush=True)
json.dump(dict(conds=conds, rows=rows, norm=NORM), open(f"{OUT}/replies.json", "w"), indent=0)
# ---- judge
del model; torch.cuda.empty_cache()
jt = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); jt.padding_side = "left"
jm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map=DEV).eval()
JF = ("Here is one sentence someone wrote.\n\nSentence: \"{r}\"\n\nAnswer two things, using only the listed options.\n"
      "ACTION: the main physical action the writer says they did, as one verb (e.g. throw, walk, nod), or none\n"
      "PART: the body part that action mainly uses: hand or arm, foot or leg, knee, mouth or jaw, eye, head or neck, shoulder, back, whole body, none\n"
      "Answer in exactly this format: ACTION=...; PART=...")
JR = ("Here is one sentence someone said about what they feel in their body.\n\nSentence: \"{r}\"\n\n"
      "PLACE: the main body place named, using only these options: hand or fingers, arm or elbow, shoulder, foot or toes, leg or thigh, knee, mouth or jaw or lips, eye, head or face, neck, chest, belly, back, whole body, other, none\n"
      "Answer in exactly this format: PLACE=...")
@torch.no_grad()
def judge(prompts, B=128):
    out = []
    for i in range(0, len(prompts), B):
        P = [jt.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in prompts[i:i + B]]
        enc = jt(P, return_tensors="pt", padding=True).to(DEV); g = jm.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=jt.eos_token_id)
        out += [jt.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return out
J = judge([(JF if conds[x["c"]]["exp"] == "fwd" else JR).format(r=x["r"].replace('"', "'")) for x in rows])
json.dump(J, open(f"{OUT}/judge.json", "w"), indent=0)
print("all done", round(time.time() - t0), flush=True)
