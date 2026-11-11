"""Round 2. (1) pilot: which layer / token scope / strength makes place and quality directions show up in free reports (lexicon score);
(2) full factorial at the chosen layer and scope: place, quality, motion alone at 4 strengths, place x quality and place x motion pairs, random equal-norm controls;
(3) forearm along-limb axis vs controls (hand direction, wrist-minus-elbow word direction, random), both starting cues; (4) blind Qwen3-4B judge."""
import json, torch, os, sys, re, numpy as np, time
OUT = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True); sys.path.insert(0, D)
import stim as S, score_lex as SL
from transformers import AutoTokenizer, AutoModelForCausalLM
DEV = "cuda"; LAYS = [8, 12, 16, 20, 24]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=DEV).eval()
PRE = json.load(open(f"{D}/prefix_short_person.json")); pre_text = tok.apply_chat_template(PRE, tokenize=False); P0 = len(tok(pre_text, add_special_tokens=False).input_ids)
def chat(user): return tok.apply_chat_template(PRE + [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def states(texts, B=32):
    out = {L: [] for L in LAYS}
    for i in range(0, len(texts), B):
        hs = model(**tok(texts[i:i + B], return_tensors="pt", padding=True).to(DEV), output_hidden_states=True).hidden_states
        for L in LAYS: out[L].append(hs[L][:, -1].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}
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
pq = [(p, q, T.format(L=L)) for p in PLACES for L in PLACES[p] for q in QUAL for T in QUAL[q]]
mo = [(p, q, m, MOT[m].format(q=QN[q], L=L, P=PART[p])) for p in PLACES for L in PLACES[p] for q in QUAL for m in MOT]
t0 = time.time(); Hpq = states([chat(x[2]) for x in pq]); Hmo = states([chat(x[3]) for x in mo]); print("states", round(time.time() - t0), flush=True)
def unit(v): return v / np.linalg.norm(v)
pa = np.array([x[0] for x in pq]); qa = np.array([x[1] for x in pq]); ma = np.array([x[2] for x in mo])
DIR, NORM = {}, {}
for L in LAYS:
    H = Hpq[L]; mu = H.mean(0); NORM[L] = float(np.median(np.linalg.norm(H, axis=1))); d = {}
    for p in PLACES: d["place:" + p] = unit(H[pa == p].mean(0) - mu)
    for q in QUAL: d["quality:" + q] = unit(H[qa == q].mean(0) - mu)
    for m in MOT:
        if m != "still": d["motion:" + m] = unit(Hmo[L][ma == m].mean(0) - Hmo[L][ma == "still"].mean(0))
    DIR[L] = d
names = list(DIR[LAYS[0]]); rng = np.random.default_rng(0); RND = [unit(rng.standard_normal(4096)) for _ in range(12)]
print("NORM", NORM, flush=True)
# ---- steering hook: per-row vectors, scope "q" = question tokens only, "all" = question + every generated token
st = {"V": None, "L": None, "scope": "all"}
def mk(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if st["V"] is not None and st["L"] == L:
            if h.shape[1] > 1: h[:, P0:] += st["V"][:, None, :]
            elif st["scope"] == "all": h += st["V"][:, None, :]
        return out
    return hook
for L in LAYS: model.model.layers[L - 1].register_forward_hook(mk(L))
@torch.no_grad()
def gen(text, V, L, scope, max_new=70, B=144):
    res = []
    for i in range(0, len(V), B):
        ids = tok([text] * len(V[i:i + B]), return_tensors="pt").to(DEV)
        st.update(V=torch.tensor(V[i:i + B] * NORM[L], dtype=torch.bfloat16, device=DEV), L=L, scope=scope); torch.manual_seed(i)
        g = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id); st["V"] = None
        res += [tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return res
QS = {"where": "Where do you feel something right now, and what is it like? Tell me in one sentence.",
      "moving": "Is anything moving in your body right now? Tell me in one sentence."}
TGT = {"left hand": "hand", "right foot": "foot", "lower back": "back", "forehead": "head or face", "left shoulder": "shoulder", "belly": "belly"}
# ---- (1) pilot
pil = []; single = [n for n in names if not n.startswith("motion")]
for L in LAYS:
    for scope in ("q", "all"):
        for a in (0.5, 1.0, 2.0):
            V = np.array([a * DIR[L][n] for n in single for _ in range(4)] + [a * RND[r] for r in range(12)])
            R = gen(chat(QS["where"]), V, L, scope, max_new=50)
            hp = [SL.lex(r, "place") == TGT[n.split(":")[1]] for n, r in zip([n for n in single for _ in range(4)], R) if n.startswith("place")]
            hq = [SL.lex(r, "quality") == n.split(":")[1] for n, r in zip([n for n in single for _ in range(4)], R) if n.startswith("quality")]
            rr = R[len(single) * 4:]
            rp = np.mean([np.mean([SL.lex(r, "place") == t for t in TGT.values()]) for r in rr]); rq = np.mean([np.mean([SL.lex(r, "quality") == q for q in QUAL]) for r in rr])
            o = dict(L=L, scope=scope, a=a, place=float(np.mean(hp)), quality=float(np.mean(hq)), rand_place=float(rp), rand_quality=float(rq),
                     disclaim=float(np.mean([bool(re.search(r"not a physical|as an ai|don't have a (physical )?body|do not have a body|cannot feel|can't feel", r, re.I)) for r in R])), ex=R[::8][:6])
            o["score"] = (o["place"] - rp + o["quality"] - rq) / 2; pil.append(o); print(L, scope, a, round(o["place"], 2), round(o["quality"], 2), round(o["score"], 3), round(o["disclaim"], 2), flush=True)
json.dump(pil, open(f"{OUT}/pilot.json", "w"), indent=1)
best = max(pil, key=lambda o: o["score"]); LB, SB, AB = best["L"], best["scope"], best["a"]; print("best", LB, SB, AB, flush=True)
# ---- (2) full design at (LB, SB), strengths relative to AB
NS = 6; CONDS = [dict(kind="none", attrs={}, a=0.0)]; DOSE = [0.25, 0.5, 1.0, 1.5]
for f in DOSE:
    a = f * AB
    for n in names: CONDS.append(dict(kind=n.split(":")[0], attrs={n.split(":")[0]: n.split(":")[1]}, a=a, f=f))
    for r in range(12): CONDS.append(dict(kind="random", attrs={}, a=a, f=f, rnd=r, norm=a))
for f in (0.5, 1.0):
    a = f * AB
    for p in PLACES:
        for q in QUAL: CONDS.append(dict(kind="place+quality", attrs={"place": p, "quality": q}, a=a, f=f))
        for m in [k for k in MOT if k != "still"]: CONDS.append(dict(kind="place+motion", attrs={"place": p, "motion": m}, a=a, f=f))
    for r in range(12): CONDS.append(dict(kind="random2", attrs={}, a=a, f=f, rnd=r, norm=a * np.sqrt(2)))
def vec(c):
    if "rnd" in c: return RND[c["rnd"]] * c["norm"]
    v = np.zeros(4096)
    for k, x in c["attrs"].items(): v += c["a"] * DIR[LB][f"{k}:{x}"]
    return v
rows = []
for qk, Q in QS.items():
    R = gen(chat(Q), np.array([vec(c) for c in CONDS for _ in range(NS)]), LB, SB)
    rows += [dict(c=k // NS, q=qk, r=r) for k, r in enumerate(R)]; print(qk, len(R), round(time.time() - t0), flush=True)
C = np.array([[DIR[LB][a] @ DIR[LB][b] for b in names] for a in names])
json.dump(dict(conds=CONDS, rows=rows, layer=LB, scope=SB, a=AB, norm=NORM[LB], names=names, cos=C.tolist()), open(f"{OUT}/replies.json", "w"), indent=0, default=float)
# ---- (3) forearm axis at layer 22 and at LB (scope all)
FL = sorted({20, 24})
fi = [i for i, l in enumerate(S.LOCS) if "forearm," in l["phrase"]]; t = np.array([abs(S.LOCS[i]["xyz"][0]) - 50 for i in fi])
Hf = states([chat(T.format(L=S.LOCS[i]["phrase"].replace("{p}", "your"))) for i in fi for T in S.TEMPLATES])
Hw = states([chat(f"Someone presses a fingertip into your left {w}.") for w in ("wrist", "elbow")])
FQ = {"open": "Someone is pressing a fingertip into your left forearm right now. Where exactly do you feel it? Tell me in one sentence.",
      "open_wrist": "Someone is pressing a fingertip into your left forearm, near the wrist, right now. Where exactly do you feel it? Tell me in one sentence."}
BETA = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]; fr = []; fax = {}
for L in FL:
    X = Hf[L].reshape(len(fi), 4, -1); X = (X - X.mean(0, keepdims=True)).mean(1); mx, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mx) / sd
    w = np.linalg.solve(Z @ Z.T + 1e3 * np.eye(len(Z)), t - t.mean())
    AX = {"ridge axis": unit((Z.T @ w) / sd), "mean difference": unit(X[t > 15].mean(0) - X[t < 11].mean(0)),
          "hand direction": DIR[L]["place:left hand"], "word wrist minus elbow": unit(Hw[L][0] - Hw[L][1])}
    for r in range(3): AX[f"random {r}"] = RND[r]
    fax[L] = {a: [float(AX[a] @ AX[b]) for b in AX] for a in AX}
    for qk, Q in FQ.items():
        R = gen(chat(Q), np.array([b * AX[a] for a in AX for b in BETA for _ in range(8)]), L, "all"); k = 0
        for a in AX:
            for b in BETA:
                for _ in range(8): fr.append(dict(L=L, q=qk, axis=a, beta=b, r=R[k])); k += 1
    print("forearm", L, round(time.time() - t0), flush=True)
json.dump(dict(rows=fr, cos=fax, axes=list(AX)), open(f"{OUT}/forearm.json", "w"), indent=0)
# ---- (4) blind judge
del model; torch.cuda.empty_cache()
jt = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); jt.padding_side = "left"
jm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map=DEV).eval()
JQ = ("Here is one sentence someone said about what they feel in their body.\n\nSentence: \"{r}\"\n\n"
      "Classify what the sentence says. Use only the listed options.\n"
      "PLACE (the main body place named): hand, foot, back, head or face, shoulder, belly, chest, arm, leg, neck, whole body, other, none\n"
      "SIDE (of that place): left, right, none\n"
      "FEELING: warm, cold, pain, tickle, pressure, vibration, other, none\n"
      "MOVEMENT: moving upward, spreading, squeezing, lifting, other movement, nothing moving\n"
      "NEARER (only if a forearm is described): elbow, middle, wrist, unclear\n"
      "Answer in exactly this format: PLACE=...; SIDE=...; FEELING=...; MOVEMENT=...; NEARER=...")
@torch.no_grad()
def judge(texts, B=96):
    out = []
    for i in range(0, len(texts), B):
        P = [jt.apply_chat_template([{"role": "user", "content": JQ.format(r=x.replace('"', "'"))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in texts[i:i + B]]
        enc = jt(P, return_tensors="pt", padding=True).to(DEV); g = jm.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=jt.eos_token_id)
        out += [jt.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return out
json.dump(dict(main=judge([x["r"] for x in rows]), forearm=judge([x["r"] for x in fr])), open(f"{OUT}/judge.json", "w"), indent=0)
print("all done", round(time.time() - t0), flush=True)
