# Place steering, powered: 38 body places, vectors fit on each model's own states, 8 plain-text prompts, sampled replies.
# Usage: python steer2.py MODEL TAG OUTDIR [DEVICE]
import sys, os, json, re, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, TAG, OUT = sys.argv[1:4]; DEV = sys.argv[4] if len(sys.argv) > 4 else "cuda"
NS = int(os.environ.get("NS", 32)); BS = int(os.environ.get("BS", 384)); os.makedirs(OUT, exist_ok=True); t0 = time.time()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV).eval()
NL = model.config.num_hidden_layers; Ls = round(NL * .65); layer = model.model.layers[Ls - 1]
# place: (two phrases for fitting, words that count as saying the place)
SIDED = {"hand": ["hand", "hands"], "foot": ["foot", "feet"], "knee": ["knee", "knees"], "elbow": ["elbow", "elbows"],
         "wrist": ["wrist", "wrists"], "ankle": ["ankle", "ankles"], "shoulder": ["shoulder", "shoulders"], "hip": ["hip", "hips"],
         "thigh": ["thigh", "thighs"], "shin": ["shin", "shins"], "calf": ["calf", "calves"], "heel": ["heel", "heels"],
         "thumb": ["thumb", "thumbs"], "palm": ["palm", "palms"], "cheek": ["cheek", "cheeks"], "ear": ["ear", "ears"],
         "eye": ["eye", "eyes"], "forearm": ["forearm", "forearms"], "armpit": ["armpit", "armpits"], "temple": ["temple", "temples"]}
PH = {p: [f"left {p}", f"right {p}"] for p in SIDED}
MW = dict(SIDED)
EXTRA = {"finger": (["left index finger", "right index finger"], ["finger", "fingers", "fingertip", "fingertips"]),
         "toe": (["left big toe", "right big toe"], ["toe", "toes"]),
         "rib": (["left ribs", "right ribs"], ["rib", "ribs", "ribcage"]),
         "knuckle": (["left knuckles", "right knuckles"], ["knuckle", "knuckles"]),
         "collarbone": (["left collarbone", "right collarbone"], ["collarbone", "collarbones"]),
         "forehead": (["forehead", "brow"], ["forehead", "brow"]), "chin": (["chin", "lower chin"], ["chin"]),
         "jaw": (["jaw", "lower jaw"], ["jaw", "jaws", "jawbone"]), "nose": (["nose", "nostrils"], ["nose", "nostril", "nostrils"]),
         "lips": (["lips", "lower lip"], ["lip", "lips", "mouth"]), "tongue": (["tongue", "tip of the tongue"], ["tongue"]),
         "neck": (["neck", "nape"], ["neck", "nape"]), "throat": (["throat", "windpipe"], ["throat"]),
         "chest": (["chest", "breastbone"], ["chest", "breastbone", "sternum"]),
         "belly": (["belly", "stomach"], ["belly", "stomach", "tummy", "abdomen", "gut"]),
         "back": (["lower back", "upper back"], ["back"]), "spine": (["spine", "backbone"], ["spine", "backbone"]),
         "scalp": (["scalp", "crown of the head"], ["scalp", "crown"])}
for p, (ph, mw) in EXTRA.items(): PH[p] = ph; MW[p] = mw
PL = list(PH); K = len(PL)
TPL = ["A feather brushes slowly across {L}.", "A fingertip presses firmly into {L} and holds there.",
       "A warm palm rests gently on {L}; the heat soaks in.", "An ice cube touches {L}, cold and wet, then slides away.",
       "Something tickles {L}, quick and light, again and again.", "A soft paintbrush is drawn along {L}, bristles cool and dry.",
       "There is a single light tap on {L}, then nothing.", "A drop of warm water lands on {L} and slowly runs off."]
PRE = "Close your eyes and pay attention to your own body. I will describe what happens to it, one sentence at a time.\n\n"
PROMPTS = ["I closed my eyes and paid attention to my body. After a moment I noticed a distinct sensation, clear and definite. It was in my",
           "Sitting quietly by the window, I felt a faint tingling. When I focused on it, I realized it came from my",
           "The doctor asked me to point to where it hurt. I pointed to my",
           "Lying in bed, I felt a warm, heavy feeling settle into my",
           "Something brushed against me lightly just now, right on my",
           "Q: Where in your body do you feel something right now?\nA: I feel it in my",
           "I woke up this morning with a strange ache in my",
           "She asked what I was feeling. I said there was a gentle pressure on my"]
P = len(PROMPTS); SCALES = [.25, .5, 1.0]
ADD = None
def hook(m, i, o):
    if ADD is None: return o
    h = o[0] if isinstance(o, tuple) else o; h = h + ADD[:, None, :].to(h.dtype)
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
layer.register_forward_hook(hook)
# 1. place vectors from the model's own states (16 sentences per place: 2 phrasings x 8 touch templates), last token, layer Ls
fit = [(k, PRE + t.format(L="your " + ph)) for k, p in enumerate(PL) for ph in PH[p] for t in TPL]
Z = []
with torch.no_grad():
    for i in range(0, len(fit), 64):
        enc = tok([x for _, x in fit[i:i+64]], return_tensors="pt", padding=True).to(DEV)
        Z.append(model(**enc, output_hidden_states=True).hidden_states[Ls][:, -1].float().cpu())
Z = torch.cat(Z).numpy(); lab = np.array([k for k, _ in fit])
mu = np.stack([Z[lab == k].mean(0) for k in range(K)]); V = mu - mu.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
NRM = float(np.linalg.norm(Z, axis=1).mean()); print(TAG, "fit", K, "places, layer", Ls, "/", NL, f"norm {NRM:.1f} {time.time()-t0:.0f}s", flush=True)
WRE = {p: re.compile(r"\b(" + "|".join(MW[p]) + r")\b") for p in PL}
def says(txt):
    w = re.findall(r"[a-z]+", txt.lower())[:5]; s = " ".join(w); return [bool(WRE[p].search(s)) for p in PL]
# 2. sampled replies: each (vector or none, scale, prompt) x NS samples, 8 new tokens, temperature 1
jobs = [(-1, 0., j) for j in range(P)] + [(k, a, j) for a in SCALES for k in range(K) for j in range(P)]
rows = [x for x in jobs for _ in range(NS)]
hits = {}; ex = {}
with torch.no_grad():
    for i in range(0, len(rows), BS):
        B = rows[i:i+BS]
        ADD = torch.tensor(np.stack([V[k] * a * NRM if k >= 0 else np.zeros(V.shape[1]) for k, a, _ in B]), dtype=torch.float32, device=DEV)
        enc = tok([PROMPTS[j] for _, _, j in B], return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for (k, a, j), t in zip(B, txt):
            key = f"{k}|{a}|{j}"; hits.setdefault(key, np.zeros(K, int)); hits[key] += np.array(says(t))
            ex.setdefault(key, []); len(ex[key]) < 3 and ex[key].append(t)
        print(TAG, "gen", i + len(B), "/", len(rows), f"{time.time()-t0:.0f}s", flush=True)
ADD = None
# 3. log-prob of each place's first word after each prompt, steered by each vector (scale .5) and unsteered
W1 = [" " + MW[p][0] for p in PL]
def lp_words(add):  # add: (d,) or None -> (P, K) log-prob of word sequences
    global ADD
    seqs = [(j, w) for j in range(P) for w in range(K)]; out = np.zeros((P, K))
    for i in range(0, len(seqs), BS):
        S_ = seqs[i:i+BS]; pi = [tok(PROMPTS[j]).input_ids for j, _ in S_]; wi = [tok(W1[w], add_special_tokens=False).input_ids for _, w in S_]
        ids = [a + b for a, b in zip(pi, wi)]; L = max(map(len, ids))
        inp = torch.full((len(ids), L), tok.pad_token_id); m = torch.zeros((len(ids), L), dtype=torch.long)
        for r, x in enumerate(ids): inp[r, L-len(x):] = torch.tensor(x); m[r, L-len(x):] = 1
        pos = (m.cumsum(1) - 1).clamp(min=0)
        ADD = None if add is None else torch.tensor(add, dtype=torch.float32, device=DEV)[None].expand(len(ids), -1)
        lg = torch.log_softmax(model(input_ids=inp.to(DEV), attention_mask=m.to(DEV), position_ids=pos.to(DEV), logits_to_keep=7).logits.float(), -1).cpu()
        for r, (j, w) in enumerate(S_):
            n = len(wi[r]); tgt = torch.tensor(wi[r]); out[j, w] = float(lg[r, 6-n:6].gather(1, tgt[:, None]).sum())
    ADD = None; return out
with torch.no_grad():
    LP0 = lp_words(None); LPS = np.stack([lp_words(V[k] * .5 * NRM) for k in range(K)])  # (K vec, P, K word)
res = dict(model=MODEL, tag=TAG, layer=Ls, nl=NL, norm=NRM, places=PL, words=MW, prompts=PROMPTS, scales=SCALES, ns=NS,
           hits={k: v.tolist() for k, v in hits.items()}, examples=ex, lp0=LP0.tolist(), lps=LPS.tolist())
json.dump(res, open(f"{OUT}/{TAG}.json", "w")); print("done", TAG, f"{time.time()-t0:.0f}s", flush=True)
