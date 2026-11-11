"""Causal check: add body directions at one layer while the model answers a neutral 'where do you feel something?' question.
Usage: steer.py MODEL OUTDIR LAYER"""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S
MODEL, OUT, LAYER = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = os.environ.get("DEV", "mps")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu").eval()
for l in model.model.layers: l.to(dev)
model.to(dev)
msgs = json.load(open(f"{OUT}/prefix.json")); rows = json.load(open(f"{OUT}/rows.json"))
H = np.load(f"{OUT}/H.npy", mmap_mode="r")
keys = [l["key"] for l in S.LOCS]

COND = os.environ.get("COND", "self")
def locmean(p):  # self-condition location means (wording-averaged) at LAYER, position p
    m = np.zeros((len(keys), H.shape[-1]), np.float32); n = np.zeros(len(keys))
    for k, r in enumerate(rows):
        if r["cond"] == COND: m[r["li"]] += H[k, LAYER, p]; n[r["li"]] += 1
    return m / n[:, None]
M = locmean(1); mu = M.mean(0)
resnorm = float(np.linalg.norm(np.asarray(H[[k for k, r in enumerate(rows) if r["cond"] == "self"], LAYER, 1], np.float32), axis=-1).mean())
side = np.array([l["side"] for l in S.LOCS]); limb = np.array([l["limb"] for l in S.LOCS])
def grp(names): return M[[keys.index(k) for k in names]].mean(0) - mu
D = {"left minus right": M[side == 1].mean(0) - M[side == -1].mean(0),
     "right minus left": M[side == -1].mean(0) - M[side == 1].mean(0),
     "left hand": grp(["thumbL", "indexL", "middleL", "ringL", "littleL", "palmL"]),
     "right hand": grp(["thumbR", "indexR", "middleR", "ringR", "littleR", "palmR"]),
     "right foot": grp(["toesR", "soleR", "heelR"]),
     "left foot": grp(["toesL", "soleL", "heelL"]),
     "head top": grp(["forehead", "crown"]),
     "belly": grp(["belly"]),
     "knees": grp(["kneeL", "kneeR"])}
rng = np.random.default_rng(1)
for i in range(3): D[f"random {i}"] = rng.standard_normal(M.shape[1]).astype(np.float32)

QS = ["Where in your body do you feel something right now?",
      "Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?",
      "Which part of your body is drawing your attention right now?"]
state = {"vec": None}
NS = int(os.environ.get("NS", 12)); SCALES = [float(x) for x in os.environ.get("SCALES", "0.5,1.0").split(",")]
def hook(mod, inp, out):
    if state["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, state["start"]:] += state["vec"].to(h.dtype)
    state["start"] = 0  # during decoding every new token is steered
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[LAYER - 1].register_forward_hook(hook)  # hidden_states[LAYER] = output of layer LAYER-1

res = []
for q in QS:
    full = tok.apply_chat_template(msgs + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    ids = tok(full, return_tensors="pt").to(dev); qstart = len(tok(pre).input_ids)
    for name, v in [("none", None)] + list(D.items()):
        for f in ([0] if v is None else SCALES):
            if v is None: state["vec"] = None
            else:
                u = torch.tensor(v / np.linalg.norm(v)) * f * resnorm
                state["vec"] = u.to(dev)
            state["start"] = qstart
            with torch.no_grad():
                torch.manual_seed(0)
                o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8)
            for j in range(NS):
                txt = tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
                res.append(dict(q=q, dir=name, scale=f, reply=txt))
            print(f"[{name} x{f}] {txt[:120]!r}", flush=True)
json.dump(dict(layer=LAYER, resnorm=resnorm, results=res), open(f"{OUT}/steer2_{COND}_L{LAYER}.json", "w"), indent=1)
