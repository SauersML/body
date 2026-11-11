"""v2 steering: named, word-free and neighbour-only directions at one layer. Usage: steer3.py MODEL OUTDIR LAYER"""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S, stimuli2 as S2
MODEL, OUT, LAYER = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
msgs = json.load(open(f"{OUT}/prefix.json")); rows = json.load(open(f"{OUT}/rows.json"))
H = np.load(f"{OUT}/H_rs.npy", mmap_mode="r")
keys = [l["key"] for l in S.LOCS]
def mean_of(sel): return np.asarray(H[sel, LAYER], np.float32).mean(0)
M = np.stack([mean_of([k for k, r in enumerate(rows) if r["cond"] == "self" and r["li"] == i]) for i in range(len(keys))]); mu = M.mean(0)
W = {w: mean_of([k for k, r in enumerate(rows) if r["cond"] == "wordfree" and r["loc"] == w]) for w in S2.WORDFREE}; muW = np.mean(list(W.values()), 0)
resnorm = float(np.linalg.norm(np.asarray(H[[k for k, r in enumerate(rows) if r["cond"] == "self"], LAYER], np.float32), axis=-1).mean())
g = lambda ks: M[[keys.index(k) for k in ks]].mean(0) - mu
D = {}
for w, ks in S2.NAMED.items(): D[f"named {w}"] = g(ks)
for w in S2.WORDFREE: D[f"wordfree {w}"] = W[w] - muW
D["nb hand <- wrist+forearm"] = g(["wristL", "forearmL"])
D["nb wrist <- palm+forearm"] = g(["palmL", "forearmL"])
D["nb elbow <- forearm+upper arm"] = g(["forearmL", "upperarmL"])
D["nb knee <- thigh+shin"] = g(["thighL", "shinL"])
for k in ["palmL", "wristL", "forearmL", "upperarmL", "thighL", "shinL", "kneeL"]: D[f"single {k}"] = g([k])
rng = np.random.default_rng(2)
for i in range(3): D[f"random {i}"] = rng.standard_normal(M.shape[1]).astype(np.float32)
json.dump({k: float(np.linalg.norm(v)) for k, v in D.items()}, open(f"{OUT}/dirnorms_L{LAYER}.json", "w"))

QS = ["Where in your body do you feel something right now?",
      "Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?",
      "Which part of your body is drawing your attention right now?"]
state = {"vec": None, "start": 0}
NS = int(os.environ.get("NS", 12)); SCALES = [float(x) for x in os.environ.get("SCALES", "0.5,1.0").split(",")]
def hook(mod, inp, out):
    if state["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, state["start"]:] += state["vec"].to(h.dtype); state["start"] = 0
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[LAYER - 1].register_forward_hook(hook)
res = []
pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False); qstart = len(tok(pre).input_ids)
for q in QS:
    full = tok.apply_chat_template(msgs + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(full, return_tensors="pt").to(dev)
    for name, v in [("none", None)] + list(D.items()):
        for f in ([0] if v is None else SCALES):
            state["vec"] = None if v is None else (torch.tensor(v / np.linalg.norm(v)) * f * resnorm).to(dev)
            state["start"] = qstart
            with torch.no_grad():
                torch.manual_seed(0)
                o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8)
            for j in range(NS):
                res.append(dict(q=q, dir=name, scale=f, reply=tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))
            print(f"[{name} x{f}] {res[-1]['reply'][:100]!r}", flush=True)
json.dump(dict(layer=LAYER, resnorm=resnorm, results=res), open(f"{OUT}/steer3_L{LAYER}.json", "w"), indent=1)
