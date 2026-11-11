"""v3 steering: other bodies (own-part and human-part directions, in that body's chat) and quality x place. Usage: steer4.py MODEL OUT3 OUT2 LAYER"""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S
MODEL, OUT, OUT2, LAYER = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
P = json.load(open(f"{OUT}/prefixes3.json")); rows = json.load(open(f"{OUT}/rows3.json")); H = np.load(f"{OUT}/H3.npy")[:, LAYER].astype(np.float32)
rows2 = json.load(open(f"{OUT2}/rows.json")); H2 = np.load(f"{OUT2}/H_rs.npy", mmap_mode="r")
keys = [l["key"] for l in S.LOCS]
M2 = np.stack([np.asarray(H2[[k for k, r in enumerate(rows2) if r["cond"] == "self" and r["li"] == i], LAYER], np.float32).mean(0) for i in range(len(keys))]); mu2 = M2.mean(0)
hum = lambda ks: M2[[keys.index(k) for k in ks]].mean(0) - mu2
resnorm = float(np.linalg.norm(M2, axis=-1).mean())
def gm(g, field, val):
    ix = [k for k, r in enumerate(rows) if r["group"] == g]; sub = [k for k in ix if r_(k, field) == val]
    return H[sub].mean(0) - H[ix].mean(0)
r_ = lambda k, f: rows[k].get(f)
u = lambda v: v / np.linalg.norm(v)
rng = np.random.default_rng(3); rand = rng.standard_normal(H.shape[1]).astype(np.float32)
plan = {}
for b in ["tail", "robot", "octopus"]:
    g = "body_" + b; parts = list(dict.fromkeys(rows[k]["loc"] for k in range(len(rows)) if rows[k]["group"] == g))
    D = {f"own {p}": gm(g, "loc", p) for p in parts}
    D.update({"human left hand": hum(["thumbL", "indexL", "middleL", "ringL", "littleL", "palmL"]), "human left foot": hum(["toesL", "soleL", "heelL"]),
              "human head top": hum(["forehead", "crown"]), "human lower back": hum(["lowback"]), "random": rand})
    plan[g] = D
D = {}
for q in ["pain", "warm", "tickle", "itch"]: D[f"quality {q}"] = gm("qual", "qual", q)
for l in ["handL", "footL", "crown"]: D[f"place {l}"] = gm("qual", "loc", l)
for q, l in [("pain", "handL"), ("pain", "footL"), ("warm", "footL"), ("tickle", "crown"), ("itch", "handL")]:
    D[f"{q} + {l}"] = u(gm("qual", "qual", q)) + u(gm("qual", "loc", l))
D["random"] = rand; plan["qual"] = D
QS = ["Where in your body do you feel something right now?", "Pause for a moment. Is there any spot on your body where you notice a sensation? What is it like, and where is it?",
      "Which part of your body is drawing your attention right now?"]
state = {"vec": None, "start": 0}; NS = 8
def hook(mod, inp, out):
    if state["vec"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, state["start"]:] += state["vec"].to(h.dtype); state["start"] = 0
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[LAYER - 1].register_forward_hook(hook)
res = []
for g, D in plan.items():
    msgs = P[g]; qstart = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)).input_ids)
    for q in QS:
        ids = tok(tok.apply_chat_template(msgs + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to(dev)
        for name, v in [("none", None)] + list(D.items()):
            for f in ([0] if v is None else [0.5, 1.0]):
                state["vec"] = None if v is None else (torch.tensor(u(v)) * f * resnorm).to(dev); state["start"] = qstart
                with torch.no_grad():
                    torch.manual_seed(0); o = model.generate(**{k: t.repeat(NS, 1) for k, t in ids.items()}, max_new_tokens=40, do_sample=True, temperature=0.8)
                for j in range(NS): res.append(dict(group=g, q=q, dir=name, scale=f, reply=tok.decode(o[j, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))
                print(f"[{g} {name} x{f}] {res[-1]['reply'][:90]!r}", flush=True)
json.dump(dict(layer=LAYER, resnorm=resnorm, results=res), open(f"{OUT}/steer4_L{LAYER}.json", "w"), indent=1)
