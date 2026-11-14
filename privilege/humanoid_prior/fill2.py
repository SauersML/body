"""add-on: fill-in states with neutral 'upper/lower' and 'one/other' limb wordings (the main run used quadruped 'front/back')."""
import sys, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + "/../nonhuman"); import stimuli as S
MODEL, D = sys.argv[1], sys.argv[2]; tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval(); L = int(model.config.num_hidden_layers * 52 / 64)
PRE = json.load(open(f"{D}/prefixes.json")); chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
W = {"upper": "the end of your left upper limb", "lower": "the end of your left lower limb", "A": "the end of your first limb", "B": "the end of your third limb"}
out = {}
for f in [k for k in PRE if k.startswith("f")]:
    for w, ph in W.items():
        ts = [chat(PRE[f] + [{"role": "user", "content": t.format(L=ph)}]) for t in S.TEMPLATES]
        b = tok(ts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): h = model.model(**b, output_hidden_states=True).hidden_states[L][:, -1].float().mean(0).cpu().numpy()
        out[(f, w)] = h
X = np.load(f"{D}/X.npy")[:, 1].astype(np.float32); rows = json.load(open(f"{D}/rows.json"))
c = lambda n, k: X[[i for i, r in enumerate(rows) if r["chat"] == n and r["loc"] == k]].mean(0)
ref = {"person hand-foot": c("16human", "8") - c("16human", "13"), "dog front-hind paw": c("16dog", "8") - c("16dog", "13")}
cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)); res = {}
for f in dict.fromkeys(k[0] for k in out):
    for pair in [("upper", "lower"), ("A", "B")]:
        d = out[(f, pair[0])] - out[(f, pair[1])]; res[f"{f} | {pair[0]}-{pair[1]}"] = {k: cs(d, v) for k, v in ref.items()}
json.dump(res, open(f"{D}/fill2.json", "w"), indent=1); [print(k, {a: round(b, 2) for a, b in v.items()}) for k, v in res.items()]
