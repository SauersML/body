"""python run2.py MODEL TAG L1,L2,L3 : single-layer subspace ablation on two-choice spatial QA.
Subspaces: top-k PCs of a source body's 16 centred part means (touch sentences, saved states). Ablation at ONE layer, all positions.
Adds a non-body spatial control (a house). Saves per-question order-averaged margins (>0 = correct)."""
import sys, json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bodies
from bodies import *
B["house"] = [("chimney", .5, -1, 9.5), ("roof peak", 0, 0, 8.5), ("attic window", 0, 4, 7.5), ("left gutter", -4, 0, 6), ("right gutter", 4, 0, 6),
              ("upstairs left window", -2.5, 4, 5), ("upstairs right window", 2.5, 4, 5), ("balcony", 0, 4.5, 4.5), ("front door", 0, 4, 1),
              ("doormat", 0, 4.6, 0), ("downstairs left window", -2.5, 4, 1.5), ("downstairs right window", 2.5, 4, 1.5), ("back door", 0, -4, 1),
              ("garage door", 6, 4, 1), ("basement", 0, 0, -2), ("foundation", 0, 0, -.5)]
NOUN["house"] = "a two-storey house"
M, TAG = sys.argv[1:3]; LAYS = [int(x) for x in sys.argv[3].split(",")]
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
LAY = model.model.layers; t0 = time.time()
Z = np.load(f"out/{TAG}_states.npz", allow_pickle=True); meta = Z["meta"]
bo = np.array([m[0] for m in meta]); pi = np.array([int(m[1]) for m in meta])
def sub(b, l, k):
    X = Z[f"L{l}"].astype(np.float32)[bo == b]; p = pi[bo == b]
    P = np.stack([X[p == i].mean(0) for i in range(16)]); P -= P.mean(0)
    return np.linalg.svd(P, full_matrices=False)[2][:k]
d = Z[f"L{LAYS[0]}"].shape[1]; rng = np.random.default_rng(0)
qs = []
for b in BODIES + ["house"]:
    for q, a, w, kind in qa(b, np.random.default_rng(1)):
        if b == "house": q = q.replace("Think of a two-storey house in its normal standing or resting position.", "Think of a two-storey house seen from the street.")
        for order in (0, 1):
            A, Bo = (a, w) if order == 0 else (w, a)
            msg = f"{q}\n(A) {A}\n(B) {Bo}\nReply with just A or B."
            p = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            qs.append((b, kind, order, p))
ida, idb = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
HOOK = {"U": None, "l": None}
def mk(l):
    def f(mod, inp, out):
        if HOOK["U"] is None or HOOK["l"] != l: return
        h = out[0] if isinstance(out, tuple) else out
        U = HOOK["U"]; h = h - (h @ U.T) @ U
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return f
for l in LAYS: LAY[l].register_forward_hook(mk(l))
@torch.no_grad()
def margins(bs=48):
    r = []
    for k in range(0, len(qs), bs):
        e = tok([q[3] for q in qs[k:k + bs]], return_tensors="pt", padding=True).to("cuda")
        lg = model(**e).logits[:, -1].float(); r.append((lg[:, ida] - lg[:, idb]).cpu())
    m = torch.cat(r).numpy() * np.array([1 if q[2] == 0 else -1 for q in qs]); return (m[0::2] + m[1::2]).tolist()
res = {"qs": [q[:2] for q in qs[0::2]], "cond": {"none": margins()}}; print("base", time.time() - t0, flush=True)
for l in LAYS:
    for k in (4, 8):
        for src in ["human", "dog", "robot", "bird", "velk", "ctrl", "rand"]:
            U = np.linalg.qr(rng.standard_normal((d, k)))[0].T if src == "rand" else sub(src, l, k)
            HOOK["U"] = torch.tensor(U, dtype=torch.bfloat16, device="cuda"); HOOK["l"] = l
            res["cond"][f"{src}_L{l}_k{k}"] = margins(); print(src, l, k, time.time() - t0, flush=True)
HOOK["U"] = None
json.dump(res, open(f"out/{TAG}_qa2.json", "w")); print("DONE", time.time() - t0, flush=True)
