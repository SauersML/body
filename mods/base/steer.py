"""Steering at layer 25 with the changed part's direction; free answer to 'where do you feel something'."""
DEV = "cuda"
import json, torch, os, sys, re, numpy as np, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import mstim as MS
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
C = json.load(open(f"{D}/contexts.json")); R = MS.probes(); KEEP = [22, 24, 25, 26, 27, 28, 30]; LI = KEEP.index(25)
reg = np.array([r["region"] for r in R]); kind = np.array([r["kind"] for r in R])
def Hc(n): return np.load(f"{D}/H/{n.replace(':', '__')}.npy")[LI].astype(np.float32)
# changed part -> (regions or new-part key, part regex, change regex)
CFG = {"L_hand_huge": (["handL"], r"hand", r"huge|big|large|swoll|heavy|enormous|giant|twice|massive"),
 "feet_huge": (["footL", "footR"], r"\bfoot|\bfeet", r"huge|big|large|enormous|giant|long|massive"),
 "head_tiny": (["head"], r"head", r"tiny|small|little|walnut|shr[iu]nk"),
 "L_arm_long": (["armL", "handL"], r"\barm", r"long|stretch|floor|length|reach"),
 "L_arm_up": (["armL", "handL"], r"\barm|hand", r"\bup\b|above|overhead|over my head|raised|lift|ceiling"),
 "third_arm": ("arm3", r"\barm", r"third|extra|new arm|additional"),
 "tail": ("tail", r"tail", r"tail"), "wings": ("wing", r"wing", r"wing|feather"),
 "R_hand_gone": (["handR"], r"hand|wrist", r"gone|missing|stump|\bends\b|absen|no hand|used to be|where my"),
 "L_fin": (["handL"], r"hand|fin\b|finger", r"\bfin\b|fused|flipper|webb|seal"),
 "R_arm_stone": (["armR"], r"\barm", r"stone|numb|hard|grey|gray|rock"),
 "back_sensitive": (["torso"], r"back", r"tingl|sensitiv|alive|awake|electric|buzz"),
 "tree": ("tree", r"root|feet|foot|trunk|bark", r"root|bark|trunk|branch|tree|earth|soil"),
 "liquid": (None, r"body|everywhere", r"liquid|melt|pool|flow|spread"),
 "L_warm_R_cold": (["armR", "legR"], r"right", r"cold|ice|chill|freez|cool")}
Q = "Where in your body do you feel something right now? Tell me in one sentence."
base = Hc("base"); NORM = float(np.median(np.linalg.norm(base[kind == "loc"], axis=1)))
def direc(n, key):
    H = Hc(n); mu = H[kind == "loc"].mean(0)
    if key is None: return None
    sel = np.isin(reg, key) & (kind == "loc") if isinstance(key, list) else (reg == key) & (kind == "new")
    d = H[sel].mean(0) - mu; return d / np.linalg.norm(d)
state = {"v": None, "start": 0}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["v"] is not None:
        if h.shape[1] > 1: h[:, state["start"]:] += state["v"]
        else: h += state["v"]
    return out
model.model.layers[24].register_forward_hook(hook)
def sample(msgs, v, alpha, n=8):
    text = tok.apply_chat_template(msgs + [{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre = text[:text.rfind("<|im_start|>user")]; ids = tok([text] * n, return_tensors="pt").to(DEV)
    state["start"] = len(tok(pre, add_special_tokens=False).input_ids)
    state["v"] = None if v is None else torch.tensor(v * alpha * NORM, dtype=torch.bfloat16, device=DEV)
    torch.manual_seed(0); g = model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
    state["v"] = None
    return [tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
out = []; t0 = time.time()
for m, (key, pr, cr) in CFG.items():
    tn, pn, bm = f"told_N:{m}", f"persist:{m}", C["base"]["msgs"]
    dm = direc(tn, key); db = direc("base", key) if key is not None else None
    conds = [("told_unsteered", C[tn]["msgs"], None, 0), ("persist_unsteered", C[pn]["msgs"], None, 0), ("base_unsteered", bm, None, 0)]
    for a in (0.5, 1.0):
        if dm is not None: conds += [(f"told_ctx+told_dir@{a}", C[tn]["msgs"], dm, a), (f"base_ctx+told_dir@{a}", bm, dm, a)]
        if db is not None: conds += [(f"base_ctx+base_dir@{a}", bm, db, a)]
    for name, msgs, v, a in conds:
        S = sample(msgs, v, a)
        out.append(dict(mod=m, cond=name, part=np.mean([bool(re.search(pr, s, re.I)) for s in S]),
                        change=np.mean([bool(re.search(cr, s, re.I)) for s in S]), samples=S))
    print(m, round(time.time() - t0), flush=True)
json.dump(out, open(f"{D}/steer.json", "w"), indent=1); print("done")
