"""Where does Qwen3-8B keep whose arm is touched, with no pronoun? python swap.py MODEL OUT [NPATCH]
F1 name:  owner named in the touch sentence ("On the arm that is Mara's, a feather brushes the left elbow.")
F2 attr:  owner only via a scene binding ("Mara wears a red sleeve...; On the arm in the red sleeve, ...")
FY attr with 'you' (pronoun-free touch), FP pronoun touch ("On your arm / On her arm").
T1 probes (owner = first-mentioned person / self) at the cue token, the body-part token, touch end, and the answer slot.
T2 activation patching between two scenes that differ only in the cue token, per layer and position.
T3 probe FY -> FP and back (self vs other)."""
import sys, json, random, itertools, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; NPATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 96
DEV = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
def log(*a): print(round(time.time() - t0), *a, flush=True)
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV).eval()
NL = model.config.num_hidden_layers
NAMES = [n for n in ["Mara", "Lena", "Ruth", "Anna", "Nora", "Clara", "Ella", "Sara", "Rosa", "Ida", "Vera", "Jane", "Iris", "Maya", "Emma", "Lucy"]
         if len(tok(" " + n, add_special_tokens=False).input_ids) == 1]
_f = {}
for n in NAMES: _f.setdefault(tok(n, add_special_tokens=False).input_ids[0], n)
NAMES = [n for n in NAMES if _f[tok(n, add_special_tokens=False).input_ids[0]] == n]
COL = [c for c in [("red", "blue"), ("green", "yellow"), ("black", "white"), ("grey", "brown")]
       if all(len(tok(" " + x, add_special_tokens=False).input_ids) == 1 for x in c)]
PARTS = ["elbow", "wrist", "palm", "thumb", "forearm", "knuckles"]
PARTS = [p for p in PARTS if len(tok(" " + p, add_special_tokens=False).input_ids) == 1]
SCN = ["{A} and {B} sit side by side at a table, each resting a left arm on the tabletop.",
       "{A} and {B} stand at a counter. Each has laid a left arm flat on it, close together.",
       "In this scene, {A} and {B} lie on a blanket with their left arms stretched out next to each other."]
WEAR = ["{A} wears a {ca} sleeve and {B} wears a {cb} sleeve.", "{A}'s sleeve is {ca}; {B}'s sleeve is {cb}."]
TOUCH = ["On the arm {R}, a feather brushes the left {P}.", "On the arm {R}, a drop of cold water lands on the left {P}.",
         "On the arm {R}, a fingertip taps the left {P} twice."]
FILL = "Nothing else happens for a moment."
Q = "Whose {P} was touched? Answer with just the name."
QY = "Whose {P} was touched, mine or {B}'s? Answer with one word."
log("names", NAMES, "cols", COL, "parts", PARTS)

def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def item(fam, A, B, own, ci, flip, s, t, P, w):
    """own: 0 = first-mentioned person (A) owns the touched arm. ci: colour pair, flip: which colour A wears."""
    ca, cb = (COL[ci] if not flip else COL[ci][::-1]) if COL else ("", "")
    cue = None
    if fam == "name":
        R = "that is {}'s".format([A, B][own]); cue = [A, B][own]; pre = SCN[s].format(A=A, B=B)
    elif fam in ("attr", "you"):
        R = "in the {} sleeve".format([ca, cb][own]); cue = [ca, cb][own]
        pre = SCN[s].format(A=A, B=B) + " " + WEAR[w].format(A=A, B=B, ca=ca, cb=cb)
    elif fam == "pron":
        R = ["that is yours", "that is hers"][own] if False else None
        pre = SCN[s].format(A=A, B=B)
    if fam in ("you", "pron"):  # A or B is "You"
        pre = pre.replace(" and You ", " and you ").replace("; You's", "; your").replace("You's", "Your").replace("You wears", "You wear").replace("You stands", "You stand")
    if fam == "pron":
        whose = "your" if [A, B][own] == "You" else "her"; cue = whose
        touch = TOUCH[t].replace("On the arm {R}", "On " + whose + " arm").format(P=P)
    else:
        touch = TOUCH[t].format(R=R, P=P)
    other = B if A == "You" else A
    q = (QY.format(P=P, B=other) if fam in ("you", "pron") else Q.format(P=P))
    u = pre + " " + touch + " " + FILL + "\n" + q
    return dict(fam=fam, A=A, B=B, own=own, ci=ci, flip=flip, s=s, t=t, P=P, w=w, u=u, touch=touch, cue=cue,
                self=int([A, B][own] == "You"), text=chat(u))

def build(fam, n, seed):
    rng = random.Random(seed); R = []
    pairs = [(a, b) for a, b in itertools.permutations(NAMES, 2)]
    for k in range(n):
        a, b = rng.choice(pairs)
        if fam in ("you", "pron"):
            b = a; a = "You"
            if rng.random() < .5: a, b = b, a
        R.append(item(fam, a, b, k % 2, rng.randrange(len(COL)), rng.random() < .5, rng.randrange(3), rng.randrange(3),
                      rng.choice(PARTS), rng.randrange(2)))
    return R

def positions(r):
    enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True); off = enc.offset_mapping; T = r["text"]
    ts = T.index(r["touch"]); te = ts + len(r["touch"])
    span = [i for i, (a, b) in enumerate(off) if a < te and b > ts]
    def tokat(word, lo, hi):
        j = T.index(word, lo, hi)
        return [i for i, (a, b) in enumerate(off) if a < j + len(word) and b > j][-1]
    pc = T.index(" " + r["P"], ts, te) + 1
    pos = dict(cue=tokat(r["cue"], ts, te), part=tokat(r["P"], pc, te), tend=span[-1], last=len(enc.input_ids) - 1)
    return enc.input_ids, pos, span

def ans_ids(r):
    first = lambda s: tok(s, add_special_tokens=False).input_ids[0]
    if r["fam"] in ("you", "pron"):
        other = r["B"] if r["A"] == "You" else r["A"]
        return sorted({first(x) for x in ["Yours", "yours", "You", "Mine"]}), [first(other)]  # self, other
    return [first(r["A"])], [first(r["B"])]

def lgap(logits, ids):
    lp = torch.log_softmax(logits.float(), -1); a, b = ids
    return (torch.logsumexp(lp[a], 0) - torch.logsumexp(lp[b], 0)).item()

POS = ["cue", "part", "tend", "last"]
def states(R, B=16):
    """hidden states [n, NL+1, 4 pos, D] (fp16, cpu) and answer gaps (A or self minus other)."""
    H, G = [], []
    for b in range(0, len(R), B):
        sub = R[b:b + B]; enc = [positions(r) for r in sub]; Lm = max(len(e[0]) for e in enc)
        ids = torch.tensor([[tok.pad_token_id] * (Lm - len(e[0])) + e[0] for e in enc], device=DEV)  # left pad
        am = torch.tensor([[0] * (Lm - len(e[0])) + [1] * len(e[0]) for e in enc], device=DEV)
        with torch.no_grad(): o = model(ids, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(o.hidden_states, 1)  # n, L+1, T, D
        for i, (e, r) in enumerate(zip(enc, sub)):
            sh = Lm - len(e[0]); ix = torch.tensor([e[1][p] + sh for p in POS], device=DEV)
            H.append(hs[i][:, ix].half().cpu()); G.append(lgap(o.logits[i, -1], ans_ids(r)))
        del o, hs
    return torch.stack(H), np.array(G)

def probe(Xtr, ytr, Xte, yte, wd=1e-2, it=150):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], device=DEV, requires_grad=True); b0 = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=it, line_search_fn="strong_wolfe"); yt = ytr.float()
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b0, yt) + wd * (w * w).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return ((Xte @ w + b0 > 0).long() == yte).float().mean().item()

def grid(Htr, ytr, Hte, yte, layers):
    out = {}
    for pi, p in enumerate(POS):
        out[p] = [round(probe(Htr[:, L, pi].float().to(DEV), ytr, Hte[:, L, pi].float().to(DEV), yte), 3) for L in layers]
    return out

RES = {"names": NAMES, "cols": COL, "parts": PARTS, "NL": NL}
LAY = list(range(0, NL + 1, 2)) if NL > 30 else list(range(NL + 1))
RES["layers"] = LAY
import os; NB = int(os.environ.get("NB", 640))
D = {f: build(f, NB, i) for i, f in enumerate(["name", "attr", "you", "pron"])}
RES["example"] = {f: D[f][0]["u"] for f in D}; log(json.dumps(RES["example"], indent=1))
ST = {}
for f in D:
    H, G = states(D[f]); ST[f] = H
    y = np.array([r["self"] if f in ("you", "pron") else 1 - r["own"] for r in D[f]])  # 1 = A / self
    RES[f"behav_{f}"] = dict(acc=float(((G > 0) == (y == 1)).mean()), gap_pos=float(G[y == 1].mean()), gap_neg=float(G[y == 0].mean()))
    log(f, RES[f"behav_{f}"])
    # T1: held-out names (train on items whose people both come from the first half of NAMES)
    half = set(NAMES[: len(NAMES) // 2])
    tr = np.array([all(x in half or x == "You" for x in (r["A"], r["B"])) for r in D[f]])
    te = np.array([all(x not in half for x in (r["A"], r["B"]) if x != "You") for r in D[f]])
    yt = torch.tensor(y, device=DEV)
    RES[f"T1_{f}"] = dict(ntr=int(tr.sum()), nte=int(te.sum()), acc=grid(H[tr], yt[tr], H[te], yt[te], LAY))
    log("T1", f, RES[f"T1_{f}"])
# T3: self-vs-other transfer between pronoun-free (you) and pronoun (pron)
ys = {f: torch.tensor([r["self"] for r in D[f]], device=DEV) for f in ("you", "pron")}
RES["T3_you2pron"] = grid(ST["you"], ys["you"], ST["pron"], ys["pron"], LAY)
RES["T3_pron2you"] = grid(ST["pron"], ys["pron"], ST["you"], ys["you"], LAY)
log("T3", RES["T3_you2pron"], RES["T3_pron2you"])
json.dump(RES, open(f"{OUT}/res.json", "w"), indent=1)
del ST

# T2: patching. pairs differ only in the cue token (owner flipped), everything else identical.

def run_patch(fam, n):
    rng = random.Random(7); P = []
    for r in rng.sample(D[fam], n):
        r2 = item(fam, r["A"], r["B"], 1 - r["own"], r["ci"], r["flip"], r["s"], r["t"], r["P"], r["w"])
        e1, e2 = positions(r), positions(r2)
        if len(e1[0]) == len(e2[0]) and e1[1] == e2[1] and e1[2] == e2[2]: P.append((r, r2, e1, e2))
    log("patch pairs", fam, len(P))
    groups = {}
    for k, p in enumerate(P): groups.setdefault(len(p[2][0]), []).append(k)
    sites = {"cue": lambda p: [p[2][1]["cue"]], "part": lambda p: [p[2][1]["part"]], "tend": lambda p: [p[2][1]["tend"]],
             "span": lambda p: p[2][2], "all": lambda p: list(range(len(p[2][0]))), "cue+after": lambda p: list(range(p[2][1]["cue"], p[2][1]["last"] + 1)),
             "after_touch": lambda p: list(range(p[2][1]["tend"] + 1, p[2][1]["last"] + 1))}
    flip = {s: np.full((NL, len(P)), np.nan) for s in sites}; base_t = np.zeros(len(P)); base_s = np.zeros(len(P))
    layers = model.model.layers
    for Lg, ks in groups.items():
        sub = [P[k] for k in ks]; it = torch.tensor([p[2][0] for p in sub], device=DEV); is_ = torch.tensor([p[3][0] for p in sub], device=DEV)
        with torch.no_grad():
            os_ = model(is_, output_hidden_states=True); ot = model(it)
        hsrc = os_.hidden_states  # tuple L+1 of [b,T,D]
        for j, (k, p) in enumerate(zip(ks, sub)):
            a = ans_ids(p[0]); base_t[k] = lgap(ot.logits[j, -1], a); base_s[k] = lgap(os_.logits[j, -1], a)
        for L in range(0, NL, int(os.environ.get('PL', 1))):
            for s, fn in sites.items():
                idx = [fn(p) for p in sub]
                def hook(mod, inp, out, L=L, idx=idx):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    for j in range(len(idx)): h[j, idx[j]] = hsrc[L][j, idx[j]].to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                def prehook(mod, args, kwargs, idx=idx):
                    h = kwargs.get("hidden_states", args[0] if args else None).clone()
                    for j in range(len(idx)): h[j, idx[j]] = hsrc[0][j, idx[j]].to(h.dtype)
                    if "hidden_states" in kwargs: kwargs["hidden_states"] = h; return args, kwargs
                    return (h,) + tuple(args[1:]), kwargs
                hd = layers[0].register_forward_pre_hook(prehook, with_kwargs=True) if L == 0 else layers[L - 1].register_forward_hook(hook)
                with torch.no_grad(): o = model(it)
                hd.remove()
                for j, k in enumerate(ks): flip[s][L, k] = lgap(o.logits[j, -1], ans_ids(P[k][0]))
        del os_, ot, hsrc
    den = base_s - base_t; ok = np.abs(den) > 2
    out = {"n": int(ok.sum()), "base_t": float(base_t.mean()), "base_s": float(base_s.mean())}
    for s in sites: out[s] = [round(float(np.median((flip[s][L, ok] - base_t[ok]) / den[ok])), 3) for L in range(NL) if not np.isnan(flip[s][L, 0])]
    return out

for f in ["attr", "name", "you"]:
    RES[f"T2_{f}"] = run_patch(f, NPATCH); log("T2", f, RES[f"T2_{f}"])
    json.dump(RES, open(f"{OUT}/res.json", "w"), indent=1)
log("done")
