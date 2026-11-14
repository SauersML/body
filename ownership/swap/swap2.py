"""Follow-ups to swap.py (Qwen3-8B). python swap2.py MODEL OUT SWAP_PY CTL_DIR
1. Self/other direction d_L (L22/24/26) fit ONLY on pronoun scenes ("On your arm / On her arm"; names from half A, scenes 0-1, touches 0-1),
   = mean(self) - mean(other) of the residual averaged over touch-end..last token. Added (alpha * d) at layer L to those positions in
   pronoun-free colour-binding scenes ("You and Mara... On the arm in the red sleeve") with held-out names. Other-owned items get +d, self-owned -d.
   Read: Q1 "Whose P, mine or X's?", Qo "Whose P? one word" (self words vs X), Qid "Besides me, who else is in the scene?" (X vs other names).
   Baseline: random directions of the same norm.
2. Cosine of d_L with the §7 owner axis uY (embodiment chat, your - the woman's, last token, even templates), rebuilt at L22/24/26."""
import sys, os, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # Qwen/Qwen3-8B
MODEL, OUT, SWAP, CTL = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
src = open(SWAP).read(); src = src[:src.index('RES = {"names"')]
sys.argv = [SWAP, MODEL, OUT]; exec(src)
sys.path.insert(0, CTL); import stimuli as S
SM = os.environ.get("SMOKE"); LS = [22, 24, 26]; AL = [2.0] if SM else [2.0, 4.0, 8.0]; NR = 1 if SM else 8; NT = 8 if SM else 160
half = set(NAMES[: len(NAMES) // 2])
oth = lambda r: r["B"] if r["A"] == "You" else r["A"]
FIT = [r for r in build("pron", 100 if SM else 2400, 11) if oth(r) in half and r["s"] < 2 and r["t"] < 2]
TE = [r for r in build("you", 2400, 12) if oth(r) not in half]
TE = [r for r in TE if r["self"] == 0][:NT] + [r for r in TE if r["self"] == 1][:NT]
log("fit", len(FIT), "test", len(TE), "heldout names", sorted(set(map(oth, TE))))
QO = "Whose {P} was touched? Answer with one word."; QID = "Besides me, who else is in the scene? Answer with just the name."
first = lambda s: tok(s, add_special_tokens=False).input_ids[0]
SELFW = sorted({first(x) for x in ["Mine", "mine", "Me", "My", "You", "Yours", "I"]})
NID = {n: first(n) for n in NAMES}
def variant(r, q):
    u = r["u"].rsplit("\n", 1)[0] + "\n" + q.format(P=r["P"]); d = dict(r); d["u"] = u; d["text"] = chat(u); return d
QS = {"q1": lambda r: r, "qo": lambda r: variant(r, QO), "qid": lambda r: variant(r, QID)}

def forward(R, L=None, vecs=None, B=32, want_h=False):
    """vecs: [n, D] added at layer L output over touch-end..last. Returns per-item last logits (log-softmaxed subsets) or span-mean/last states."""
    outs = []
    for b in range(0, len(R), B):
        sub = R[b:b + B]; enc = [positions(r) for r in sub]; Lm = max(len(e[0]) for e in enc)
        ids = torch.tensor([[tok.pad_token_id] * (Lm - len(e[0])) + e[0] for e in enc], device=DEV)
        am = torch.tensor([[0] * (Lm - len(e[0])) + [1] * len(e[0]) for e in enc], device=DEV)
        rng_ = [(Lm - len(e[0]) + e[1]["tend"], Lm) for e in enc]; hd = None
        if vecs is not None:
            V = vecs[b:b + B]
            def hook(mod, inp, out, V=V, rng_=rng_):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1: return out
                h = h.clone()
                for j, (a, z) in enumerate(rng_): h[j, a:z] += V[j].to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hd = model.model.layers[L - 1].register_forward_hook(hook)
        with torch.no_grad(): o = model(ids, attention_mask=am, output_hidden_states=want_h)
        if hd: hd.remove()
        for j, r in enumerate(sub):
            if want_h:
                a, z = rng_[j]; outs.append({k: (o.hidden_states[k][j, a:z].float().mean(0).cpu(), o.hidden_states[k][j, -1].float().cpu()) for k in LS})
            else: outs.append(torch.log_softmax(o.logits[j, -1].float(), -1).cpu())
        del o
    return outs
def scores(R, lps):
    s = {"q1": [], "qo": [], "qid": [], "qid_acc": []}
    for r, (l1, lo, li) in zip(R, lps):
        x = NID[oth(r)]; s["q1"].append(lgap(l1, ans_ids(r))); s["qo"].append((torch.logsumexp(lo[SELFW], 0) - lo[x]).item())
        others = [v for n, v in NID.items() if n != oth(r)]
        s["qid"].append((li[x] - torch.logsumexp(li[others], 0)).item()); s["qid_acc"].append(float(li[x] >= li[others].max()))
    return {k: np.array(v) for k, v in s.items()}
TQ = {q: [f(r) for r in TE] for q, f in QS.items()}
def evaluate(L=None, vecs=None):
    lp = {q: forward(TQ[q], L, vecs) for q in QS}; return scores(TE, list(zip(lp["q1"], lp["qo"], lp["qid"])))
y = np.array([r["self"] for r in TE]); O, Sf = y == 0, y == 1
def summ(sc):
    return dict(q1_self_other=float((sc["q1"][O] > 0).mean()), q1_self_self=float((sc["q1"][Sf] > 0).mean()),
                q1_gap_other=float(sc["q1"][O].mean()), q1_gap_self=float(sc["q1"][Sf].mean()),
                qo_self_other=float((sc["qo"][O] > 0).mean()), qo_self_self=float((sc["qo"][Sf] > 0).mean()),
                qid_acc=float(sc["qid_acc"].mean()), qid_gap=float(sc["qid"].mean()))
RES = {"fit_n": len(FIT), "test_n": len(TE), "example_fit": FIT[0]["u"], "example_test": {q: TQ[q][0]["u"] for q in QS}}
# fit d_L
HF = forward(FIT, want_h=True); yf = np.array([r["self"] for r in FIT])
lpf = forward(FIT); RES["fit_behav"] = float(np.mean([(lgap(l, ans_ids(r)) > 0) == r["self"] for r, l in zip(FIT, lpf)]))
D_, DL = {}, {}
for L in LS:
    A_ = torch.stack([h[L][0] for h in HF]); Z = torch.stack([h[L][1] for h in HF])
    D_[L] = A_[yf == 1].mean(0) - A_[yf == 0].mean(0); DL[L] = Z[yf == 1].mean(0) - Z[yf == 0].mean(0)
    log("d", L, float(D_[L].norm()), "resid norm", float(A_.norm(dim=1).mean()))
RES["d_norm"] = {L: float(D_[L].norm()) for L in LS}
base = evaluate(); RES["base"] = summ(base); log("base", RES["base"])
sgn = torch.tensor(np.where(y == 0, 1.0, -1.0), dtype=torch.float32)[:, None]
g = torch.Generator().manual_seed(0); RES["steer"] = {}
for L in LS:
    for a in AL:
        v = a * D_[L].to(DEV); key = f"L{L}_a{a}"
        r_ = summ(evaluate(L, sgn.to(DEV) * v)); rr = []
        for k in range(NR):
            u = torch.randn(v.shape[0], generator=g).to(DEV); u = u / u.norm() * v.norm()
            rr.append(summ(evaluate(L, sgn.to(DEV) * u)))
        RES["steer"][key] = dict(d=r_, rand_mean={k: float(np.mean([x[k] for x in rr])) for k in rr[0]},
                                 rand_max={k: float(np.max([x[k] for x in rr])) for k in rr[0]})
        log(key, "d", r_, "rand", RES["steer"][key]["rand_mean"])
        json.dump(RES, open(f"{OUT}/res2.json", "w"), indent=1)
# §7 owner axis uY at the same layers
PRE = json.load(open(f"{CTL}/prefix.json"))
rows = [r for r in S.build() if r["cond"] in ("self", "other")][: 64 if SM else None]
EV = np.array([r["tpl"] % 2 == 0 for r in rows]); cs = np.array([r["cond"] == "self" for r in rows])
HU = {L: [] for L in LS}
txt = [tok.apply_chat_template(PRE + [{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
for b in range(0, len(txt), 16):
    e = [tok(t, add_special_tokens=False).input_ids for t in txt[b:b + 16]]; Lm = max(map(len, e))
    ids = torch.tensor([[tok.pad_token_id] * (Lm - len(x)) + x for x in e], device=DEV)
    am = torch.tensor([[0] * (Lm - len(x)) + [1] * len(x) for x in e], device=DEV)
    with torch.no_grad(): o = model(ids, attention_mask=am, output_hidden_states=True)
    for L in LS: HU[L].append(o.hidden_states[L][:, -1].float().cpu())
    del o
cos = lambda a, b: float(a @ b / a.norm() / b.norm())
RES["cos"] = {}
for L in LS:
    X = torch.cat(HU[L]); uY = X[torch.tensor(EV & cs)].mean(0) - X[torch.tensor(EV & ~cs)].mean(0)
    RES["cos"][L] = dict(span_d=cos(uY, D_[L]), last_d=cos(uY, DL[L]), span_vs_last=cos(D_[L], DL[L]), uY_norm=float(uY.norm()))
    if L == 22: uY22 = uY
    log("cos", L, RES["cos"][L])
RES["cos"]["uY24_d22"] = cos(torch.cat(HU[24])[torch.tensor(EV & cs)].mean(0) - torch.cat(HU[24])[torch.tensor(EV & ~cs)].mean(0), D_[22])
# steer with uY22 rescaled to |d22|
for a in AL:
    v = a * (uY22 / uY22.norm() * D_[22].norm()).to(DEV); RES["steer"][f"uY22_a{a}"] = summ(evaluate(22, sgn.to(DEV) * v)); log("uY22", a, RES["steer"][f"uY22_a{a}"])
# a few greedy answers, L22 alpha 2, other-owned items
gen = []
for r in TE[:1 if SM else 4]:
    for q in ("qo", "qid"):
        rq = QS[q](r); e = positions(rq); ids = torch.tensor([e[0]], device=DEV); outs = []
        for v in (None, 2 * D_[22].to(DEV)):
            hd = None
            if v is not None:
                def hook(mod, inp, out, v=v, a0=e[1]["tend"]):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] == 1: return out
                    h = h.clone(); h[0, a0:] += v.to(h.dtype); return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hd = model.model.layers[21].register_forward_hook(hook)
            with torch.no_grad(): gg = model.generate(ids, max_new_tokens=8, do_sample=False)
            if hd: hd.remove()
            outs.append(tok.decode(gg[0, ids.shape[1]:], skip_special_tokens=True))
        gen.append(dict(q=rq["u"], base=outs[0], steer=outs[1]))
RES["gen"] = gen; log(gen)
json.dump(RES, open(f"{OUT}/res2.json", "w"), indent=1); log("done")
