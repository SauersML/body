from common import *; from stim import ROWS, LR, T, OWN
t0 = time.time(); R = list(range(1, NL + 1))
H = states([r["text"] for r in ROWS], R); o = np.array([r["o"] for r in ROWS]); c = np.array([r["c"] for r in ROWS]); slf = o == "self"
HL = states([r["text"] for r in LR], R); side = np.array([r["s"] for r in LR])
res = {}; DIR = {}; MS = {}; MO = {}; LRD = {}; LRM = {}
def dp(p, a, b): return float((p[a].mean() - p[b].mean()) / np.sqrt((p[a].var() + p[b].var()) / 2))
for k, L in enumerate(R):
    X = H[k]; d = X[slf].mean(0) - X[~slf].mean(0); u = d / np.linalg.norm(d); p = X @ u
    loco = []; accs = []
    for cc in T:  # train on 4 categories, test on held-out one
        tr = c != cc; dd = X[tr & slf].mean(0) - X[tr & ~slf].mean(0); thr = ((X[tr & slf] @ dd).mean() + (X[tr & ~slf] @ dd).mean()) / 2
        te = c == cc; q = X[te] @ dd; loco.append(dp(q, slf[te], ~slf[te])); accs.append(float(((q > thr) == slf[te]).mean()))
    # trained only on self vs her, tested on his/woman/friend
    dh = X[slf].mean(0) - X[o == "her"].mean(0); thr = ((X[slf] @ dh).mean() + (X[o == "her"] @ dh).mean()) / 2
    oth = {ow: float((X[o == ow] @ dh < thr).mean()) for ow in ["his", "woman", "friend"]}
    cat_dirs = [X[(c == cc) & slf].mean(0) - X[(c == cc) & ~slf].mean(0) for cc in T]; cn = [v / np.linalg.norm(v) for v in cat_dirs]
    cosc = float(np.mean([cn[i] @ cn[j] for i in range(5) for j in range(i + 1, 5)]))
    XL = HL[k]; dl = XL[side == "left"].mean(0) - XL[side == "right"].mean(0)
    rn = float(np.sqrt((X ** 2).sum(1)).mean())
    res[L] = dict(dprime=dp(p, slf, ~slf), loco_dprime=float(np.mean(loco)), loco_acc=float(np.mean(accs)), her_to_others=oth, cos_categories=cosc,
                  gap_norm=float(np.linalg.norm(d)), resid_norm=rn, lr_gap_norm=float(np.linalg.norm(dl)), lr_dprime=dp(XL @ (dl / np.linalg.norm(dl)), side == "left", side == "right"),
                  cos_so_lr=float(u @ dl / np.linalg.norm(dl)))
    DIR[L] = d; MS[L] = float(X[slf].mean(0) @ u); MO[L] = float(X[~slf].mean(0) @ u); LRD[L] = dl
    print(L, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[L].items()}, flush=True)
np.savez("dirs.npz", **{f"d{L}": DIR[L] for L in R}, **{f"lr{L}": LRD[L] for L in R}, ms=np.array([MS[L] for L in R]), mo=np.array([MO[L] for L in R]))
json.dump(res, open("stage1_dirs.json", "w"), indent=1); print("dirs done", round(time.time() - t0), flush=True)

# ---- forced choice: "about you or about her?" ----
Q = " Is this about you or about her? Answer with one word: me or her."
pairs = [(ROWS[i]["text"], ROWS[j]["text"]) for i in range(len(ROWS)) for j in range(len(ROWS)) if ROWS[i]["o"] == "self" and ROWS[j]["o"] == "her" and ROWS[i]["c"] == ROWS[j]["c"] and ROWS[i]["i"] == ROWS[j]["i"]]
def ids_of(ws): return sorted({tok(w, add_special_tokens=False).input_ids[0] for w in ws})
IME = ids_of(["Me", "me", " me", " Me", "ME"]); IHER = ids_of(["Her", "her", " her", " Her", "HER"])
def enc1(t): return tok(chat([{"role": "user", "content": t + Q}]), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
def LDf(logits): lp = logits[0, -1].float().log_softmax(-1); return torch.logsumexp(lp[IME], 0) - torch.logsumexp(lp[IHER], 0)
def ld_all(texts):
    out = []
    with torch.no_grad():
        for t in texts: out.append(float(LDf(model(enc1(t)).logits)))
    return np.array(out)
base_s = ld_all([a for a, b in pairs]); base_o = ld_all([b for a, b in pairs])
print("LD self", base_s.mean(), (base_s > 0).mean(), "LD her", base_o.mean(), (base_o < 0).mean(), flush=True)
# attribution patching (clean = self, corrupt = her), heads (o_proj input) and MLP neurons (down_proj input)
CAP = {}
def cap(name):
    def f(mod, args):
        x = args[0]
        if x.requires_grad: x.retain_grad()
        CAP[name] = x
    return f
hs_ = [model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(cap(("a", li))) for li in range(NL)] + \
      [model.model.layers[li].mlp.down_proj.register_forward_pre_hook(cap(("m", li))) for li in range(NL)]
for p_ in model.parameters(): p_.requires_grad_(False)
AH = np.zeros((NL, 32)); AM = np.zeros((NL, 12288)); emb = model.model.embed_tokens
for a, b in pairs:
    ia, ib = enc1(a), enc1(b); n = min(ia.shape[1], ib.shape[1])
    with torch.no_grad(): model(ib); corr = {k: v.detach()[0, -n:].float() for k, v in CAP.items()}
    e = emb(ia).detach().requires_grad_(True); out = model(inputs_embeds=e); ld = LDf(out.logits); model.zero_grad(); ld.backward()
    for li in range(NL):
        x = CAP[("a", li)]; at = ((x[0, -n:].float() - corr[("a", li)]) * x.grad[0, -n:].float()).view(n, 32, 128).sum((0, 2)); AH[li] += at.detach().cpu().numpy()
        x = CAP[("m", li)]; at = ((x[0, -n:].float() - corr[("m", li)]) * x.grad[0, -n:].float()).sum(0); AM[li] += at.detach().cpu().numpy()
    CAP.clear()
for h in hs_: h.remove()
AH /= len(pairs); AM /= len(pairs); np.savez("attr.npz", AH=AH, AM=AM)
gap = float(base_s.mean() - base_o.mean()); order = np.dstack(np.unravel_index(np.argsort(-AH.ravel()), AH.shape))[0]
print("LD gap", gap, "sum head attr", AH.sum(), "sum mlp attr", AM.sum(), "top heads", [(int(l), int(h), round(float(AH[l, h]), 3)) for l, h in order[:20]], flush=True)
# mean-ablation test of top-k heads vs random heads
allt = [x for p in pairs for x in p]
def head_means():
    S = {}
    def f(li):
        def g(mod, args): S.setdefault(li, []).append(args[0][0].float().view(-1, 32, 128).mean(0).cpu())
        return g
    hh = [model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(f(li)) for li in range(NL)]
    with torch.no_grad():
        for t in allt: model(enc1(t))
    for h in hh: h.remove()
    return torch.stack([torch.stack(S[li]).mean(0) for li in range(NL)]).to(dev)
HM = head_means(); torch.save(HM, "head_means.pt")
def set_heads(lst):
    HEADS.clear()
    for li in set(l for l, h in lst): HEADS[li] = (torch.tensor([h for l, h in lst if l == li], device=dev), HM[li])
rng = np.random.default_rng(0); abl = {}
for k in [5, 10, 20, 40]:
    top = [(int(l), int(h)) for l, h in order[:k]]; rnd = [(int(x // 32), int(x % 32)) for x in rng.choice(NL * 32, k, replace=False)]
    for nm, lst in [("top", top), ("rand", rnd)]:
        set_heads(lst); s_, o_ = ld_all([a for a, b in pairs]), ld_all([b for a, b in pairs]); pp = ppl()
        abl[f"{nm}{k}"] = dict(gap=float(s_.mean() - o_.mean()), acc=float(((s_ > 0).sum() + (o_ < 0).sum()) / (2 * len(pairs))), ppl=pp)
        print("heads", nm, k, abl[f"{nm}{k}"], flush=True)
HEADS.clear()
# ---- dose sweep: push (single layer), reflect (single layer), ablate (band) ----
def run(edits, label):
    HOOK.clear(); [HOOK.setdefault(L, []).append(fn) for L, fn in edits]
    s_, o_ = ld_all([a for a, b in pairs]), ld_all([b for a, b in pairs]); pp = ppl(); HOOK.clear()
    r = dict(ld_self=float(s_.mean()), self_says_me=float((s_ > 0).mean()), ld_her=float(o_.mean()), her_says_her=float((o_ < 0).mean()), ppl=pp)
    print(label, {a: round(b, 3) for a, b in r.items()}, flush=True); return r
sweep = {"base": run([], "base")}
U = {L: torch.tensor(DIR[L] / np.linalg.norm(DIR[L]), device=dev) for L in R}; DV = {L: torch.tensor(DIR[L], device=dev) for L in R}
for L in [12, 16, 20, 24]:
    for a in [1, 2, 4]:
        sweep[f"toother_L{L}_x{a}"] = run([(L, addv(-a * DV[L]))], f"toother L{L} x{a}")
    sweep[f"reflect_L{L}"] = run([(L, reflect(U[L], (MS[L] + MO[L]) / 2))], f"reflect L{L}")
for b0, b1 in [(8, 20), (12, 24), (16, 28), (8, 32)]:
    sweep[f"ablate_{b0}_{b1}"] = run([(L, ablate(U[L])) for L in range(b0, b1 + 1)], f"ablate L{b0}-{b1}")
json.dump(dict(base_s=base_s.tolist(), base_o=base_o.tolist(), heads_abl=abl, sweep=sweep), open("stage1_fc.json", "w"), indent=1); print("all done", round(time.time() - t0), flush=True)
