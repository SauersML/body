"""Rearranged bodies vs rearranged objects in Qwen3-32B: what carries the body advantage? python nov_run.py OUT MODEL PHASES
phases: base, states, dirs (ablate/steer item-fitted directions), das, attr (attribution patching), patch (validate top components)
margin = log p(correct first token) - log p(wrong first token) at the answer position (casings pooled, shared first tokens dropped)."""
import json, os, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT, M, PH = sys.argv[1], sys.argv[2], sys.argv[3].split(","); os.makedirs(OUT, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map=dev).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg = model.config; NL, D, H = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads; HD = getattr(cfg, "head_dim", D // H)
FR = NL / 64; LY = lambda l: max(1, min(NL, round(l * FR)))
ITEMS = json.load(open("items_nov.json"))
if os.environ.get("SMOKE"):  # scenarios 0-1 only, twins re-indexed
    keep = [it for it in ITEMS if it["scen"] < 2]; idm = {it["id"]: k for k, it in enumerate(keep)}
    for it in keep: it["twin"] = idm[it["twin"]]
    ITEMS = keep
N = len(ITEMS)
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ENC = [tok(chat(it["q"] + " Answer with one word."), add_special_tokens=False).input_ids for it in ITEMS]
def ft(w): return {tok(v, add_special_tokens=False).input_ids[0] for v in (w, w[0].upper() + w[1:])}
TA, TB = [], []
for it in ITEMS: a, b = ft(it["a"]), ft(it["b"]); TA.append(sorted(a - b)); TB.append(sorted(b - a))
G = {t: [i for i, it in enumerate(ITEMS) if it["task"] == t] for t in ["body_swap", "body_norm", "obj_swap", "obj_norm"]}
FOLD = np.array([it["scen"] % 2 for it in ITEMS]); SWAP = [i for i, it in enumerate(ITEMS) if it["task"].endswith("swap")]
def J(o, f): json.dump(o, open(f"{OUT}/{f}", "w"))
def LJ(f): return json.load(open(f"{OUT}/{f}")) if os.path.exists(f"{OUT}/{f}") else {}

# ---- residual hooks (output of decoder layer L == hidden_states[L]) ----
HOOK = {}; POS = {"m": "all"}; LAST = {"i": None}
def rh(L):
    def f(mod, inp, out):
        ops = HOOK.get(L)
        if not ops: return out
        h = out[0] if isinstance(out, tuple) else out; hf = h.float(); new = hf
        for op in ops:
            if op[0] == "proj": new = new - (new @ op[1]) @ op[1].T
            elif op[0] == "add": new = new + op[1]
            elif op[0] == "set": new = new + (op[2] - new @ op[1]) @ op[1].T
        if POS["m"] == "last":
            m = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device); m[torch.arange(h.shape[0]), LAST["i"]] = True
            new = torch.where(m[..., None], new, hf)
        h2 = new.to(h.dtype); return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f
# ---- component hooks: per-head attention output (o_proj input) and MLP output ----
CM = {"cap": False, "grad": False, "patch": {}}; CAP = {}
def ah(L):
    def f(mod, inp):
        x = inp[0]
        if ("a", L) in CM["patch"]:
            hs, rep = CM["patch"][("a", L)]; x = _rep(x.view(*x.shape[:2], H, HD).clone(), rep, hs).view(*x.shape[:2], H * HD)
        if CM["cap"]:
            if CM["grad"]: x.retain_grad()
            CAP[("a", L)] = x if CM["grad"] else x.detach()
        return (x,)
    return f
def _rep(x, rep, hs):
    r = rep.view(rep.shape[0], rep.shape[1], H, HD) if rep.dim() == 3 else rep.view(1, 1, H, HD)
    r = r.expand(x.shape[0], x.shape[1], H, HD); x[:, :, hs] = r[:, :, hs].to(x.dtype); return x
def mh(L):
    def f(mod, inp, out):
        x = out
        if ("m", L) in CM["patch"]:
            _, rep = CM["patch"][("m", L)]; x = (rep if rep.dim() == 3 else rep.view(1, 1, D)).expand_as(x).to(x.dtype).clone()
        if CM["cap"]:
            if CM["grad"]: x.retain_grad()
            CAP[("m", L)] = x if CM["grad"] else x.detach()
        return x
    return f
for L in range(1, NL + 1):
    lay = model.model.layers[L - 1]; lay.register_forward_hook(rh(L)); lay.self_attn.o_proj.register_forward_pre_hook(ah(L)); lay.mlp.register_forward_hook(mh(L))

def batch(ids):
    ml = max(len(ENC[i]) for i in ids); x = torch.full((len(ids), ml), tok.pad_token_id); am = torch.zeros_like(x)
    for r, i in enumerate(ids): x[r, :len(ENC[i])] = torch.tensor(ENC[i]); am[r, :len(ENC[i])] = 1
    last = am.sum(1) - 1; LAST["i"] = last.to(dev); return x.to(dev), am.to(dev), last.to(dev)
def marg(logits, ids, last):
    lp = logits[torch.arange(len(ids), device=dev), last].float().log_softmax(-1)
    return torch.stack([lp[r, TA[i]].logsumexp(0) - lp[r, TB[i]].logsumexp(0) for r, i in enumerate(ids)])
BS = 48
@torch.no_grad()
def run(ids=None, bs=BS):
    ids = list(range(N)) if ids is None else ids; out = np.full(N, np.nan)
    srt = sorted(ids, key=lambda i: len(ENC[i]))
    for s in range(0, len(srt), bs):
        b = srt[s:s + bs]; x, am, last = batch(b); out[b] = marg(model(input_ids=x, attention_mask=am).logits, b, last).cpu().numpy()
    return out
def setop(spec, pos="all"):
    HOOK.clear(); POS["m"] = pos
    for L, op in spec: HOOK.setdefault(L, []).append(op)
def acc(m, g): v = m[G[g]]; v = v[~np.isnan(v)]; return round(float((v > 0).mean()), 3)
def show(nm, m): print(nm, {g: acc(m, g) for g in G}, flush=True)

R = LJ("margins.json")
def cond(name, spec, pos="all", ids=None):
    if name in R: return np.array(R[name], dtype=float)
    setop(spec, pos); m = run(ids); setop([]); R[name] = [None if np.isnan(v) else float(v) for v in m]; J(R, "margins.json"); show(name, m); return m

if "base" in PH:
    base = cond("base", [])
    # full-word check (both casings summed, whole answer word) for the baseline
    tot = {}
    for t in ["a", "b"]:
        sc = np.full(N, -np.inf)
        with torch.no_grad():
            for i in range(N):
                for v in {ITEMS[i][t], ITEMS[i][t][0].upper() + ITEMS[i][t][1:]}:
                    w = tok(v, add_special_tokens=False).input_ids; ids = torch.tensor([ENC[i] + w], device=dev)
                    lp = model(input_ids=ids).logits[0, len(ENC[i]) - 1:-1].float().log_softmax(-1).gather(1, torch.tensor(w, device=dev)[:, None]).sum().item()
                    sc[i] = np.logaddexp(sc[i], lp)
        tot[t] = sc
    J(dict(full=(tot["a"] - tot["b"]).tolist()), "base_full.json"); print("full-word acc", {g: round(float(((tot["a"] - tot["b"])[G[g]] > 0).mean()), 3) for g in G}, flush=True)

LAY = list(range(1, NL + 1))
if "states" in PH or "dirs" in PH or "das" in PH or "das2" in PH:
    if os.path.exists(f"{OUT}/states.pt"): ST = torch.load(f"{OUT}/states.pt")
    else:
        ST = torch.zeros(N, NL + 1, D, dtype=torch.float16)
        with torch.no_grad():
            for s in range(0, N, 32):
                b = list(range(s, min(N, s + 32))); x, am, last = batch(b)
                hs = model(input_ids=x, attention_mask=am, output_hidden_states=True).hidden_states
                for L in range(NL + 1): ST[b, L] = hs[L][torch.arange(len(b), device=dev), last].half().cpu()
        torch.save(ST, f"{OUT}/states.pt")
    def mu(g, f, L): idx = [i for i in G[g] if FOLD[i] != f]; return ST[idx, L].float().mean(0)
    DIRS = {}
    for f in [0, 1]:
        for L in LAY:
            bs_, bn, os_, on = (mu(g, f, L) for g in ["body_swap", "body_norm", "obj_swap", "obj_norm"])
            DIRS[(f, L)] = dict(dom=bs_ - os_, swapB=bs_ - bn, swapO=os_ - on, domN=bn - on, swapdiff=(bs_ - bn) - (os_ - on))
    cosd = {L: {k: float(torch.nn.functional.cosine_similarity(DIRS[(0, L)][a], DIRS[(0, L)][b], dim=0)) for k, (a, b) in
                dict(dom_swapB=("dom", "swapB"), dom_swapO=("dom", "swapO"), swapB_swapO=("swapB", "swapO"), dom_domN=("dom", "domN")).items()} for L in LAY}
    J(cosd, "dir_cos.json")
gR = torch.Generator().manual_seed(7)
def unit(v): return (v / v.norm()).to(dev)
BANDS = [(8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 56), (56, 64)]
if "dirs" in PH:
    RND = {(s, L): torch.randn(D, generator=gR) for s in range(3) for L in LAY}
    for f in [0, 1]:
        test = [i for i in range(N) if FOLD[i] == f]
        for a, b in BANDS:
            Lb = list(range(LY(a), LY(b)))
            for k in ["dom", "swapB", "swapO", "swapdiff"]:
                cond(f"f{f}_abl_{k}_{a}-{b}", [(L, ("proj", unit(DIRS[(f, L)][k])[:, None])) for L in Lb], "all", test)
            for s in range(3): cond(f"f{f}_abl_rand{s}_{a}-{b}", [(L, ("proj", unit(RND[(s, L)])[:, None])) for L in Lb], "all", test)
        # framing patch: shift the last token along body-minus-object (swap items' means) at one layer, in units of the full mean gap
        for l in [8, 16, 24, 32, 40, 48, 56]:
            L = LY(l); d = DIRS[(f, L)]["dom"].to(dev); r = unit(RND[(0, L)]) * d.norm()
            for sc in [1, 3]:
                for sg in [1, -1]:
                    cond(f"f{f}_add_dom{sg * sc:+d}_L{l}", [(L, ("add", sg * sc * d))], "last", test)
                    cond(f"f{f}_add_rand{sg * sc:+d}_L{l}", [(L, ("add", sg * sc * r))], "last", test)
        # same shift at every position of a band (the question is read under the other framing)
        for a, b in BANDS[:5]:
            Lb = list(range(LY(a), LY(b)))
            for sg in [1, -1]:
                cond(f"f{f}_addall_dom{sg:+d}_{a}-{b}", [(L, ("add", sg * DIRS[(f, L)]["dom"].to(dev) / len(Lb))) for L in Lb], "all", test)
                cond(f"f{f}_addall_rand{sg:+d}_{a}-{b}", [(L, ("add", sg * unit(RND[(0, L)]) * DIRS[(f, L)]["dom"].norm() / len(Lb))) for L in Lb], "all", test)

if "das" in PH or "das2" in PH:
    # learn a k-dim subspace at layer L (last token) whose value, set to the other side's mean, flips the swap behaviour:
    # object-swap items set to the body-swap mean should answer the displaced part; body-swap items set to the object-swap mean should answer by habit;
    # normal items (set to the other side's normal mean) must stay right. Fit on one half of the scenarios, test on the other. Random subspaces as control.
    DAS = LJ("das.json"); OTHER = {"body_swap": "obj_swap", "obj_swap": "body_swap", "body_norm": "obj_norm", "obj_norm": "body_norm"}
    SIGN = {"body_swap": -1.0, "obj_swap": 1.0, "body_norm": 1.0, "obj_norm": 1.0}
    gof = {i: g for g in G for i in G[g]}
    def das_forward(ids, Q, L, f):
        tgt = {g: (mu(g, f, L).to(dev) @ Q) for g in G}
        x, am, last = batch(ids); T = torch.stack([tgt[OTHER[gof[i]]] for i in ids])
        def hk(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out; hl = h[torch.arange(len(ids), device=dev), last].float()
            new = hl + (T - hl @ Q) @ Q.T; h = h.clone(); h[torch.arange(len(ids), device=dev), last] = new.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = model.model.layers[L - 1].register_forward_hook(hk)
        try: m = marg(model(input_ids=x, attention_mask=am).logits, ids, last)
        finally: hd.remove()
        return m
    for f in ([0, 1] if "das" in PH else []):
        train = [i for i in range(N) if FOLD[i] != f]; test = [i for i in range(N) if FOLD[i] == f]
        for l in [16, 24, 32, 40, 48, 56]:
            for k in [1, 8]:
                key = f"f{f}_L{l}_k{k}"
                if key in DAS: continue
                L = LY(l); torch.manual_seed(f * 100 + l + k)
                W = torch.nn.Parameter(torch.randn(D, k, device=dev) * 0.01); opt = torch.optim.Adam([W], lr=1e-2); rng = np.random.default_rng(l + k + f)
                for step in range(int(os.environ.get("STEPS", 120))):
                    b = list(rng.choice(train, 12, replace=False)); Q = torch.linalg.qr(W)[0]
                    m = das_forward(b, Q, L, f); sg = torch.tensor([SIGN[gof[i]] for i in b], device=dev)
                    loss = torch.nn.functional.softplus(-sg * m).mean(); opt.zero_grad(); loss.backward(); opt.step()
                res = {}
                with torch.no_grad():
                    for nm, Qe in [("learned", torch.linalg.qr(W)[0].detach())] + [(f"rand{s}", torch.linalg.qr(torch.randn(D, k, device=dev))[0]) for s in range(3)]:
                        mm = np.full(N, np.nan)
                        for s in range(0, len(test), 24):
                            b = test[s:s + 24]; mm[b] = das_forward(b, Qe, L, f).cpu().numpy()
                        res[nm] = [None if np.isnan(v) else float(v) for v in mm]
                    Ql = torch.linalg.qr(W)[0].detach().cpu()
                    res["cos"] = {kk: float((Ql.T @ (DIRS[(f, L)][kk] / DIRS[(f, L)][kk].norm())).norm()) for kk in ["dom", "swapB", "swapO", "domN"]}
                DAS[key] = res; J(DAS, "das.json")
                print(key, "loss", round(loss.item(), 3), {nm: {g: acc(np.array(res[nm], dtype=float), g) for g in G} for nm in ["learned", "rand0"]}, res["cos"], flush=True)

if "attr" in PH:
    # attribution patching: clean = swapped item, corrupt = its token-aligned normal twin. attr = (a_corrupt - a_clean) . d margin / d a, summed over positions
    if not os.path.exists(f"{OUT}/attr.pt"):
        A = torch.zeros(N, NL, H + 1)
        for s in range(0, len(SWAP), 8):
            b = SWAP[s:s + 8]; tw = [ITEMS[i]["twin"] for i in b]
            CM["cap"] = True; CM["grad"] = False; CAP.clear()
            with torch.no_grad():
                x, am, last = batch(tw); model(input_ids=x, attention_mask=am)
            corr = {k: v.clone() for k, v in CAP.items()}; CAP.clear(); CM["grad"] = True
            x, am, last = batch(b); emb = model.model.embed_tokens
            hk = emb.register_forward_hook(lambda m_, i_, o_: o_.requires_grad_(True))
            m = marg(model(input_ids=x, attention_mask=am).logits, b, last); m.sum().backward(); hk.remove()
            msk = am[..., None].float()
            for L in range(1, NL + 1):
                ca, cl = corr[("a", L)].float(), CAP[("a", L)]; g = cl.grad.float(); dlt = ((ca - cl.detach().float()) * g * msk).view(len(b), -1, H, HD).sum((1, 3))
                A[b, L - 1, :H] = dlt.cpu()
                cm_, clm = corr[("m", L)].float(), CAP[("m", L)]; A[b, L - 1, H] = ((cm_ - clm.detach().float()) * clm.grad.float() * msk).sum((1, 2)).cpu()
            CM["cap"] = False; CM["grad"] = False; CAP.clear(); model.zero_grad(set_to_none=True)
            print("attr", s, flush=True)
        torch.save(A, f"{OUT}/attr.pt")

if "patch" in PH or "patch2" in PH:
    A = torch.load(f"{OUT}/attr.pt").view(N, -1)  # [N, NL*(H+1)]; comp c -> (layer c//(H+1)+1, head c%(H+1), H == MLP)
    P = LJ("patch.json"); rng = np.random.default_rng(3)
    def comps_spec(cs, src):
        """src: dict comp-> replacement tensor (per-batch acts or global mean)"""
        sp = {}
        for c in cs:
            L, h = c // (H + 1) + 1, c % (H + 1)
            if h == H: sp[("m", L)] = (None, src[("m", L)])
            else:
                hs = sp.get(("a", L), ([], src[("a", L)]))[0] + [h]; sp[("a", L)] = (hs, src[("a", L)])
        return sp
    # mean activations over all items and positions (for mean-ablation)
    MEAN = {}
    with torch.no_grad():
        CM["cap"] = True; cnt = 0
        for s in range(0, N, 32):
            b = list(range(s, min(N, s + 32))); x, am, last = batch(b); CAP.clear(); model(input_ids=x, attention_mask=am); w = am[..., None].float()
            for k, v in CAP.items(): MEAN[k] = MEAN.get(k, 0) + (v.float() * w).sum((0, 1))
            cnt += am.sum().item()
        CM["cap"] = False; CAP.clear(); MEAN = {k: (v / cnt) for k, v in MEAN.items()}
    @torch.no_grad()
    def patch_run(cs, ids, mode):
        out = np.full(N, np.nan)
        for s in range(0, len(ids), 24):
            b = ids[s:s + 24]
            if mode == "twin":
                tw = [ITEMS[i]["twin"] for i in b]; CM["cap"] = True; CAP.clear(); x, am, last = batch(tw); model(input_ids=x, attention_mask=am)
                src = {k: v.clone() for k, v in CAP.items()}; CM["cap"] = False; CAP.clear()
            else: src = MEAN
            CM["patch"] = comps_spec(cs, src); x, am, last = batch(b); out[b] = marg(model(input_ids=x, attention_mask=am).logits, b, last).cpu().numpy(); CM["patch"] = {}
        return [None if np.isnan(v) else float(v) for v in out]
    for f in ([0, 1] if "patch" in PH else []):
        tr = [i for i in SWAP if FOLD[i] != f]; te = [i for i in range(N) if FOLD[i] == f]; te_sw = [i for i in te if i in set(SWAP)]
        bodyA = A[[i for i in tr if ITEMS[i]["fam"] == "body"]].mean(0); objA = A[[i for i in tr if ITEMS[i]["fam"] == "obj"]].mean(0)
        P.setdefault("rank", {})[f"f{f}"] = dict(body=bodyA.argsort()[:200].tolist(), sel=(bodyA - objA).argsort()[:200].tolist(), bodyA=bodyA.tolist(), objA=objA.tolist())
        for k in [5, 20, 80]:
            for rk in ["body", "sel"]:
                cs = P["rank"][f"f{f}"][rk][:k]
                for mode, ids in [("twin", te_sw), ("mean", te)]:
                    key = f"f{f}_{mode}_{rk}_top{k}"
                    if key not in P: P[key] = patch_run(cs, ids, mode); J(P, "patch.json"); show(key, np.array(P[key], dtype=float))
            for s in range(4):
                cs = rng.choice(NL * (H + 1), k, replace=False).tolist()
                for mode, ids in [("twin", te_sw), ("mean", te)]:
                    key = f"f{f}_{mode}_rand{s}_top{k}"
                    if key not in P: P[key] = patch_run(cs, ids, mode); J(P, "patch.json"); show(key, np.array(P[key], dtype=float))
if "patch2" in PH:  # follow-ups: components that oppose the swap (positive attribution) in objects / bodies, and body-only carriers
    for f in [0, 1]:
        te_sw = [i for i in SWAP if FOLD[i] == f]; r = P["rank"][f"f{f}"]; bA, oA = np.array(r["bodyA"]), np.array(r["objA"])
        RK = dict(objsup=np.argsort(-oA), bodysup=np.argsort(-bA), bonly=np.argsort(bA - np.minimum(oA, 0)), oonly=np.argsort(oA - np.minimum(bA, 0)))
        for k in [5, 20]:
            for rk, order in RK.items():
                key = f"f{f}_twin_{rk}_top{k}"
                if key not in P: P[key] = patch_run(order[:k].tolist(), te_sw, "twin"); J(P, "patch.json"); show(key, np.array(P[key], dtype=float))
if "das2" in PH:  # retrain the 1-dim subspace at L40 per fold, keep it, and split it into its body-object-mean part and the rest
    D2 = LJ("das2.json")
    for f in [0, 1]:
        train = [i for i in range(N) if FOLD[i] != f]; test = [i for i in range(N) if FOLD[i] == f]; L = LY(40); torch.manual_seed(f * 100 + 41)
        W = torch.nn.Parameter(torch.randn(D, 1, device=dev) * 0.01); opt = torch.optim.Adam([W], lr=1e-2); rng = np.random.default_rng(41 + f)
        for step in range(120):
            b = list(rng.choice(train, 12, replace=False)); Q = torch.linalg.qr(W)[0]; m = das_forward(b, Q, L, f)
            sg = torch.tensor([SIGN[gof[i]] for i in b], device=dev); loss = torch.nn.functional.softplus(-sg * m).mean(); opt.zero_grad(); loss.backward(); opt.step()
        u = torch.linalg.qr(W)[0].detach()[:, 0]; dm = DIRS[(f, L)]["dom"].to(dev); dh = dm / dm.norm()
        par = (u @ dh) * dh; perp = u - par; perp = perp / perp.norm(); torch.save(u.cpu(), f"{OUT}/das_u_f{f}.pt")
        res = {"cos_u_dom": float(u @ dh)}
        for nm, v in [("full", u), ("perp", perp), ("par", dh), ("rand", unit(torch.randn(D, generator=gR)))]:
            Qe = v[:, None] / v.norm(); mm = np.full(N, np.nan)
            with torch.no_grad():
                for s in range(0, len(test), 24): b = test[s:s + 24]; mm[b] = das_forward(b, Qe, L, f).cpu().numpy()
            res[nm] = [None if np.isnan(x) else float(x) for x in mm]
            # gap between the body and object means along this axis, in units of the item spread
            pb = ST[[i for i in G["body_swap"] if FOLD[i] != f], L].float() @ (Qe[:, 0].cpu()); po = ST[[i for i in G["obj_swap"] if FOLD[i] != f], L].float() @ (Qe[:, 0].cpu())
            res[nm + "_dprime"] = float((pb.mean() - po.mean()) / torch.cat([pb - pb.mean(), po - po.mean()]).std())
            print(f, nm, {g: acc(mm, g) for g in G}, round(res[nm + "_dprime"], 2), flush=True)
        D2[f"f{f}"] = res; J(D2, "das2.json")
print("DONE", PH, flush=True)
