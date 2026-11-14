"""Attribution over text (12 Nov). python attr_run.py OUT MODEL [PHASES=atp,patch]
For twin pairs (items_attr.json) measures where in the text and depth the touch-derived body map (40-dim, as in run32) is used:
(1) AtP: grad(LD) . P_S (h_twin - h_clean) per (layer, token), S = body map PU / object map OU / 10 random 40-dim / full residual;
    also grad x act projected (grad . P_S h_clean) and energy share |P_S dh|^2/|dh|^2.
(2) Patch: replace only the S-component of the residual with the twin's, at position set x layer band; LD moved as a fraction of the twin gap."""
import json, os, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT, M = sys.argv[1], sys.argv[2]; PH = (sys.argv[3] if len(sys.argv) > 3 else "atp,patch").split(","); os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
for p in model.parameters(): p.requires_grad_(False)
NL = model.config.num_hidden_layers; D = model.config.hidden_size; K = 40; NR = 10
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------- hooks ----------
ST = {"mode": None}  # mode: None | "grad" | "patch"
CACHE = {}
def mk(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        m = ST["mode"]
        if m == "cap": CACHE[L] = h.detach()
        elif m == "grad":
            h.retain_grad(); CACHE[L] = h
        elif m == "patch" and L in ST["layers"]:
            hf = h.float(); d = ST["src"][L].to(hf.device).float() - hf
            U = ST["U"]
            if U is not None: d = (d @ U[L]) @ U[L].T
            h2 = (hf + ST["mask"][..., None] * d).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return out
    return f
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk(L))
model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_() if ST["mode"] == "grad" else o)

@torch.no_grad()
def states_last(texts, bs=32):
    res = {L: [] for L in range(1, NL + 1)}
    for i in range(0, len(texts), bs):
        enc = tok([chat(t) for t in texts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states; last = enc.attention_mask.sum(1) - 1
        for L in res: res[L].append(hs[L][torch.arange(len(last)), last].float().cpu())
    return {L: torch.cat(v) for L, v in res.items()}

# ---------- subspaces (same construction as run32: top-40 of 60 touched-place class means) ----------
FR = ["A feather brushes slowly across {L}.", "Someone presses a thumb firmly into {L}.", "An ice cube touches {L}, cold and wet, then slides away.",
      "The doctor presses two fingers on {L} and asks if it hurts.", "A mosquito bites {L}.", "There is a single light tap on {L}, then nothing.",
      "A heavy weight presses down on {L}.", "A small beetle crawls over {L}."]
OWN = ["your", "the woman's", "the man's"]
PP30 = ["hand", "foot", "knee", "shoulder", "elbow", "wrist", "cheek", "ear", "thigh", "hip", "ankle", "palm", "forearm", "shin", "calf", "heel", "toe",
        "thumb", "index finger", "upper arm", "eyebrow", "temple", "jaw", "neck", "collarbone", "rib", "buttock", "big toe", "knuckle", "armpit"]
OB30 = ["cup", "chair", "lamp", "book", "bottle", "table", "clock", "spoon", "bucket", "window", "bicycle", "pillow", "kettle", "plate", "shoe", "glove",
        "drawer", "door", "mirror", "basket", "box", "jar", "fork", "hammer", "brush", "candle", "phone", "vase", "towel", "bowl"]
WORD = os.environ.get("WORD") == "1"  # map read at the body-part token instead of the end of the prompt
@torch.no_grad()
def states_word(texts, phrases, bs=32):
    res = {L: [] for L in range(1, NL + 1)}
    for i in range(0, len(texts), bs):
        ch = [chat(t) for t in texts[i:i + bs]]; enc = tok(ch, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
        off = enc.pop("offset_mapping"); enc = enc.to(dev); pos = []
        for r, (c, ph) in enumerate(zip(ch, phrases[i:i + bs])):
            end = c.index(ph) + len(ph) - 1; pos.append(next(j for j, (a, b) in enumerate(off[r].tolist()) if a <= end < b))
        hs = model(**enc, output_hidden_states=True).hidden_states
        for L in res: res[L].append(hs[L][torch.arange(len(pos)), torch.tensor(pos)].float().cpu())
    return {L: torch.cat(v) for L, v in res.items()}
def basis(names):
    phr = [[f"{o} {s} {p}" for o in OWN] for p in names for s in ["left", "right"]]
    txt = [f.format(L=x) for p in phr for x in p for f in FR]; xs = [x for p in phr for x in p for f in FR]
    S = states_word(txt, xs) if WORD else states_last(txt); n = len(OWN) * len(FR); out = {}
    for L in S:
        Mn = S[L].view(len(phr), n, D).mean(1); Mn = Mn - Mn.mean(0); out[L] = torch.linalg.svd(Mn, full_matrices=False)[2][:K].T.contiguous()
    return out
if os.path.exists(f"{OUT}/subs.pt"): SUB = torch.load(f"{OUT}/subs.pt")
else:
    g = torch.Generator().manual_seed(7)
    SUB = dict(PU=basis(PP30), OU=basis(OB30))
    for r in range(NR): SUB[f"R{r}"] = {L: torch.linalg.qr(torch.randn(D, K, generator=g))[0] for L in range(1, NL + 1)}
    torch.save(SUB, f"{OUT}/subs.pt")
SUBD = {k: {L: v[L].to(dev) for L in v} for k, v in SUB.items()}
SN = ["PU", "OU"] + [f"R{r}" for r in range(NR)]
UALL = {L: torch.cat([SUBD[s][L] for s in SN], 1) for L in range(1, NL + 1)}  # D x (12K)
log("subspaces ready; cos-overlap PU/OU at mid layer", float(torch.linalg.svdvals(SUBD["PU"][NL // 2].T @ SUBD["OU"][NL // 2]).pow(2).mean()))

# ---------- items ----------
IT = json.load(open(os.environ.get("ITEMS", "items_attr.json")))
def first_ids(w):
    return sorted({tok(v, add_special_tokens=False).input_ids[0] for v in [w, w.capitalize(), " " + w, " " + w.capitalize()]})
keep = []
for it in IT:
    c = tok(chat(it["clean"]), add_special_tokens=False).input_ids; t = tok(chat(it["twin"]), add_special_tokens=False).input_ids
    A, B = first_ids(it["a"]), first_ids(it["b"])
    if len(c) != len(t) or set(A) & set(B): continue
    diff = [i for i in range(len(c)) if c[i] != t[i]]
    if not diff: continue
    it.update(cids=c, tids=t, A=A, B=B, diff=diff); keep.append(it)
log(f"kept {len(keep)}/{len(IT)} aligned items")
IT = keep

def batch(items, key):
    T = max(len(it[key]) for it in items); ids = torch.full((len(items), T), tok.pad_token_id); am = torch.zeros_like(ids)
    for r, it in enumerate(items): ids[r, :len(it[key])] = torch.tensor(it[key]); am[r, :len(it[key])] = 1
    return ids.to(dev), am.to(dev)
def LD(logits, items):
    lp = logits.float().log_softmax(-1); out = []
    for r, it in enumerate(items):
        x = lp[r, len(it["cids"]) - 1]; out.append(torch.logsumexp(x[it["A"]], 0) - torch.logsumexp(x[it["B"]], 0))
    return torch.stack(out)
@torch.no_grad()
def twin_states(items):
    ids, am = batch(items, "tids"); CACHE.clear(); ST["mode"] = "cap"; o = model(input_ids=ids, attention_mask=am); ST["mode"] = None
    H = dict(CACHE); CACHE.clear(); return H, LD(o.logits, items)

BS = 8 if NL <= 36 else 4
# ---------- (1) AtP ----------
if "atp" in PH:
    res = []
    for i in range(0, len(IT), BS):
        items = IT[i:i + BS]; Ht, ldt = twin_states(items)
        ids, am = batch(items, "cids"); ST["mode"] = "grad"; CACHE.clear()
        with torch.enable_grad():
            o = model(input_ids=ids, attention_mask=am); ld = LD(o.logits, items); ld.sum().backward()
        ST["mode"] = None
        T = ids.shape[1]
        full = torch.zeros(len(items), NL, T); sub = torch.zeros(len(items), len(SN), NL, T); gxa = torch.zeros_like(sub); en = torch.zeros_like(sub); gxa_full = torch.zeros_like(full)
        with torch.no_grad():
            for L in range(1, NL + 1):
                g = CACHE[L].grad.float(); hc = CACHE[L].detach().float(); dh = Ht[L].float() - hc
                full[:, L - 1] = (g * dh).sum(-1).cpu(); gxa_full[:, L - 1] = (g * hc).sum(-1).cpu()
                gU = (g @ UALL[L]).view(*g.shape[:2], len(SN), K); dU = (dh @ UALL[L]).view_as(gU); hU = (hc @ UALL[L]).view_as(gU)
                sub[:, :, L - 1] = (gU * dU).sum(-1).permute(0, 2, 1).cpu(); gxa[:, :, L - 1] = (gU * hU).sum(-1).permute(0, 2, 1).cpu()
                en[:, :, L - 1] = (dU.pow(2).sum(-1) / dh.pow(2).sum(-1).clamp_min(1e-6)[..., None]).permute(0, 2, 1).cpu()
        CACHE.clear()
        for r, it in enumerate(items):
            n = len(it["cids"])
            res.append(dict(id=it["id"], ld=float(ld[r]), ldt=float(ldt[r]), full=full[r, :, :n].numpy().astype(np.float16), sub=sub[r, :, :, :n].numpy().astype(np.float16),
                            gxa=gxa[r, :, :, :n].numpy().astype(np.float16), gxa_full=gxa_full[r, :, :n].numpy().astype(np.float16), en=en[r, :, :, :n].numpy().astype(np.float16)))
        if i % (BS * 10) == 0: log("atp", i, len(IT))
    torch.save(dict(res=res, SN=SN, items=[{k: v for k, v in it.items()} for it in IT]), f"{OUT}/atp.pt"); log("atp done")

# ---------- (2) patching: position set x layer band x subspace ----------
def Lb(a, b): return set(range(max(1, round(a * NL) + 1), round(b * NL) + 1))
BANDS = {"all": set(range(1, NL + 1))}
BANDS["q1"] = Lb(0, .25); BANDS["q2"] = Lb(.25, .5); BANDS["q3"] = Lb(.5, .75); BANDS["q4"] = Lb(.75, 1)
POS = ["swap", "post", "last", "all"]
def posmask(items, T, kind):
    m = torch.zeros(len(items), T, device=dev)
    for r, it in enumerate(items):
        n = len(it["cids"]); d0, d1 = it["diff"][0], it["diff"][-1]
        if kind == "swap": m[r, it["diff"]] = 1
        elif kind == "post": m[r, d1 + 1:n - 1] = 1
        elif kind == "last": m[r, n - 1] = 1
        elif kind == "all": m[r, d0:n] = 1
    return m
if "patch" in PH:
    out = {}
    for i in range(0, len(IT), BS * 2):
        items = IT[i:i + BS * 2]; Ht, ldt = twin_states(items); ids, am = batch(items, "cids")
        with torch.no_grad():
            ld0 = LD(model(input_ids=ids, attention_mask=am).logits, items)
            for pk in POS:
                ST["mask"] = posmask(items, ids.shape[1], pk)
                for bk, bl in BANDS.items():
                    ST["layers"] = bl; ST["src"] = Ht
                    for sk in ["full"] + SN:
                        ST["U"] = None if sk == "full" else SUBD[sk]; ST["mode"] = "patch"
                        ldp = LD(model(input_ids=ids, attention_mask=am).logits, items); ST["mode"] = None
                        for r, it in enumerate(items): out.setdefault(it["id"], {})[f"{pk}|{bk}|{sk}"] = float(ldp[r])
        for r, it in enumerate(items): out[it["id"]].update(ld0=float(ld0[r]), ldt=float(ldt[r]))
        log("patch", i, len(IT)); json.dump(out, open(f"{OUT}/patch.json", "w"))
    log("patch done")

# ---------- (3) grid: patch one token x one band of NL/8 layers, for selected item ranges (env GRID="12-23,112-127") ----------
if "grid" in PH:
    want = set()
    for rg in os.environ.get("GRID", "12-23,112-127").split(","):
        a, b = map(int, rg.split("-")); want |= set(range(a, b + 1))
    GI = [it for it in IT if it["id"] in want]; NB = 8; GB = {j: Lb(j / NB, (j + 1) / NB) for j in range(NB)}; GS = ["full", "PU", "OU", "R0", "R1"]
    out = {}
    for i in range(0, len(GI), BS * 2):
        items = GI[i:i + BS * 2]; Ht, ldt = twin_states(items); ids, am = batch(items, "cids"); T = ids.shape[1]
        with torch.no_grad():
            ld0 = LD(model(input_ids=ids, attention_mask=am).logits, items)
            span = max(len(it["cids"]) - it["diff"][0] for it in items)
            for k in range(span):
                m = torch.zeros(len(items), T, device=dev)
                for r, it in enumerate(items):
                    p = it["diff"][0] + k
                    if p < len(it["cids"]): m[r, p] = 1
                ST["mask"] = m; ST["src"] = Ht
                for j, bl in GB.items():
                    ST["layers"] = bl
                    for sk in GS:
                        ST["U"] = None if sk == "full" else SUBD[sk]; ST["mode"] = "patch"
                        ldp = LD(model(input_ids=ids, attention_mask=am).logits, items); ST["mode"] = None
                        for r, it in enumerate(items):
                            if it["diff"][0] + k < len(it["cids"]): out.setdefault(it["id"], {})[f"{it['diff'][0] + k}|{j}|{sk}"] = float(ldp[r])
        for r, it in enumerate(items): out[it["id"]].update(ld0=float(ld0[r]), ldt=float(ldt[r]))
        log("grid", i, len(GI)); json.dump(out, open(f"{OUT}/grid.json", "w"))
    log("grid done")
