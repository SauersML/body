
# ---------- run 32: base accuracy, answer-position pushes, subspace removal, stem swaps; one script, MODE picks the edited tokens ----------
def setop(spec):
    HOOK.clear()
    for L, op in spec: HOOK.setdefault(L, []).append(op)
SI = states([it["q"] + suffix(it) for it in ITEMS], LAY)
gR = torch.Generator().manual_seed(5)
RU = {L: (lambda r: r / r.norm())(torch.randn(D, generator=gR)) for L in LAY}
RU2 = {L: (lambda r: r / r.norm())(torch.randn(D, generator=gR)) for L in LAY}
DIR["rand"] = {L: dict(u=RU[L], gap=DIR["hf"][L]["gap"], mid=float(SI[L].mean(0) @ RU[L])) for L in LAY}
DIR["randB"] = {L: dict(u=RU2[L], gap=DIR["hf"][L]["gap"], mid=float(SI[L].mean(0) @ RU2[L])) for L in LAY}
DIR["rand_lr"] = {L: dict(u=RU[L], gap=DIR["lr_body"][L]["gap"], mid=float(SI[L].mean(0) @ RU[L])) for L in LAY}
# body-place subspace: 30 parts x 2 sides = 60 touched places; object subspace: 30 objects x 2 sides, same wordings/owners. Top-k of the class means.
PP30 = ["hand", "foot", "knee", "shoulder", "elbow", "wrist", "cheek", "ear", "thigh", "hip", "ankle", "palm", "forearm", "shin", "calf", "heel", "toe",
        "thumb", "index finger", "upper arm", "eyebrow", "temple", "jaw", "neck", "collarbone", "rib", "buttock", "big toe", "knuckle", "armpit"]
OB30 = ["cup", "chair", "lamp", "book", "bottle", "table", "clock", "spoon", "bucket", "window", "bicycle", "pillow", "kettle", "plate", "shoe", "glove",
        "drawer", "door", "mirror", "basket", "box", "jar", "fork", "hammer", "brush", "candle", "phone", "vase", "towel", "bowl"]
def basis(phr, k):
    means = []
    for p in phr:
        S = states([f.format(L=x) for x in p for f in FR], LAY); means.append(S)
    out = {}
    for L in LAY:
        Mn = torch.stack([S[L].mean(0) for S in means]); Mn = Mn - Mn.mean(0); out[L] = torch.linalg.svd(Mn, full_matrices=False)[2][:k].T.contiguous()
    return out
PU = basis([[f"{o} {s} {p}" for o in OWN] for p in PP30 for s in ["left", "right"]], 40)
OU = basis([[f"{o} {s} {p}" for o in OWN] for p in OB30 for s in ["left", "right"]], 40)
RQ = {L: torch.linalg.qr(torch.randn(D, 40, generator=gR))[0] for L in LAY}
torch.save(dict(PU=PU, OU=OU), f"{OUT}/subs.pt")
def add(n, Lb, a, base=None): return [(L, ("add", (a * DIR[n][L]["gap"] * DIR[n][L]["u"]).to(dev))) for L in Lb]
def refl(n, Lb): return [(L, ("refl", DIR[n][L]["u"].to(dev), DIR[n][L]["mid"])) for L in Lb]
def proj(U, Lb, k): return [(L, ("proj", U[L][:, :k].contiguous().to(dev))) for L in Lb]
C = {"base": ("ans", [])}
late, allb, stemb, midb = Ls(22, 30), Ls(6, 34), Ls(2, 30), Ls(10, 22)
for a in [-4, -3, 3, 4]:
    for n in ["hf", "rand", "randB"]: C[f"ans_{n}_{a:+d}"] = ("ans", add(n, late, a))
for a in [-3, 3]:
    for n in ["lr_body", "lr_obj", "rand_lr"]: C[f"ans_{n}_{a:+d}"] = ("ans", add(n, late, a))
for k in [11, 40]:
    for nm, U in [("parts", PU), ("objs", OU), ("rand", RQ)]: C[f"all_abl_{nm}{k}"] = ("all", proj(U, allb, k))
for nm, U in [("parts", PU), ("objs", OU), ("rand", RQ)]: C[f"ans_abl_{nm}40"] = ("ans", proj(U, late, 40)); C[f"stem_abl_{nm}40"] = ("stem", proj(U, stemb, 40))
for bn, Lb in [("all", stemb), ("mid", midb)]:
    for n in ["lr_body", "lr_obj", "rand_lr", "hf", "rand"]: C[f"stem_refl_{n}_{bn}"] = ("stem", refl(n, Lb))
print(len(C), "conditions", flush=True)
R = json.load(open(f"{OUT}/scores.json")) if os.path.exists(f"{OUT}/scores.json") else {}
for name, (mode, spec) in C.items():
    if name in R: continue
    MODE["m"] = mode; setop(spec); R[name] = score().tolist(); setop([])
    json.dump(R, open(f"{OUT}/scores.json", "w")); print(name, flush=True)
MODE["m"] = "ans"; G = {}
for name in ["base", "ans_hf_+4", "ans_hf_-4"]:
    setop(C[name][1]); G[name] = gen(list(range(len(ITEMS)))); setop([])
    json.dump(G, open(f"{OUT}/gens.json", "w"))
print("ALL DONE", flush=True)
