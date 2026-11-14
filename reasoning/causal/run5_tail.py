
# ---------- stem conditions ----------
def setop(spec):
    HOOK.clear()
    for L, op in spec: HOOK.setdefault(L, []).append(op)
SI = states([it["q"] + suffix(it) for it in ITEMS], LAY)
json.dump({k: {L: ((SI[L] @ DIR[k][L]["u"]) - DIR[k][L]["mid"]).div(DIR[k][L]["gap"]).tolist() for L in LAY} for k in DIR}, open(f"{OUT}/proj.json", "w"))
gR = torch.Generator().manual_seed(5)
RU = {L: (lambda r: r / r.norm())(torch.randn(D, generator=gR)) for L in LAY}
DIR["rand"] = {L: dict(u=RU[L], gap=DIR["hf"][L]["gap"], mid=float(SI[L].mean(0) @ RU[L])) for L in LAY}
# body-part subspace (12 touched parts) and object subspace (12 touched objects), 11 dims each, for ablation
PS = {p: states([f.format(L=f"{o} {s} {p}") for o in OWN for s in ["left", "right"] for f in FR], LAY) for p in PP}
OBJN = ["cup", "chair", "lamp", "book", "bottle", "table", "clock", "spoon", "bucket", "window", "bicycle", "pillow"]
OS = {o: states([f.replace("{L}", "the " + o) for f in FR], LAY) for o in OBJN}
def basis(S):
    out = {}
    for L in LAY:
        Mn = torch.stack([v[L].mean(0) for v in S.values()]); Mn = Mn - Mn.mean(0); out[L] = torch.linalg.svd(Mn, full_matrices=False)[2][:11].T.contiguous()
    return out
PU, OU = basis(PS), basis(OS); RQ = {L: torch.linalg.qr(torch.randn(D, 11, generator=gR))[0] for L in LAY}
def add(n, Lb, a): return [(L, ("add", (a * DIR[n][L]["gap"] * DIR[n][L]["u"]).to(dev))) for L in Lb]
def refl(n, Lb): return [(L, ("refl", DIR[n][L]["u"].to(dev), DIR[n][L]["mid"])) for L in Lb]
def proj(U, Lb): return [(L, ("proj", U[L].to(dev))) for L in Lb]
C = {"base": []}
for bname, (a0, b0) in [("early", (2, 14)), ("mid", (10, 22)), ("all", (2, 30))]:
    Lb = Ls(a0, b0)
    for n in ["hf", "lr_body", "lr_obj", "rand"]:
        C[f"refl_{n}_{bname}"] = refl(n, Lb)
        for a in [-2, -1, 1, 2]: C[f"add_{n}_{bname}_{a:+d}"] = add(n, Lb, a)
    C[f"abl_hf_{bname}"] = proj({L: DIR["hf"][L]["u"][:, None] for L in LAY}, Lb)
    C[f"abl_parts_{bname}"] = proj(PU, Lb); C[f"abl_objs_{bname}"] = proj(OU, Lb); C[f"abl_rand11_{bname}"] = proj(RQ, Lb)
print(len(C), "conditions", flush=True)
R = {}
for name, spec in C.items():
    setop(spec); R[name] = score().tolist(); setop([])
    json.dump(R, open(f"{OUT}/scores.json", "w")); print(name, flush=True)
G = {}
for name in ["base", "refl_hf_mid", "refl_hf_all", "refl_lr_body_mid", "refl_lr_body_all", "refl_rand_all", "add_hf_mid_-2", "add_hf_mid_+2", "abl_parts_all"]:
    setop(C[name]); G[name] = gen(list(range(len(ITEMS)))); setop([])
    json.dump(G, open(f"{OUT}/gens.json", "w"))
print("ALL DONE", flush=True)
