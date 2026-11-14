
# ---------- run 6: hand/foot direction read from the task stems themselves, pushed at stem tokens ----------
def setop(spec):
    HOOK.clear()
    for L, op in spec: HOOK.setdefault(L, []).append(op)
@torch.no_grad()
def stem_states(ks, bs=32 if NL == 36 else 8):
    res = {L: [] for L in LAY}; MASK["m"] = None
    for i in range(0, len(ks), bs):
        ch = ks[i:i + bs]; enc = tok([chat(ITEMS[k]["q"] + suffix(ITEMS[k])) for k in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states; pos = torch.tensor([SPAN[k][1] - 1 for k in ch], device=dev)
        for L in LAY: res[L].append(hs[L][torch.arange(len(ch), device=dev), pos].float().cpu())
    return {L: torch.cat(v) for L, v in res.items()}
K = {t: [k for k, it in enumerate(ITEMS) if it["task"] == t] for t in set(it["task"] for it in ITEMS)}
EFF = K["effector"]; A, B = EFF[0::2], EFF[1::2]   # fit on A, test on B (and the reverse)
def hfdir(ks):
    S = stem_states(ks); y = torch.tensor([ITEMS[k]["hf"] == "hand" for k in ks]); out = {}
    for L in LAY:
        m = S[L][y].mean(0) - S[L][~y].mean(0); out[L] = dict(u=m / m.norm(), gap=float(m.norm()))
    return out
DD = {"effA": hfdir(A), "effB": hfdir(B), "pain": hfdir(K["pain"])}
TD = torch.load(f"{OUT}/dirs.pt") if os.path.exists(f"{OUT}/dirs.pt") else None
DD["touch"] = {L: dict(u=DIR["hf"][L]["u"], gap=DD["effA"][L]["gap"]) for L in LAY}
gR = torch.Generator().manual_seed(6); DD["rand"] = {L: dict(u=(lambda r: r / r.norm())(torch.randn(D, generator=gR)), gap=DD["effA"][L]["gap"]) for L in LAY}
json.dump({L: dict(effA_touch=float(DD["effA"][L]["u"] @ DIR["hf"][L]["u"]), effA_effB=float(DD["effA"][L]["u"] @ DD["effB"][L]["u"]), effA_pain=float(DD["effA"][L]["u"] @ DD["pain"][L]["u"]),
                   gap=DD["effA"][L]["gap"], touchgap=DIR["hf"][L]["gap"]) for L in LAY}, open(f"{OUT}/cos6.json", "w"), indent=0)
print("cos effA/touch", [round(float(DD["effA"][L]["u"] @ DIR["hf"][L]["u"]), 2) for L in Ls(4, 32)[::3]], flush=True)
# push every item toward foot (sign -1) or hand (+1); the per-item sign is applied in score via a separate pass per sign
POS = {"last": lambda k: (SPAN[k][1] - 1, SPAN[k][1]), "stem": lambda k: SPAN[k]}
def run_push(dname, Lb, a, posname, sign):
    global STEM
    sp = [POS[posname](s[0]) for s in SEQ]; old = STEM; STEM = sp
    setop([(L, ("add", (sign * a * DD[dname][L]["gap"] * DD[dname][L]["u"]).to(dev))) for L in Lb]); S = score(); setop([]); STEM = old
    return S
R = {"base": score().tolist()}
for posname in ["last", "stem"]:
    for bname, (a0, b0) in [("mid", (10, 22)), ("late", (18, 28))]:
        for dname in ["effA", "pain", "touch", "rand"]:
            for a in [2, 4]:
                for sign, sn in [(1, "tohand"), (-1, "tofoot")]:
                    R[f"{dname}_{posname}_{bname}_{a}_{sn}"] = run_push(dname, Ls(a0, b0), a, posname, sign).tolist()
                json.dump(R, open(f"{OUT}/scores6.json", "w")); print(dname, posname, bname, a, flush=True)
# a few greedy replies for the figure
G = {}
ks = EFF + K["pain"] + K["hf_word"] + K["geo"][:8]
for name, (dn, a, sg) in {"base": ("rand", 0, 1), "effA_stem_mid_4_tofoot": ("effA", 4, -1), "effA_stem_mid_4_tohand": ("effA", 4, 1), "rand_stem_mid_4_tofoot": ("rand", 4, -1)}.items():
    setop([(L, ("add", (sg * a * DD[dn][L]["gap"] * DD[dn][L]["u"]).to(dev))) for L in Ls(10, 22)]); G[name] = gen(ks); setop([])
json.dump(G, open(f"{OUT}/gens6.json", "w")); print("ALL DONE", flush=True)
