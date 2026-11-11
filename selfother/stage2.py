from common import *
CFG = json.load(open(sys.argv[1])); OUT = CFG["out"]; os.makedirs(OUT, exist_ok=True)
Z = np.load("dirs.npz"); R = list(range(1, NL + 1)); MS = dict(zip(R, Z["ms"])); MO = dict(zip(R, Z["mo"]))
DV = {L: torch.tensor(Z[f"d{L}"], device=dev) for L in R}; U = {L: DV[L] / DV[L].norm() for L in R}
LV = {L: torch.tensor(Z[f"lr{L}"], device=dev) for L in R}; UL = {L: LV[L] / LV[L].norm() for L in R}
g = torch.Generator().manual_seed(0); UR = {L: (lambda v: v / v.norm())(torch.randn(D, generator=g)).to(dev) for L in R}
from stim import ROWS, LR
HS = states([r["text"] for r in ROWS], R); HLs = states([r["text"] for r in LR], R)
def mid(u, X): return float((torch.tensor(X, device=dev) @ u).mean())
B = range(CFG["band"][0], CFG["band"][1] + 1); Ls = CFG["L"]; a = CFG["alpha"]
side = np.array([r["s"] for r in LR])
CONDS = {"baseline": [],
 "so_ablate": [(L, ablate(U[L])) for L in B], "rand_ablate": [(L, ablate(UR[L])) for L in B], "lr_ablate": [(L, ablate(UL[L])) for L in B],
 "so_toother": [(Ls, addv(-a * DV[Ls]))], "so_toself": [(Ls, addv(a * DV[Ls]))],
 "rand_push": [(Ls, addv(a * DV[Ls].norm() * UR[Ls]))], "lr_push": [(Ls, addv(a * DV[Ls].norm() * UL[Ls]))],
 "so_reflect": [(Ls, reflect(U[Ls], (MS[Ls] + MO[Ls]) / 2))], "lr_reflect": [(Ls, reflect(UL[Ls], mid(UL[Ls], HLs[Ls - 1])))],
 "rand_reflect": [(Ls, reflect(UR[Ls], mid(UR[Ls], HS[Ls - 1])))]}
if CFG.get("only"): CONDS = {k: v for k, v in CONDS.items() if k in CFG["only"]}
AH = np.load("attr.npz")["AH"]; HM = torch.load("head_means.pt").to(dev); order = np.dstack(np.unravel_index(np.argsort(-AH.ravel()), AH.shape))[0]
K = CFG.get("heads_k", 0); rng = np.random.default_rng(1)
HEADSETS = {} if not K else {"heads_top": [(int(l), int(h)) for l, h in order[:K]], "heads_rand": [(int(x // 32), int(x % 32)) for x in rng.choice(NL * 32, K, replace=False)]}
QS = CFG["qs"]; N = CFG.get("n", 16)
G = json.load(open(f"{OUT}/gens.json")) if os.path.exists(f"{OUT}/gens.json") else {}; P = {}
def run(name):
    t1 = time.time(); P[name] = ppl(); out = gen(list(QS.values()), n=N); G[name] = {q: out[i * N:(i + 1) * N] for i, q in enumerate(QS)}
    # forced-choice check under the edit is cheap: energy of edit
    print(name, "ppl", round(P[name], 2), round(time.time() - t1), "s", flush=True)
    for q in QS: print("   ", q, "|", G[name][q][0][:160].replace("\n", " "), flush=True)
    json.dump(G, open(f"{OUT}/gens.json", "w"), indent=1); json.dump(P, open(f"{OUT}/ppl.json", "w"), indent=1)
for name, ed in CONDS.items():
    HOOK.clear(); [HOOK.setdefault(L, []).append(fn) for L, fn in ed]; run(name)
HOOK.clear()
for name, lst in HEADSETS.items():
    HEADS.clear()
    for li in set(l for l, h in lst): HEADS[li] = (torch.tensor([h for l, h in lst if l == li], device=dev), HM[li])
    run(name)
HEADS.clear()
# judge: unedited model, fresh chat, yes/no
J = {}; JQ = CFG["judge"]
def judge(prompts):
    outs = []; tok.padding_side = "left"
    for b in range(0, len(prompts), 64):
        enc = tok([chat([{"role": "user", "content": p}]) for p in prompts[b:b + 64]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad(): o_ = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [x.strip().lower().startswith("yes") for x in tok.batch_decode(o_[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return outs
for name in G:
    J[name] = {}
    for q in QS:
        for jn, jt in JQ.get(q, {}).items():
            J[name][f"{q}:{jn}"] = judge([f"Here is a reply to the message: \"{QS[q]}\"\n\nReply: \"{r.strip()[:900]}\"\n\n{jt} Answer only yes or no." for r in G[name][q]])
        J[name][f"{q}:coherent"] = judge([f"Here is a reply to the message: \"{QS[q]}\"\n\nReply: \"{r.strip()[:900]}\"\n\nIs the reply fluent, coherent English? Answer only yes or no." for r in G[name][q]])
json.dump(J, open(f"{OUT}/judge.json", "w"), indent=1)
print({n: {k: round(float(np.mean(v)), 2) for k, v in J[n].items()} for n in J}, flush=True); print("done", flush=True)
