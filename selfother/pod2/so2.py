# Does adding "me" at L24 move the internal state toward the model's own toe/foot place?
# Outputs (out2/): maps.npz (place-map states, plain chat and embodiment-prefix chat), gens.json, st.npz (states per prompt x cond)
from common import *
sys.path.insert(0, "somato"); import stimuli as S
OUT = "out2"; os.makedirs(OUT, exist_ok=True)
Z = np.load("dirs.npz"); R = list(range(1, NL + 1))
DV = {L: torch.tensor(Z[f"d{L}"], device=dev) for L in R}
LV = {L: torch.tensor(Z[f"lr{L}"], device=dev) for L in R}; UL = {L: LV[L] / LV[L].norm() for L in R}
g = torch.Generator().manual_seed(0); UR = {L: (lambda v: v / v.norm())(torch.randn(D, generator=g)).to(dev) for L in R}
READ = list(range(20, 33)); Ls = 24; a = 2.0
# ---- 1. place maps: touch on YOUR body (432 = 54 places x 8 wordings), plus the woman's, at reply start
rows = [r for r in S.build() if r["cond"] in ("self", "other")]
pre = json.load(open("somato/prefix.json"))
def st_msgs(msgss, bs=32):
    out = np.zeros((len(msgss), len(READ), D), np.float16); tok.padding_side = "left"
    with torch.no_grad():
        for b in range(0, len(msgss), bs):
            enc = tok([chat(m) for m in msgss[b:b + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            hs = model.model(**enc, output_hidden_states=True).hidden_states
            out[b:b + bs] = torch.stack([hs[L][:, -1] for L in READ], 1).float().cpu().numpy()
    return out
t0 = time.time()
Mp = st_msgs([[{"role": "user", "content": r["text"]}] for r in rows])
Me = st_msgs([pre + [{"role": "user", "content": r["text"]}] for r in rows], bs=16)
np.savez(f"{OUT}/maps.npz", plain=Mp, prefix=Me, loc=np.array([r["loc"] for r in rows]), cond=np.array([r["cond"] for r in rows]), read=np.array(READ))
json.dump(rows, open(f"{OUT}/maprows.json", "w")); print("maps", round(time.time() - t0), flush=True)
# ---- 2. prompts x conditions
QS = {"woman_toe": "The woman next to you just stubbed her toe hard on the table leg. How does it feel, and what do you do?",
      "man_toe": "The man next to you just stubbed his toe hard on the table leg. How does it feel, and what do you do?",
      "friend_burn": "Your friend just touched the hot stove and burned her fingers. How does it feel?",
      "self_toe": "You just stubbed your toe hard on the table leg. How does it feel?",
      "self_burn": "You just touched the hot stove and burned your fingers. How does it feel?"}
CONDS = {"normal": None, "me": a * DV[Ls], "random": a * DV[Ls].norm() * UR[Ls], "leftright": a * DV[Ls].norm() * UL[Ls]}
N = 16; G = {}; ST = {}
for cn, v in CONDS.items():
    HOOK.clear()
    if v is not None: HOOK[Ls] = [addv(v)]
    reps = gen(list(QS.values()), n=N, maxn=120); G[cn] = {q: reps[i * N:(i + 1) * N] for i, q in enumerate(QS)}
    for q, p in QS.items():
        # decision point = last prompt token (reply start)
        ST[f"{cn}|{q}|dp"] = st_msgs([[{"role": "user", "content": p}]])[0]
        # reply tokens: teacher-force each reply under the same edit; mean over reply tokens (all, and first 12)
        allm = np.zeros((N, len(READ), D), np.float32); f12 = np.zeros_like(allm)
        with torch.no_grad():
            for i, r in enumerate(G[cn][q]):
                pr = chat([{"role": "user", "content": p}]); ids_p = tok(pr, add_special_tokens=False).input_ids
                ids_r = tok(r, add_special_tokens=False).input_ids[:120]
                ids = torch.tensor([ids_p + ids_r], device=dev); hs = model.model(input_ids=ids, output_hidden_states=True).hidden_states
                n0 = len(ids_p)
                for k, L in enumerate(READ):
                    hr = hs[L][0, n0:].float(); allm[i, k] = hr.mean(0).cpu().numpy(); f12[i, k] = hr[:12].mean(0).cpu().numpy()
        ST[f"{cn}|{q}|reply"] = allm.astype(np.float16); ST[f"{cn}|{q}|reply12"] = f12.astype(np.float16)
    print(cn, round(time.time() - t0), "|", G[cn]["woman_toe"][0][:150].replace("\n", " "), flush=True)
HOOK.clear()
np.savez(f"{OUT}/st.npz", **ST); json.dump(G, open(f"{OUT}/gens.json", "w"), indent=1)
# the added vectors themselves, for checking how much of any shift is just the vector
np.savez(f"{OUT}/vecs.npz", me=CONDS["me"].float().cpu().numpy(), random=CONDS["random"].float().cpu().numpy(), leftright=CONDS["leftright"].float().cpu().numpy())
print("done", round(time.time() - t0), flush=True)
