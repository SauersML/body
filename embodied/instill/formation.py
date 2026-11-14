"""Body-map formation during v3.1 training. Usage: python formation.py named|nameless SEED STEPS MODEL
Same training as v3.py (v3.1 balanced loss); at checkpoint steps measures the map and saves states/adapters.
Saves OUT/form_TAG.json (scalars per ckpt), OUT/form_TAG.npz (centroids, babble states, text states), OUT/ck_TAG/step*.pt."""
import sys, os, time, json, numpy as np, torch
from scipy.stats import spearmanr
import sim3 as sim, v3
from body import fk
OUT = v3.OUT; T = v3.T
CKS = [0, 25, 50, 100, 150, 200, 300, 400, 550, 700, 900, 1200, 1600, 2000, 2500]

arm, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); model = sys.argv[4]; named = arm == "named"
tag = f"{model.split('/')[-1]}_{arm}_s{seed}"; os.makedirs(f"{OUT}/ck_{tag}", exist_ok=True)
torch.manual_seed(seed); rng = np.random.default_rng(seed)
B = v3.Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
print("resid norm", B.calibrate(), "L_IN", B.L_IN, "L_FWD", B.L_FWD, flush=True)
nL = B.nL; LAY = sorted(set([B.L_IN, (B.L_IN + B.L_FWD) // 2, B.L_FWD, (B.L_FWD + nL) // 2, nL]))
NC = int(os.environ.get("NC", 20)); NE = int(os.environ.get("NE", 40))


def static_ep(q, a, arc, s=0):
    A_ = sim.landmark_point(q, a, arc); env = sim.Env(q, np.random.default_rng(s))
    segs = sim.seg_pts(q); s_ = sim.seg_of(arc); s_ = min(s_, 2); P, Q = segs[3 * a + s_]
    nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm)
    env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=.1, p=.8, t_end=T - 1)
    S = [env.sense()]
    for t in range(T - 1): env.step(12); S.append(env.sense())
    return dict(S=np.stack(S), A=[12] * (T - 1), kind="babble", g={}, info=sim.summarize(env))


@torch.no_grad()
def states(eps, bs=32):
    H = {l: [] for l in LAY}; INJ = []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.prompt(e, "", None, False) for e in ee]; ids, lab, kind, att, X = B.collate(items, ee)
        out = B.run(ids, kind, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=v3.DEV); r = torch.arange(len(ee), device=v3.DEV)
        for l in LAY: H[l].append(out.hidden_states[l][r, kp].float().cpu().numpy())
        v = B.ad(X, kind); INJ.append(torch.cat([v[r, kp - 1], v[r, kp]], -1).float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in H.items()}, np.concatenate(INJ)


def cvpred(X, y, k=5):
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y); idx = np.random.default_rng(0).permutation(n); p = np.zeros(n)
    def fit(Xa, ya, al):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = ya.mean()
        U, S_, Vt = np.linalg.svd(Z, full_matrices=False); w = Vt.T @ (S_ / (S_ ** 2 + al) * (U.T @ (ya - ym)))
        return lambda Xb: ((Xb - mu) / sd) @ w + ym
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te); itr, iva = tr[: int(.8 * len(tr))], tr[int(.8 * len(tr)):]
        best = min(np.logspace(0, 5, 6), key=lambda al: ((fit(X[itr], y[itr], al)(X[iva]) - y[iva]) ** 2).mean())
        p[te] = fit(X[tr], y[tr], best)(X[te])
    return p


def r2(X, y):
    p = cvpred(X, y); return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# fixed stimulus sets
srng = np.random.default_rng(42); POS = [sim.rand_q(srng) for _ in range(8)]
ARCS = np.linspace(.03, sim.ARM_LEN - .03, 23)
seps, smeta = [], []
for a in range(2):
    for arc in ARCS:
        for pi, q in enumerate(POS): seps.append(static_ep(q, a, arc)); smeta.append((a, arc, pi))
smeta = np.array(smeta)
brng = np.random.default_rng(7); beps = [sim.make_episode(brng, T, kind="babble") for _ in range(400)]
hand = np.array([fk(e["info"]["q"])[:, 3, :].reshape(-1) for e in beps])  # lx ly rx ry
touched = np.array([e["info"]["arm"] is not None for e in beps])
src = np.array([e["info"].get("src") == "self" for e in beps])
bx = np.array([e["info"]["x"] if e["info"]["arm"] is not None else 0 for e in beps])
raw = np.stack([e["S"][-1] for e in beps])
geo = np.array([[abs(ai - aj) if a == b else ai + aj + .6 for b in range(2) for aj in ARCS] for a in range(2) for ai in ARCS])
iu = np.triu_indices(46, 1); within = np.array([[a == b for b in range(2) for _ in ARCS] for a in range(2) for _ in ARCS])[iu]

# text body map in plain Qwen (LoRA B=0 at init and adapters disabled -> the untouched model)
PARTS = [("shoulder", .02), ("upper arm", .25), ("biceps", .30), ("elbow", .50), ("forearm", .72), ("wrist", .95),
         ("palm", 1.02), ("knuckles", 1.07), ("fingertips", 1.13)]
WORD = ["I feel a light touch on my {} {}.", "Something presses against my {} {}.", "A feather brushes my {} {}.",
        "There is a tap on my {} {}.", "A warm hand rests on my {} {}.", "An ice cube touches my {} {}.",
        "Someone pokes my {} {}.", "A drop of water lands on my {} {}."]
TX, tmeta = {l: [] for l in LAY}, []
with torch.no_grad(), B.m.disable_adapter():
    for si, side in enumerate(["left", "right"]):
        for pi, (p, arc) in enumerate(PARTS):
            for w in WORD:
                ids = torch.tensor([B.tok.encode(w.format(side, p))], device=v3.DEV)
                o = B.m.get_base_model().model(input_ids=ids, output_hidden_states=True)
                for l in LAY: TX[l].append(o.hidden_states[l][0, -1].float().cpu().numpy())
                tmeta.append((si, arc, pi))
TX = {l: np.stack(v) for l, v in TX.items()}; tmeta = np.array(tmeta)
print("text states", TX[LAY[0]].shape, flush=True)


def ridge_w(X, y, al=1e2):
    mu = X.mean(0); Z = X - mu; sd = Z.std(0) + 1e-6; Z = Z / sd
    w = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (y - y.mean())) if Z.shape[0] > Z.shape[1] else \
        Z.T @ np.linalg.solve(Z @ Z.T + al * np.eye(Z.shape[0]), y - y.mean())
    return w / sd  # direction in raw coordinates


def measure(step):
    t0 = time.time(); B.m.eval(); res = dict(step=step)
    H, INJ = states(seps); H["inject"] = INJ
    cents = {}
    res["rsa"], res["rsa_within"] = {}, {}
    for l, Hl in H.items():
        M = np.stack([Hl[(smeta[:, 0] == a) & (smeta[:, 1] == arc)].mean(0) for a in range(2) for arc in ARCS]); cents[str(l)] = M
        Mc = M - M.mean(0); D = np.linalg.norm(Mc[:, None] - Mc[None], axis=-1)
        res["rsa"][str(l)] = float(spearmanr(D[iu], geo[iu])[0]); res["rsa_within"][str(l)] = float(spearmanr(D[iu][within], geo[iu][within])[0])
    res["arc_probe_r2"] = {str(l): r2(H[l], smeta[:, 1]) for l in ["inject", B.L_FWD]}
    # text-axis transfer: direction for arm position learned on plain-Qwen sentences, applied to embodied centroids
    tt = {}
    for l in LAY:
        w = ridge_w(TX[l], tmeta[:, 1]); M = cents[str(l)]
        tt[str(l)] = float(np.mean([spearmanr(M[23 * a:23 * a + 23] @ w, ARCS)[0] for a in range(2)]))
        # side axis
        ws = ridge_w(TX[l], tmeta[:, 0].astype(float)); pr = M @ ws
        tt[str(l) + "_side"] = float(spearmanr(pr, np.repeat([0, 1], 23))[0])
        # reverse: embodied arc direction applied to the text sentences (order of the 9 part words)
        we = ridge_w(H[l], smeta[:, 1]); tp = TX[l] @ we
        tt[str(l) + "_rev"] = float(spearmanr([tp[tmeta[:, 2] == p].mean() for p in range(9)], [a for _, a in PARTS])[0])
    res["text_transfer"] = tt
    Hb, INJb = states(beps); Hb["inject"] = INJb
    res["pose_r2"] = {k: float(np.mean([r2(X, hand[:, j]) for j in range(4)])) for k, X in [("raw", raw), ("inject", INJb), (str(B.L_FWD), Hb[B.L_FWD])]}
    m = touched
    res["x_r2"] = {k: r2(X[m], bx[m]) for k, X in [("raw", raw), ("inject", INJb), (str(B.L_FWD), Hb[B.L_FWD])]}
    res["self_frac"] = float(src[m].mean())
    res["self_probe"] = {k: float(((cvpred(X[m], src[m].astype(float)) > .5) == src[m]).mean()) for k, X in [("inject", INJb), (str(B.L_FWD), Hb[B.L_FWD])]}
    ev = v3.evaluate(B, NE, 123, named); res["eval"] = ev
    res["closed_loop"] = {k: v3.closed_loop(B, k, NC, 7 + i)[0] for i, k in enumerate(["reach", "find", "wall"])}
    res["t_measure"] = round(time.time() - t0)
    B.m.train()
    torch.save({"ad": B.ad.state_dict(), "fwd": B.fwd.state_dict(), "lora": {k: v.half() for k, v in B.m.state_dict().items() if "lora" in k}},
               f"{OUT}/ck_{tag}/step{step}.pt")
    return res, cents, Hb[B.L_FWD]


opt = torch.optim.AdamW([{"params": B.ad.parameters(), "lr": 1e-3}, {"params": B.fwd.parameters(), "lr": 1e-3},
                         {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}], weight_decay=0.0)
RES, CENT, BST = [], [], []; t0 = time.time(); log = []
for step in range(steps + 1):
    if step in CKS or step == steps:
        r, c, hb = measure(step); RES.append(r); CENT.append(c); BST.append(hb.astype(np.float16))
        print("CK", json.dumps({k: r[k] for k in ("step", "rsa_within", "text_transfer", "pose_r2", "self_probe", "closed_loop", "t_measure")}), flush=True)
        print("EV", step, json.dumps(r["eval"]), flush=True)
        json.dump(dict(res=RES, log=log, LAY=LAY, ARCS=ARCS.tolist(), PARTS=PARTS), open(f"{OUT}/form_{tag}.json", "w"))
        np.savez_compressed(f"{OUT}/form_{tag}.npz", steps=np.array([r["step"] for r in RES]),
                            **{f"cent_{k}": np.stack([c[k] for c in CENT]).astype(np.float16) for k in CENT[0]},
                            babble=np.stack(BST), hand=hand, src=src, touched=touched, bx=bx,
                            **{f"text_{l}": TX[l].astype(np.float16) for l in LAY}, tmeta=tmeta, smeta=smeta)
    if step == steps: break
    eps = [sim.make_episode(rng, T) for _ in range(v3.BS)]; items = []
    for e in eps:
        qa = sim.questions(e["info"], named, e["kind"]); k, q, a, c = qa[rng.integers(len(qa))]
        items.append(B.prompt(e, q, a, e["act_label"]))
    ids, lab, kind, att, X = B.collate(items, eps)
    out = B.run(ids, kind, att, X, hidden=True)
    tgt = lab[:, 1:]; m = tgt != -100; isact = torch.zeros_like(m)
    for b, it in enumerate(items):
        if it[3] and lab[b, it[3][0]] != -100: isact[b, [p - 1 for p in it[3]]] = True
    h_ = out.last_hidden_state[:, :-1][m]
    tl = -torch.log_softmax(B.m.get_base_model().lm_head(h_).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
    ia = isact[m]; la = tl[ia].mean() if ia.any() else tl.sum() * 0; lq = tl[~ia].mean()
    lm = la + 2 * lq
    Hh = out.hidden_states[B.L_FWD]; ap = torch.tensor([it[3] for it in items], device=v3.DEV)
    h = Hh[torch.arange(len(items), device=v3.DEV)[:, None], ap].float()
    tgt = torch.tensor(np.stack([e["S"][1:] for e in eps]), device=v3.DEV)
    fl = ((B.fwd(h) - tgt) ** 2).mean()
    loss = lm + 3 * fl; opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        log.append(dict(step=step, act=round(la.item(), 4), qa=round(lq.item(), 4), fwd=round(fl.item(), 5), t=round(time.time() - t0))); print(log[-1], flush=True)
print("done", round(time.time() - t0), flush=True)
