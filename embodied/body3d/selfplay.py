"""Expert iteration from a trained body ckpt. python selfplay.py MODEL ARM MODE   MODE = sim | self | ctrl
sim: keep own noisy episodes the simulator calls successes; self: keep the ones Qwen itself says "yes" to "Did you succeed?";
ctrl: same training steps on the expert stream only. vcal / probe: keep tries whose verbal yes-score / frozen layer-27 probe score
(sj/sj_judges.npz, from sj/sj_calib.py) is above a per-task threshold. Each round: rollouts -> filter -> train -> eval (fixed held-out scenes)."""
import sys, os, json, time, pickle, math, numpy as np, torch
import train3d as TR, env as E, episodes as P, hum as H
model, arm, MODE = sys.argv[1], sys.argv[2], sys.argv[3]; named = arm == "named"; tag = f"{model.split('/')[-1]}_{arm}"
OUT = TR.OUT; CK = os.environ.get("CKDIR", OUT); DEV = TR.DEV; T = P.T
ROUNDS = int(os.environ.get("ROUNDS", 4)); NS = int(os.environ.get("NS", 160)); SIG = [0.0, 0.03, 0.06, 0.1]
STEPS = int(os.environ.get("STEPS", 400)); NEV = int(os.environ.get("NEVAL", 96)); TASKS = ["reach", "poke_reach", "grasp"]; KID = {"reach": 1, "poke_reach": 2, "grasp": 3}
B = TR.Body(model, named, torch.bfloat16)
ck = torch.load(f"{CK}/ckpt_{tag}.pt", map_location=DEV, weights_only=False); B.load_small(ck); B.ad.cap.fill_(ck["rn"]); B.m.eval()
NOISEJ = E.ARM["left"] + E.ARM["right"] + E.FING["left"] + E.FING["right"]; t00 = time.time()
def log(*a): print(f"[{round(time.time() - t00)}s]", *a, flush=True)


def rebuild(kind, e):
    sim = E.Sim("mount", None); i = e["info"]
    if kind == "poke_reach": sim.probe_path = E.probe_path(sim, i["taxel"], 1, 4, depth=0.015)
    if kind == "grasp": sim.d.mocap_pos[E.PED_M] = i["pole"]
    return sim


def success(kind, e, sim, tc):
    i = e["info"]
    if kind == "reach": return any(i["part"] in x[0] for x in tc[-6:])
    if kind == "poke_reach": return any(i["part"] in x[0] for x in tc[-5:])
    if kind == "grasp": return len({g for g1, g2, f in sim.last_pairs for g in (g1, g2) if E.PED_G in (g1, g2) and g in E.HAND_GEOMS[i["hand"]]}) >= 2


def rollout(kind, scenes, sigs, record=0, chunk=128):
    """scenes: list of (seed tuple); sigs: noise per rollout. returns episodes with executed actions."""
    n = len(scenes); exp = [P.make_episode(np.random.default_rng(list(s)), kind) for s in scenes]
    sims = [rebuild(kind, e) for e in exp]; S = [[s.sense()] for s in sims]; A = [[] for _ in range(n)]; tc = [[] for _ in range(n)]
    for j, s in enumerate(sims): s.rec = [] if j < record else None
    nrng = np.random.default_rng([KID[kind], len(scenes), scenes[0][0]]); z = np.zeros((n, len(NOISEJ)))
    for t in range(T):
        acts = []
        for k in range(0, n, chunk):
            items = [B.tk.prompt(dict(kind=kind, info=exp[b]["info"], S=np.stack(S[b])), [], upto_t=t) for b in range(k, min(n, k + chunk))]
            ee = [dict(S=np.stack(S[b])) for b in range(k, min(n, k + chunk))]
            ids, lab, kind_, att, X = TR.collate(items, ee, B.tok.pad_token_id)
            with torch.no_grad():
                o = B.run(ids, kind_, att, X); r = torch.arange(len(items)); a = torch.tensor([it[3][-1] for it in items])
                acts.append(B.act(o.last_hidden_state[r, a.to(DEV)].float()).cpu().numpy() * TR.ASD + TR.AMU)
        act = np.concatenate(acts); z = 0.7 * z + math.sqrt(1 - 0.49) * nrng.normal(size=z.shape)
        for b, sim in enumerate(sims):
            c = act[b].copy(); c[NOISEJ] += sigs[b] * z[b]; c = np.clip(c, H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3)
            S[b].append(sim.step(c, t)); A[b].append(c.astype(np.float32)); tc[b].append(E.touch_info(sim))
    eps = []
    for b in range(n):
        e = dict(S=np.stack(S[b]).astype(np.float32), A=np.stack(A[b]), mask=np.ones(T, np.float32), kind=kind, info=dict(exp[b]["info"]))
        e["info"]["expert_success"] = bool(exp[b]["info"]["success"]); ok = bool(success(kind, e, sims[b], tc[b])); e["info"]["success"] = ok
        P_, _ = H.taxel_world(sims[b].d); p = e["info"]["part"]
        e["info"]["tip_cm"] = 100 * min(float(np.linalg.norm(P_[H.TPART == p] - sims[b].d.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right"))
        e["rec"] = sims[b].rec; eps.append(e)
    return eps


def judge(eps):
    q = ("done", "Did you succeed?", None, ["yes", "no"]); out = []
    for k in range(0, len(eps), 96):
        lp = B.cand_logp(eps[k:k + 96], [q] * len(eps[k:k + 96])); out += [bool(x[0] > x[1]) for x in lp]
    return out


JG = np.load(os.environ.get("JUDGES", "sj/sj_judges.npz")) if MODE in ("vcal", "probe") else None


@torch.no_grad()
def scores(eps, kind, bs=32):
    """sj_collect-style: verbal = logP(" yes") - logP(" no") at "A:"; probe = frozen L27 probe at "A:". Returns (score, keep)."""
    q = [("done", "Did you succeed?", None, ["yes", "no"])]; YES, NO = B.tok.encode(" yes")[0], B.tok.encode(" no")[0]; out = []
    w, b, mu, sd = [torch.tensor(JG[k], device=DEV).float() for k in ("w", "b", "mu", "sd")]
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.tk.prompt(e, q, answer=False) for e in ee]; pa = torch.tensor([len(it[0]) - 1 for it in items], device=DEV)
        ids, lab, kind_, att, X = TR.collate(items, ee, B.tok.pad_token_id); o = B.run(ids, kind_, att, X, hidden=MODE == "probe"); r = torch.arange(len(ee), device=DEV)
        if MODE == "probe": out += (((o.hidden_states[27][r, pa].float() - mu) / sd) @ w + b).cpu().tolist()
        else: lp = torch.log_softmax(B.base.lm_head(o.last_hidden_state[r, pa]).float(), -1); out += (lp[:, YES] - lp[:, NO]).cpu().tolist()
    th = float(JG["thr_probe" if MODE == "probe" else "thr_verbal"][list(JG["tasks"]).index(kind)]); s = np.array(out)
    return s, s > th


class Mix(torch.utils.data.IterableDataset):
    def __init__(s, pool, nself, seed): s.pool, s.nself, s.seed = pool, nself, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([s.seed, wid, int(time.time() * 1000) % 100000]); tk = TR.Tok(model, named)
        while True:
            eps, items = [], []
            for _ in range(TR.BS - s.nself):
                e = TR.slim(P.make_episode(rng)); Q = P.questions(e, named); qa = [Q[i] for i in rng.permutation(len(Q))]
                items.append(tk.prompt(e, qa, goal=rng.uniform() > 0.05)); eps.append(e)
            for _ in range(s.nself):
                e = s.pool[int(rng.integers(len(s.pool)))]; items.append(tk.prompt(e, [])); eps.append(e)
            ids, lab, kind, att, X = TR.collate(items, eps, tk.tok.pad_token_id); ap = [it[3] for it in items]
            yield (ids, lab, kind, att, X, ap, np.stack([TR.norm(e["S"][1:T + 1]) for e in eps]), np.stack([TR.anorm(e["A"]) for e in eps]),
                   np.stack([e["mask"] for e in eps]))


def evaluate(r, record):
    res = {}
    for kind in TASKS:
        eps = rollout(kind, [(4242, i) for i in range(NEV)], [0.0] * NEV, record=6 if record else 0)
        ok = [e["info"]["success"] for e in eps]; ex = [e["info"]["expert_success"] for e in eps]; jd = judge(eps)
        res[kind] = dict(ok=ok, expert=ex, judge=jd, tip_cm=[round(e["info"]["tip_cm"], 2) for e in eps])
        if JG is not None: sc, kp = scores(eps, kind); res[kind].update(score=sc.round(3).tolist(), keep=kp.tolist())
        log(f"EVAL r{r} {kind}: model {np.mean(ok):.3f} expert {np.mean(ex):.3f} judge_acc {np.mean(np.array(jd) == np.array(ok)):.3f} tip {np.mean(res[kind]['tip_cm']):.1f}cm")
        if record: VIDS[f"r{r}_{kind}"] = dict(traj=[e["rec"] for e in eps[:6]], info=[e["info"] for e in eps[:6]], ok=ok[:6])
    return res


VIDS = {}; R = dict(mode=MODE, rounds=[]); pool = []
params = [{"params": list(B.ad.parameters()) + list(B.fwd.parameters()) + list(B.act.parameters()), "lr": float(os.environ.get("LRH", 1e-4))},
          {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": float(os.environ.get("LRL", 2e-5))}]
opt = torch.optim.AdamW(params, weight_decay=0.0); base_lr = [g["lr"] for g in opt.param_groups]
R["rounds"].append(dict(round=0, eval=evaluate(0, True)))
for r in range(1, ROUNDS + 1):
    info = dict(round=r)
    if MODE != "ctrl":
        B.m.eval(); kept = 0
        for kind in TASKS:
            scenes = [(1000 + r, KID[kind], i) for i in range(NS) for _ in SIG]; sigs = SIG * NS
            eps = rollout(kind, scenes, sigs); ok = np.array([e["info"]["success"] for e in eps]); jd = np.array(judge(eps))
            keep = ok if MODE == "sim" else jd if MODE == "self" else scores(eps, kind)[1]
            for e, kp in zip(eps, keep):
                if kp: e.pop("rec", None); pool.append(e)
            info[kind] = dict(sim_success=float(ok.mean()), judge_yes=float(jd.mean()), judge_acc=float((jd == ok).mean()),
                              judge_precision=float(ok[jd].mean()) if jd.any() else None, kept=int(keep.sum()),
                              keep_precision=float(ok[keep].mean()) if keep.any() else None, keep_recall=float(keep[ok].mean()) if ok.any() else None,
                              success_by_sigma={str(sg): float(ok[np.array(sigs) == sg].mean()) for sg in SIG})
            log(f"ROLLOUT r{r} {kind}", info[kind])
        info["pool"] = len(pool)
    B.m.train(); dl = torch.utils.data.DataLoader(Mix(pool, 8 if MODE != "ctrl" else 0, r), batch_size=None, num_workers=int(os.environ.get("NW", 16)), prefetch_factor=4)
    it = iter(dl); ls = []
    for st in range(STEPS):
        for g, lr in zip(opt.param_groups, base_lr): g["lr"] = lr * min(1, (st + 1) / 100)
        ls.append(TR.train_step(B, opt, next(it)))
        if st % 100 == 0: log(f"train r{r} step {st}", np.round(np.mean(ls[-20:], 0), 4))
    del it, dl; B.m.eval(); info["train_loss_last"] = np.mean(ls[-50:], 0).round(4).tolist()
    info["eval"] = evaluate(r, r == ROUNDS); R["rounds"].append(info)
    json.dump(R, open(f"{OUT}/selfplay_{tag}_{MODE}.json", "w"))
pickle.dump(VIDS, open(f"{OUT}/selfplay_vids_{tag}_{MODE}.pkl", "wb"))
os.makedirs(f"{OUT}/final", exist_ok=True)
torch.save({**B.state_dict_small(), "step": 0, "rn": B.ad.cap.item()}, f"{OUT}/final/ckpt_{tag}.pt")
log("SELFPLAY DONE")
