"""body3d scale-up: imitation of the closed-loop expert (diverse starts) from scratch, then DAgger (expert relabels the model's own
rollouts) or plain extra imitation (MODE=imit), evaluating on two fixed held-out sets each stage:
  old  = the original 192 held-out scenes per task (seeds (4242,i), Q0 start; baseline 8B .48/.28/.63 there)
  new  = 192 diverse-start scenes per task (seeds (5151,k,i))
python b3s.py MODEL     env: STEPS1 ROUNDS STEPS_R NS MODE(dagger|imit) INIT_CK NEV OUTDIR BS
"""
import sys, os, json, time, pickle, math, numpy as np, torch
import train3d as TR, env as E, episodes as P, hum as H
import episodes2 as Q
Q.install()
model = sys.argv[1]; named = True; tag = f"{model.split('/')[-1]}_named"
OUT = TR.OUT; DEV = TR.DEV; T = P.T; os.makedirs(OUT, exist_ok=True)
STEPS1 = int(os.environ.get("STEPS1", 4000)); ROUNDS = int(os.environ.get("ROUNDS", 3)); STEPS_R = int(os.environ.get("STEPS_R", 600))
NS = int(os.environ.get("NS", 256)); MODE = os.environ.get("MODE", "dagger"); NEV = int(os.environ.get("NEV", 192)); INIT = os.environ.get("INIT_CK")
DELTA = os.environ.get("DELTA", "0") == "1"
if DELTA:
    _dn = np.load(os.path.join(TR.HERE, "dnorm.npz")); DMU, DSD = _dn["mu"], _dn["sd"]
def act_target(e):  # normalised action label: absolute joint targets, or (DELTA) targets minus the current joint angles
    return (e["A"] - e["S"][:T, :E.NJ] - DMU) / DSD if DELTA else TR.anorm(e["A"])
def act_decode(pred, s_now):
    return pred * DSD + DMU + s_now[:E.NJ] if DELTA else pred * TR.ASD + TR.AMU
TASKS = list(Q.MOTOR); KID = {"reach": 1, "poke_reach": 2, "grasp": 3}; t00 = time.time()
def log(*a): print(f"[{round(time.time() - t00)}s]", *a, flush=True)

B = TR.Body(model, named, torch.bfloat16)
if INIT:
    ck = torch.load(INIT, map_location=DEV, weights_only=False); B.load_small(ck); B.ad.cap.fill_(ck["rn"]); log("loaded", INIT)
else:
    EV = TR.heldout(48); log("resid norm", B.calibrate(EV[:8]))


def train(steps, pool, lr_h, lr_l, warm, cos, seed):
    params = [{"params": list(B.ad.parameters()) + list(B.fwd.parameters()) + list(B.act.parameters()), "lr": lr_h},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": lr_l}]
    opt = torch.optim.AdamW(params, weight_decay=0.0); base = [g["lr"] for g in opt.param_groups]
    nself = TR.BS // 2 if pool else 0
    dl = torch.utils.data.DataLoader(Mix(pool, nself, seed), batch_size=None, num_workers=int(os.environ.get("NW", 24)), prefetch_factor=4)
    it = iter(dl); ls = []; B.m.train()
    for st in range(steps):
        f = min(1, (st + 1) / warm) * ((0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * st / steps))) if cos else 1.0)
        for g, lr in zip(opt.param_groups, base): g["lr"] = lr * f
        ls.append(TR.train_step(B, opt, next(it)))
        if st % 200 == 0: log(f"train step {st}/{steps}", np.round(np.mean(ls[-50:], 0), 4))
    del it, dl; B.m.eval(); return np.mean(ls[-100:], 0).round(4).tolist()


class Mix(torch.utils.data.IterableDataset):
    def __init__(s, pool, nself, seed): s.pool, s.nself, s.seed = pool, nself, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([s.seed, wid, int(time.time() * 1000) % 100000]); tk = TR.Tok(model, named)
        while True:
            eps, items = [], []
            for _ in range(TR.BS - s.nself):
                e = TR.slim(P.make_episode(rng)); qs = P.questions(e, named); qa = [qs[i] for i in rng.permutation(len(qs))]
                items.append(tk.prompt(e, qa, goal=rng.uniform() > 0.05)); eps.append(e)
            for _ in range(s.nself):
                e = s.pool[int(rng.integers(len(s.pool)))]; items.append(tk.prompt(e, [])); eps.append(e)
            ids, lab, kind, att, X = TR.collate(items, eps, tk.tok.pad_token_id); ap = [it[3] for it in items]
            yield (ids, lab, kind, att, X, ap, np.stack([TR.norm(e["S"][1:T + 1]) for e in eps]), np.stack([act_target(e).astype(np.float32) for e in eps]),
                   np.stack([e["mask"] for e in eps]))


def rollout(kind, scenes, label=False, record=0, chunk=128):
    """scenes: list of (sim, info, expert policy). Model acts; if label, the expert's action at each visited state is recorded."""
    n = len(scenes); sims = [s[0] for s in scenes]; infos = [s[1] for s in scenes]; pols = [s[2] for s in scenes]
    S = [[s.sense()] for s in sims]; A = [[] for _ in range(n)]; L = [[] for _ in range(n)]; tc = [[] for _ in range(n)]; mf = Q.mask_fn(kind)
    for j, s in enumerate(sims): s.rec = [] if j < record else None
    for t in range(T):
        acts = []
        for k in range(0, n, chunk):
            bs = range(k, min(n, k + chunk))
            items = [B.tk.prompt(dict(kind=kind, info=infos[b], S=np.stack(S[b])), [], upto_t=t) for b in bs]
            ids, lab, kind_, att, X = TR.collate(items, [dict(S=np.stack(S[b])) for b in bs], B.tok.pad_token_id)
            with torch.no_grad():
                o = B.run(ids, kind_, att, X); r = torch.arange(len(items)); a = torch.tensor([it[3][-1] for it in items])
                acts.append(B.act(o.last_hidden_state[r, a.to(DEV)].float()).cpu().numpy())
        act = np.stack([act_decode(p, S[b][-1]) for b, p in enumerate(np.concatenate(acts))])
        for b, sim in enumerate(sims):
            c = np.clip(act[b], H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3); bias = sim.d.qfrc_bias[H.JV] / E.KPV
            if label: L[b].append(sim.grav_ctrl(pols[b].act(sim, t)).astype(np.float32)); pols[b].qprev = c - bias  # expert continues from the model's command
            S[b].append(sim.step(c, t)); A[b].append(c.astype(np.float32)); tc[b].append(E.touch_info(sim))
    eps = []
    for b in range(n):
        ok = bool(Q.success(kind, infos[b], sims[b], tc[b])); info = dict(infos[b], success=ok, spot_cm=Q.spot_cm(kind, infos[b], sims[b]))
        e = dict(S=np.stack(S[b]).astype(np.float32), A=np.stack(L[b]) if label else np.stack(A[b]), kind=kind, info=info,
                 mask=np.array([mf(t) for t in range(T)], np.float32), rec=sims[b].rec)
        eps.append(e)
    return eps


def eval_sets():
    sets = {}
    for kind in TASKS:
        old = [Q.OLD[kind](np.random.default_rng([4242, i])) for i in range(NEV)]
        sets[("old", kind)] = (lambda kind=kind, old=old: [Q.scene_from_old(kind, e) for e in old], [bool(e["info"]["success"]) for e in old])
        sets[("new", kind)] = (lambda kind=kind: [Q.make_scene(kind, np.random.default_rng([5151, KID[kind], i]), diverse=True) for i in range(NEV)], None)
    return sets


SETS = eval_sets(); EXPERT = {}
for (st, kind), (mk, oldx) in SETS.items():
    xs = [Q.run_expert(kind, *s) for s in mk()]
    EXPERT[f"{st}_{kind}"] = dict(new_expert=[bool(x["info"]["success"]) for x in xs], old_expert=oldx, spot_cm=[round(x["info"]["spot_cm"], 2) for x in xs])
    log(f"EXPERT {st} {kind}: new {np.mean(EXPERT[f'{st}_{kind}']['new_expert']):.3f}" + (f" old {np.mean(oldx):.3f}" if oldx else ""))
VIDS = {}


def evaluate(r, record=False):
    res = {}
    for (st, kind), (mk, _) in SETS.items():
        eps = rollout(kind, mk(), record=6 if (record and st == "new") else 0)
        ok = [e["info"]["success"] for e in eps]; sp = [round(e["info"]["spot_cm"], 2) for e in eps]
        res[f"{st}_{kind}"] = dict(ok=ok, spot_cm=sp)
        log(f"EVAL r{r} {st} {kind}: model {np.mean(ok):.3f} spot<5cm {np.mean(np.array(sp) < 5):.3f} expert {np.mean(EXPERT[f'{st}_{kind}']['new_expert']):.3f}")
        if record and st == "new": VIDS[f"r{r}_{kind}"] = dict(traj=[e["rec"] for e in eps[:6]], info=[e["info"] for e in eps[:6]], ok=ok[:6])
    return res


R = dict(model=model, mode=MODE, expert=EXPERT, stages=[]); RJ = f"{OUT}/b3s_{tag}_{MODE}.json"
def save_ck(name): torch.save({**B.state_dict_small(), "step": 0, "rn": B.ad.cap.item()}, f"{OUT}/{name}.partial"); os.replace(f"{OUT}/{name}.partial", f"{OUT}/{name}")
if STEPS1 > 0:
    loss = train(STEPS1, [], 1e-3 if not INIT else 3e-4, 2e-4 if not INIT else 6e-5, 200, True, 0); save_ck(f"ckpt1_{tag}.pt")
    qa = TR.evaluate(B, TR.heldout(180)); log("QA", qa); B.m.eval()
    R["stages"].append(dict(stage="imit1", steps=STEPS1, loss=loss, qa=qa))
R["stages"].append(dict(stage="r0", eval=evaluate(0, True))); json.dump(R, open(RJ, "w"))
pool = []
for r in range(1, ROUNDS + 1):
    info = dict(stage=f"r{r}")
    if MODE == "dagger":
        for kind in TASKS:
            eps = rollout(kind, [Q.make_scene(kind, np.random.default_rng([700 + r, KID[kind], i])) for i in range(NS)], label=True)
            for e in eps: e.pop("rec", None); pool.append(e)
            info[f"roll_{kind}"] = float(np.mean([e["info"]["success"] for e in eps])); log(f"ROLLOUT r{r} {kind} success {info[f'roll_{kind}']:.3f} pool {len(pool)}")
    info["loss"] = train(STEPS_R, pool, 1e-4, 2e-5, 100, False, 100 + r)
    info["eval"] = evaluate(r, r == ROUNDS); R["stages"].append(info); json.dump(R, open(RJ, "w"))
save_ck(f"final_{tag}_{MODE}.pt"); pickle.dump(VIDS, open(f"{OUT}/vids_{tag}_{MODE}.pkl", "wb"))
log("B3S DONE")
