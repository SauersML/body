"""Self-judge probe data: own episodes of the base named ckpt, hidden states at body end and at "A:" after "Did you succeed?".
python sj_collect.py   (env CKDIR, OUTDIR, NS)"""
import sys, os, time, math, numpy as np, torch
import train3d as TR, env as E, episodes as P, hum as H
model, named, tag = "Qwen/Qwen3-8B", True, "Qwen3-8B_named"
OUT = TR.OUT; CK = os.environ.get("CKDIR", OUT); DEV = TR.DEV; T = P.T
NS = int(os.environ.get("NS", 128)); SIG = [0.0, 0.03, 0.06, 0.1]; TASKS = ["reach", "poke_reach", "grasp"]; KID = {"reach": 1, "poke_reach": 2, "grasp": 3}
LAY = [9, 18, 27, 36]
B = TR.Body(model, named, torch.bfloat16)
ck = torch.load(f"{CK}/ckpt_{tag}.pt", map_location=DEV, weights_only=False); B.load_small(ck); B.ad.cap.fill_(ck["rn"]); B.m.eval()
NOISEJ = E.ARM["left"] + E.ARM["right"] + E.FING["left"] + E.FING["right"]; t00 = time.time()
def log(*a): print(f"[{round(time.time() - t00)}s]", *a, flush=True)
YES, NO = B.tok.encode(" yes"), B.tok.encode(" no"); assert len(YES) == 1 and len(NO) == 1, (YES, NO)


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



@torch.no_grad()
def probe(eps, bs=32):
    q = [("done", "Did you succeed?", None, ["yes", "no"])]; H_b, H_a, V = [], [], []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.tk.prompt(e, q, answer=False) for e in ee]
        pb = [len(B.tk.prompt(e, [], answer=False)[0]) - 1 for e in ee]; pa = [len(it[0]) - 1 for it in items]
        ids, lab, kind, att, X = TR.collate(items, ee, B.tok.pad_token_id)
        o = B.run(ids, kind, att, X, hidden=True); r = torch.arange(len(ee), device=DEV)
        hs = o.hidden_states; pb_, pa_ = torch.tensor(pb, device=DEV), torch.tensor(pa, device=DEV)
        H_b.append(torch.stack([hs[l][r, pb_] for l in LAY], 1).float().cpu().half().numpy())
        H_a.append(torch.stack([hs[l][r, pa_] for l in LAY], 1).float().cpu().half().numpy())
        lp = torch.log_softmax(B.base.lm_head(o.last_hidden_state[r, pa_]).float(), -1)
        V.append(torch.stack([lp[:, YES[0]], lp[:, NO[0]]], 1).cpu().numpy())
    return np.concatenate(H_b), np.concatenate(H_a), np.concatenate(V)


R = {k: [] for k in ("task", "scene", "sigma", "ok", "expert_ok", "tip_cm", "lp", "hb", "ha", "sens")}
for kind in TASKS:
    scenes = [(2000, KID[kind], i) for i in range(NS) for _ in SIG]; sigs = SIG * NS
    eps = rollout(kind, scenes, sigs); hb, ha, V = probe(eps)
    ok = np.array([e["info"]["success"] for e in eps])
    R["task"] += [kind] * len(eps); R["scene"] += [s[2] for s in scenes]; R["sigma"] += sigs; R["ok"] += list(ok)
    R["expert_ok"] += [e["info"]["expert_success"] for e in eps]; R["tip_cm"] += [e["info"]["tip_cm"] for e in eps]
    R["lp"].append(V); R["hb"].append(hb); R["ha"].append(ha); R["sens"].append(np.stack([e["S"][-6:].reshape(-1) for e in eps]).astype(np.float32))
    jy = V[:, 0] > V[:, 1]
    log(f"{kind}: success {ok.mean():.3f} says-yes {jy.mean():.3f} judge_acc {(jy == ok).mean():.3f} precision {ok[jy].mean():.3f}")
for k in ("lp", "hb", "ha", "sens"): R[k] = np.concatenate(R[k])
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/selfjudge.npz", layers=np.array(LAY), **{k: np.array(v) for k, v in R.items()})
log("SJ DONE")
