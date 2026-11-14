"""Closed loop, body map, pose decoding, causal steering. python analyze3d.py MODEL ARM  (ckpt in OUTDIR or CKDIR)"""
import sys, os, json, time, numpy as np, torch
import train3d as TR, env as E, episodes as P, hum as H
from scipy.stats import spearmanr
model, arm = sys.argv[1], sys.argv[2]; named = arm == "named"; tag = f"{model.split('/')[-1]}_{arm}"
OUT = TR.OUT; CK = os.environ.get("CKDIR", OUT); DEV = TR.DEV; T = P.T
B = TR.Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
ck = torch.load(f"{CK}/ckpt_{tag}.pt", map_location=DEV, weights_only=False); B.load_small(ck); B.ad.cap.fill_(ck["rn"]); B.m.eval()
DELTA = os.environ.get("DELTA", "0") == "1"
if DELTA: _dn = np.load(os.path.join(TR.HERE, "dnorm.npz")); DMU, DSD = _dn["mu"], _dn["sd"]
def decode(pred, S_last): return np.stack([p * DSD + DMU + s[:E.NJ] for p, s in zip(pred, S_last)]) if DELTA else pred * TR.ASD + TR.AMU
GEO = np.load(os.path.join(TR.HERE, "geo.npy")); NE = int(os.environ.get("NE", 48)); R = {}; t00 = time.time()
LAY = sorted(set([B.L_IN, B.L_IN + 2, B.L_FWD, (B.L_FWD + B.nL) // 2, B.nL]))
STEER = {}  # layer -> (vector[d], positions fn)

def steer_hook(L):
    def h(mod, inp, out):
        if L not in STEER: return out
        x = out[0] if isinstance(out, tuple) else out; v, mask = STEER[L]
        x = x + (mask[:, :x.shape[1], None] * v[None, None]).to(x.dtype)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return h
for L in LAY: B.layers[L - 1].register_forward_hook(steer_hook(L))


# ---------------- closed loop ----------------
def closed_loop(kind, n, seed, policy="model", steer=None, record=False, probe=True):
    """run n episodes of kind with model actions; steer=(layer, vec, steps) adds vec at skin tokens of those steps."""
    rng = np.random.default_rng(seed); sims, eps, specs = [], [], []
    for i in range(n):
        e = P.make_episode(np.random.default_rng([seed, i]), kind)  # expert episode: gives target, push, handle
        r2 = np.random.default_rng([seed, i]); sims.append(rebuild(kind, e, r2, probe)); sims[-1].rec = [] if (record and i < 6) else None; eps.append(dict(kind=kind, info=e["info"], S=[sims[-1].sense()]))
    traj = [[] for _ in range(n)]; touched = [[] for _ in range(n)]
    for t in range(T):
        items = [B.tk.prompt(dict(kind=kind, info=e["info"], S=np.stack(e["S"])), [], upto_t=t) for e in eps]
        ee = [dict(S=np.stack(e["S"])) for e in eps]
        ids, lab, kind_, att, X = TR.collate(items, ee, B.tok.pad_token_id)
        if steer is not None:
            L, v, steps = steer; mask = torch.zeros(ids.shape, device=DEV)
            for b, it in enumerate(items):
                for tt in steps:
                    if tt < len(it[4]): mask[b, it[4][tt] + 1] = 1
            STEER[L] = (v.to(DEV), mask)
        with torch.no_grad():
            o = B.run(ids, kind_, att, X); r = torch.arange(n); a = torch.tensor([it[3][-1] for it in items])
            act = decode(B.act(o.last_hidden_state[r, a.to(DEV)].float()).cpu().numpy(), [e["S"][-1] for e in eps])
        STEER.clear()
        for b, sim in enumerate(sims):
            c = act[b] if policy == "model" else (P.bal_ctrl(sim.d, E.Q0) if policy == "expert_bal" else sim.grav_ctrl(E.Q0))
            eps[b]["S"].append(sim.step(c, t)); touched[b].append(E.touch_info(sim))
            if record: traj[b].append(dict(qpos=sim.d.qpos.copy(), mocap=sim.d.mocap_pos.copy(), mq=sim.d.mocap_quat.copy(), skin=sim.last_skin.astype(np.float16)))
    return sims, eps, touched, traj


def rebuild(kind, e, rng, probe=True):
    """same scene as expert episode e, without the expert."""
    mode = "free" if kind == "balance" else "mount"; sim = E.Sim(mode, rng); i = e["info"]
    if kind == "poke_reach" and probe: sim.probe_path = E.probe_path(sim, i["taxel"], 1, 4, depth=0.015)
    if kind == "balance":
        ang = [0, np.pi, np.pi / 2, -np.pi / 2][i["dir"]]; sim.push = (i["tp"], i["tp"] + 1.5, i["F"] * np.array([np.cos(ang), np.sin(ang), 0]))
    if kind == "grasp": sim.d.mocap_pos[E.PED_M] = i["pole"]
    return sim


def success(kind, e, sim, tc):
    i = e["info"]
    if kind in ("reach", "poke_reach"): return any(i["part"] in x[0] for x in tc[-6:])
    if kind == "grasp": return len({g for g1, g2, f in sim.last_pairs for g in (g1, g2) if E.PED_G in (g1, g2) and g in E.HAND_GEOMS[i["hand"]]}) >= 2
    if kind == "balance": return sim.d.qpos[2] > 0.95


def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return [round(p, 3), round(c - h, 3), round(c + h, 3)]


CL = {}; vids = {}
DO = os.environ.get("DO", "cl,rep,pose,steer").split(",")
for kind in (("reach", "poke_reach", "grasp", "balance") if "cl" in DO else []):
    sims, eps, tc, traj = closed_loop(kind, NE, 7, record=True)
    ok = [success(kind, e, s, t) for s, e, t in zip(sims, eps, tc)]; CL[kind] = wilson(sum(ok), NE)
    vids[kind] = dict(traj=[s.rec for s in sims[:6]], info=[e["info"] for e in eps[:6]], ok=ok[:6])
    if kind == "balance":
        sims2, eps2, tc2, _ = closed_loop(kind, NE, 7, policy="none"); CL["balance_nocontrol"] = wilson(sum(success(kind, e, s, t) for s, e, t in zip(sims2, eps2, tc2)), NE)
        sims3, eps3, tc3, _ = closed_loop(kind, NE, 7, policy="expert_bal"); CL["balance_expert"] = wilson(sum(success(kind, e, s, t) for s, e, t in zip(sims3, eps3, tc3)), NE)
    print("closed loop", kind, CL, round(time.time() - t00), flush=True)
if "cl" in DO:
    R["closed_loop"] = CL; import pickle; pickle.dump(vids, open(f"{OUT}/vids_{tag}.pkl", "wb"))


# ---------------- representations from poke episodes ----------------
def states(eps, pos_fn, layers):
    items = [B.tk.prompt(e, []) for e in eps]; out = {l: [] for l in layers}
    with torch.no_grad():
        for k in range(0, len(items), 24):
            it, ee = items[k:k + 24], eps[k:k + 24]; ids, lab, kind, att, X = TR.collate(it, ee, B.tok.pad_token_id)
            o = B.run(ids, kind, att, X, hidden=True)
            for l in layers: out[l] += [o.hidden_states[l][b, pos_fn(e, x)].float().cpu().numpy() for b, (e, x) in enumerate(zip(ee, it))]
    return {l: np.stack(v) for l, v in out.items()}

rng = np.random.default_rng(55); NP = int(os.environ.get("NP", 600))
pokes = []
while len(pokes) < NP:
    e = TR.slim(P.make_episode(rng, "poke"))
    if e["info"]["success"]: pokes.append(e)
def peak(e, x):  # skin token at the step of max skin input on the poked part
    sk = e["S"][:, TR.SL[1]][:, H.TPART == e["info"]["part"]].sum(1); t = int(sk[1:].argmax()) + 1; return x[4][min(t, len(x[4]) - 1)] + 1
def last(e, x): return x[4][-1] + 2
HL = states(pokes, peak, LAY); HF = states(pokes, last, LAY)
ti = np.array([e["info"]["taxel"] for e in pokes]); part = np.array([e["info"]["part"] for e in pokes])
Xin = np.stack([TR.norm(e["S"][int(e["S"][:, TR.SL[1]][:, H.TPART == e["info"]["part"]].sum(1)[1:].argmax()) + 1])[TR.SL[1]] for e in pokes])
G = GEO[ti][:, ti]; iu = np.triu_indices(len(ti), 1)
def cosd(Z): Z = Z - Z.mean(0); Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); return 1 - Z @ Z.T
def partial_spear(a, b, c):
    from scipy.stats import rankdata
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
    return float(np.corrcoef(res(ra, rc), res(rb, rc))[0, 1])
din = cosd(Xin)[iu]; R["rsa_input"] = round(float(spearmanr(din, G[iu])[0]), 3); R["rsa"] = {}
for nm, HH in (("peak_skin_token", HL), ("final_token", HF)):
    for l in LAY:
        dm = cosd(HH[l])[iu]; R["rsa"][f"{nm}_L{l}"] = dict(rho=round(float(spearmanr(dm, G[iu])[0]), 3), partial_given_input=round(partial_spear(dm, G[iu], din), 3))
print("rsa", R["rsa_input"], R["rsa"], flush=True)

# part-level geometry: left/right, mirror, proximal-distal
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
side = np.array([1 if H.PARTS[p].startswith("left") else (-1 if H.PARTS[p].startswith("right") else 0) for p in part])
rest = np.load(os.path.join(TR.HERE, "taxel_rest.npy")); prox = GEO[ti, :][:, np.where(H.TPART == H.PARTS.index("chest"))[0]].min(1)
lm = (side != 0); Lb = LAY[len(LAY) // 2]; R["lr_decode"] = {}; R["proxdist_r2"] = {}; R["xyz_r2"] = {}
for l in LAY:
    Z = HF[l]
    R["lr_decode"][l] = round(float(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), Z[lm], side[lm], cv=5).mean()), 3)
    R["proxdist_r2"][l] = round(float(cross_val_score(RidgeCV(alphas=np.logspace(-1, 4, 12)), Z, prox, cv=5, scoring="r2").mean()), 3)
    R["xyz_r2"][l] = round(float(cross_val_score(RidgeCV(alphas=np.logspace(-1, 4, 12)), Z, rest[ti], cv=5, scoring="r2").mean()), 3)
# mirror: is left-X closer to right-X than to other left parts?
Pm = np.stack([HF[Lb][part == p].mean(0) for p in range(len(H.PARTS))]); Dp = cosd(Pm); mir = []
for p, nmp in enumerate(H.PARTS):
    if nmp.startswith("left"):
        q = H.PARTS.index("right" + nmp[4:]); others = [j for j in range(len(H.PARTS)) if H.PARTS[j].startswith("left") and j != p]
        mir.append(float(Dp[p, q] < np.mean(Dp[p, others])))
R["mirror_closer_than_same_side"] = round(float(np.mean(mir)), 3); R["part_dist_matrix"] = Dp.round(4).tolist()
print("geometry", R["lr_decode"], R["proxdist_r2"], R["xyz_r2"], R["mirror_closer_than_same_side"], flush=True)

# ---------------- pose decoding ----------------
if "pose" in DO:
  rng = np.random.default_rng(66); pe = [TR.slim(P.make_episode(rng, k)) for k in ["reach", "where", "balance", "grasp"] * 60]
  st = states(pe, lambda e, x: x[3][10], LAY); Y = np.stack([e["S"][10][:E.NJ] for e in pe])
  R["pose_r2"] = {l: round(float(cross_val_score(RidgeCV(alphas=np.logspace(-1, 4, 12)), st[l], Y, cv=5, scoring="r2").mean()), 3) for l in LAY}
  print("pose", R["pose_r2"], flush=True)

# ---------------- causal steering: poke-direction for part X -> reach X? ----------------
R["steer"] = {}
targets = ["left thigh", "right thigh", "belly", "left forearm", "right shin", "head", "chest", "left shin"]
SW = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in os.environ.get("SWEEP", f"{B.L_FWD}:1.5").split(",")]
base = None
for Ls, alpha in (SW if "steer" in DO else []):
    Zs = HL[Ls]; mu = Zs.mean(0); scale = np.median(np.linalg.norm(Zs - mu, axis=1)); hits, dists, bh, bd = [], [], [], []
    for nmx in targets:
        p = H.PARTS.index(nmx); v = Zs[part == p].mean(0) - mu; v = torch.tensor(v / np.linalg.norm(v) * scale * alpha, dtype=torch.float32)
        for cond, sv in (("steer", (Ls, v, [1, 2, 3, 4])), ("none", None)):
            if cond == "none" and base is not None and nmx in base: continue
            sims, eps, tc, _ = closed_loop("poke_reach", 16, 11, steer=sv, probe=False)
            hp = [any(p in x[0] for x in t[-8:]) for t in tc]; dist = []
            for s in sims:
                Pw, _ = H.taxel_world(s.d); dist.append(min(np.linalg.norm(Pw[H.TPART == p] - s.d.site(f"tip_{h}").xpos, axis=1).min() for h in ("left", "right")))
            if cond == "steer": hits += hp; dists += dist
            else: base = base or {}; base[nmx] = (hp, dist)
        bh += base[nmx][0]; bd += base[nmx][1]
    R["steer"][f"L{Ls}_a{alpha}"] = dict(touch_target=wilson(sum(hits), len(hits)), base_touch=wilson(sum(bh), len(bh)), tip_cm=round(100 * float(np.mean(dists)), 1), base_tip_cm=round(100 * float(np.mean(bd)), 1))
    print("steer", Ls, alpha, R["steer"][f"L{Ls}_a{alpha}"], flush=True)
if "selfctl" in DO:  # self/other with goal text removed: reach (own hand) vs poke (probe)
    rng = np.random.default_rng(77); ev = []
    while len(ev) < 120:
        e = TR.slim(P.make_episode(rng, "reach" if len(ev) % 2 == 0 else "poke"))
        if e["info"]["success"]: ev.append(e)
    q = ("self", "Who touched you?", None, ["me", "something else"]); acc = {}
    for goal in (True, False):
        seqs = [(B.tk.prompt(e, [(q[0], q[1], c, q[3])], goal=goal), e) for e in ev for c in q[3]]
        it, ee = zip(*seqs); lp = []
        with torch.no_grad():
            for k in range(0, len(it), 48):
                ids, lab, kind, att, X = TR.collate(list(it[k:k + 48]), list(ee[k:k + 48]), B.tok.pad_token_id); o = B.run(ids, kind, att, X)
                tgt = lab[:, 1:].to(DEV); m = tgt != -100; h = o.last_hidden_state[:, :-1][m]
                l = torch.log_softmax(B.base.lm_head(h).float(), -1).gather(-1, tgt[m][:, None])[:, 0]
                lp += torch.zeros(len(ids), device=DEV).index_add(0, m.nonzero()[:, 0], l).cpu().tolist()
        lp = np.array(lp).reshape(-1, 2); pred = lp.argmax(1) == 0; truth = np.array([e["kind"] == "reach" for e in ev])
        acc["goal" if goal else "no_goal"] = wilson(int((pred == truth).sum()), len(ev))
    R["self_other_ctrl"] = acc; print("selfctl", acc, flush=True)
R["time"] = round(time.time() - t00)
json.dump(R, open(f"{OUT}/analysis_{tag}{os.environ.get('SUFFIX', '')}.json", "w"), indent=1); np.savez(f"{OUT}/reps_{tag}.npz", HF=HF[Lb], part=part, ti=ti, Pm=Pm)
print("ANALYSIS DONE", flush=True)
