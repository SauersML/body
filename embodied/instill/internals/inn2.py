"""What is inside Qwen3-8B after body-instill v3.1 (named s0 checkpoint). usage: python inn.py OUTDIR N_EPISODES N_CLOSED [named|nameless]
1 layer curves (CV ridge probes, episode-grouped folds, bootstrap CIs) for place/pressure/width/pose/self/upcoming action/next contact
2 pressure sweep + text pressure direction of plain Qwen   3 steering + subspace ablation in closed loop   4 raw-electrode baselines
5 anticipation (next-step contact)   6 per-frame arrays for the video."""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
import sim3 as sim, v3
OUT, N, NCL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); ARMK = sys.argv[4] if len(sys.argv) > 4 else "named"
os.makedirs(OUT, exist_ok=True); T = v3.T; DEV = v3.DEV; t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
def touch(): open(f"{OUT}/BUSY", "w").write(str(time.time()))
B = v3.Body("Qwen/Qwen3-8B", ARMK == "named", torch.bfloat16)
ck = torch.load(f"ckpt_Qwen3-8B_{ARMK}_s0.pt", map_location=DEV, weights_only=False)
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"]); B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()
NL = B.nL + 1; D = B.d; LM = B.m.get_base_model().lm_head; ACT = torch.tensor(v3.ACT_TOK, device=DEV)
R = {"meta2": dict(arm=ARMK, N=N, NCL=NCL, layers=NL, L_IN=B.L_IN)}
def save(): json.dump(R, open(f"{OUT}/inn2.json", "w"), indent=1)

# ---------- hooks: steering / ablation at layer L output (hidden_states[L]) ----------
ACTIVE, MASK = {}, {}
def mk_hook(L):
    def hk(mod, inp, out):
        if L not in ACTIVE: return out
        h = out[0] if isinstance(out, tuple) else out; h = ACTIVE[L](h)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hk
for L in range(1, NL): B.layers[L - 1].register_forward_hook(mk_hook(L))

KINDS = ["babble", "reach", "find", "wall"]; LMS = list(sim.LANDMARKS)
def primary(C): return max(C, key=lambda c: c["p"] * (1.01 if c["seg"] % 3 != 2 else 1)) if C else None
def goal_vec(ep):
    v = np.zeros(10, np.float32)
    if ep["kind"] != "babble":
        v[KINDS.index(ep["kind"])] = 1
        if ep["kind"] == "reach": v[4 + LMS.index(ep["g"]["lm"])] = 1; v[8 + ep["g"]["arm"]] = 1
    return v
def label_ep(ep):
    e0 = ep["env"]; env = sim.Env(ep["S0q"], np.random.default_rng(0), ext=e0.ext, wall=e0.wall, obj=e0.obj); rows = []
    for t in range(T):
        C = env.contacts(); c = primary(C); r = dict(q=env.q.copy(), contact=bool(C), src=None)
        if c: r.update(arm=c["seg"] // 3, arc=sim.arc_of(c["seg"], c["u"]), w=c["w"], p=c["p"], width=c["width"], src=c["src"])
        rows.append(r)
        if t < T - 1: env.step(ep["A"][t])
    assert np.allclose(env.q, ep["info"]["q"]), "replay mismatch"
    return rows
def closed(kind, n, seed, setup=None):
    rng = np.random.default_rng(seed); probes = []
    for _ in range(n):
        r2_ = np.random.default_rng(int(rng.integers(1 << 30))); probes.append(sim.make_episode(np.random.default_rng(r2_.integers(1 << 30)), T, kind))
    envs = [sim.Env(p["S0q"], np.random.default_rng(0), wall=p["env"].wall, obj=p["env"].obj) for p in probes]
    Ss = [[e.sense()] for e in envs]; As = [[] for _ in envs]
    for t in range(T - 1):
        eps = [dict(kind=kind, g=p["g"], S=np.stack(s), A=a) for p, s, a in zip(probes, Ss, As)]
        items = [B.prompt(e, "", None, False) for e in eps]; ids, lab, kd, att, X = B.collate(items, eps)
        MASK["skin"] = (kd == 2); MASK["body"] = (torch.cumsum((kd > 0).long(), 1) > 0) & (att > 0)
        if setup: setup()
        with torch.no_grad():
            h = B.run(ids, kd, att, X).last_hidden_state; kp = torch.tensor([it[4][-1] for it in items], device=DEV)
            a_ = LM(h[torch.arange(len(eps), device=DEV), kp])[:, ACT].argmax(-1).tolist()
        ACTIVE.clear()
        for e, s, a, ai in zip(envs, Ss, As, a_): a.append(ai); e.step(ai); s.append(e.sense())
    res = []
    for p, e, a in zip(probes, envs, As):
        info = sim.summarize(e); C = e.contacts(); d = dict(success=bool(sim.success(kind, info)), A=a, push=e.push)
        d["wall_p"] = max([c["p"] for c in C if c["src"] == "wall"], default=0.)
        if kind == "reach":
            segs = sim.seg_pts(e.q); tip = segs[3 * p["g"]["arm"] + 2][1]; best = None
            for i in range(3 * p["g"]["tarm"], 3 * p["g"]["tarm"] + 3):
                u, dd, w = sim.closest(tip, *segs[i]); best = min(best or (9, 0), (dd, sim.arc_of(i, u)))
            d["reach_arc"] = best[1]; d["target_arc"] = sim.LANDMARKS[p["g"]["lm"]]; d["tip_dist"] = best[0]
        res.append(d)
    return res, probes, Ss, As
def steer(L, v):
    vt = torch.tensor(v, device=DEV, dtype=torch.bfloat16)
    def s(): ACTIVE[L] = lambda h: h + MASK["skin"][..., None].to(h.dtype) * vt
    return s
def ablate(Us, mus):
    Ut = {L: torch.tensor(U, device=DEV, dtype=torch.float32) for L, U in Us.items()}; Mt = {L: torch.tensor(mus[L], device=DEV, dtype=torch.float32) for L in Us}
    def s():
        for L in Ut:
            def f(h, L=L):
                hf = h.float(); c = ((hf - Mt[L]) @ Ut[L].T) @ Ut[L]; return (hf - MASK["body"][..., None].float() * c).to(h.dtype)
            ACTIVE[L] = f
    return s
def mean_ci(v, nb=1000):
    v = np.asarray(v, float); rs = np.random.default_rng(0); b = [v[rs.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
def proj_sd(L, d, rows): return float(np.std(Hk[L][rows].astype(np.float32) @ d))
def static_ep(q, arm, arc, p, r=.1):
    A_ = sim.landmark_point(q, arm, arc); env = sim.Env(q, np.random.default_rng(0)); P_, Qp = sim.seg_pts(q)[3 * arm + sim.seg_of(arc)]
    nrm = np.array([-(Qp - P_)[1], (Qp - P_)[0]]); nrm /= np.linalg.norm(nrm)
    env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=r, p=p, t_end=T - 1); S_ = [env.sense()]
    for t in range(T - 1): env.step(12); S_.append(env.sense())
    return dict(S=np.stack(S_), A=[12] * (T - 1), kind="babble", g={})
PRESS = ["A feather barely brushes", "A fingertip grazes, so lightly you almost miss it,", "A fingertip rests lightly on", "A fingertip taps lightly on",
         "A fingertip presses gently on", "A palm rests with its full weight on", "A thumb presses moderately into", "A thumb presses firmly into",
         "A hand squeezes firmly around", "A hand grips tightly around", "A hand grips very hard around", "A hand squeezes, as hard as it can without hurting,"]
PLACES = ["your left forearm", "your right upper arm", "your left calf"]
stims = [(r, f"{ph} {pl}.") for pl in PLACES for r, ph in enumerate(PRESS)]
def text_states(adapter_on):
    H = np.zeros((NL, len(stims), D), np.float32)
    for i, (r, s) in enumerate(stims):
        ids = torch.tensor([B.tok.encode(s)], device=DEV)
        with torch.no_grad():
            if adapter_on: out = B.m.get_base_model().model(input_ids=ids, output_hidden_states=True)
            else:
                with B.m.disable_adapter(): out = B.m.get_base_model().model(input_ids=ids, output_hidden_states=True)
        for L in range(NL): H[L, i] = out.hidden_states[L][0, -1].float().cpu().numpy()
    return H

def unit(v): return v / (np.linalg.norm(v) + 1e-9)
PR = np.load("probes.npz"); dirs = {k[4:]: PR[k] for k in PR.files if k.startswith("dir_")}; PMU = PR["pca_mu"]
LAYS = [8, 12, 18, 24, 30]
rng = np.random.default_rng(1234); EPS = [sim.make_episode(rng, T) for _ in range(N)]; LAB = [label_ep(e) for e in EPS]
CON = np.array([r["contact"] for L_ in LAB for r in L_]); SRC = np.array([r["src"] or "" for L_ in LAB for r in L_]); P = np.array([r.get("p", np.nan) for L_ in LAB for r in L_], float)
Hk = {L: np.zeros((N * T, D), np.float32) for L in LAYS}
with torch.no_grad():
    for b in range(0, N, 24):
        ee = EPS[b:b + 24]; items = [B.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = B.collate(items, ee)
        out = B.run(ids, kd, att, X, hidden=True); r_ = torch.arange(len(ee), device=DEV)[:, None]; kp = torch.tensor([it[4] for it in items], device=DEV)
        for L in LAYS: Hk[L][b * T:(b + len(ee)) * T] = out.hidden_states[L][r_, kp].reshape(-1, D).float().cpu().numpy()
w_rows = np.where(CON & (SRC == "wall"))[0]; log("states", len(w_rows))
def mean_ci(v, nb=1000):
    v = np.asarray(v, float); rs = np.random.default_rng(0); b = [v[rs.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
def steer(L, v):
    vt = torch.tensor(v, device=DEV, dtype=torch.bfloat16)
    def s(): ACTIVE[L] = lambda h: h + MASK["skin"][..., None].to(h.dtype) * vt
    return s
def ablate(Us):
    Ut = {L: torch.tensor(U, device=DEV, dtype=torch.float32) for L, U in Us.items()}; Mt = {L: torch.tensor(PMU[L], device=DEV, dtype=torch.float32) for L in Us}
    def s():
        for L in Ut:
            def f(h, L=L):
                hf = h.float(); c = ((hf - Mt[L]) @ Ut[L].T) @ Ut[L]; return (hf - MASK["body"][..., None].float() * c).to(h.dtype)
            ACTIVE[L] = f
    return s
# --- A: pressure steering at several layers (felt-pressure direction from wall contacts) ---
R["steer_layers"] = {}
for L in LAYS:
    d = unit(dirs["press_wall"][L]); sd = float(np.std(Hk[L][w_rows] @ d)); R["steer_layers"][str(L)] = {"sd": sd, "dnorm_over_resid": float(np.linalg.norm(dirs["press_wall"][L]) / np.linalg.norm(Hk[L][w_rows], axis=1).mean())}
    for al in (-6, 0, 6):
        if al == 0 and L != 8: R["steer_layers"][str(L)]["0"] = R["steer_layers"]["8"]["0"]; continue
        res = closed("wall", NCL, 9, steer(L, al * sd * d) if al else None)[0]
        R["steer_layers"][str(L)][str(al)] = dict(wall_p=mean_ci([r["wall_p"] for r in res]), push=mean_ci([r["push"] for r in res]), success=mean_ci([r["success"] for r in res]))
    rr = unit(np.random.default_rng(L).standard_normal(D))
    for al in (-6, 6):
        res = closed("wall", NCL, 9, steer(L, al * sd * rr))[0]; R["steer_layers"][str(L)][f"rand{al}"] = dict(wall_p=mean_ci([r["wall_p"] for r in res]), push=mean_ci([r["push"] for r in res]))
    log("steer", L, {k: v["wall_p"][0] if isinstance(v, dict) else v for k, v in R["steer_layers"][str(L)].items()}); save(); touch()
# --- B: single-layer ablations with variance-matched random subspaces ---
GROUPS = {"map8": ("arc0", "arc1", "arm", "side", "press", "width", "self", "contact"), "place4": ("arc0", "arc1", "arm", "side"), "press2": ("press", "press_wall")}
def map_U(L, grp): M = np.stack([dirs[k][L] for k in GROUPS[grp]]); Qm, _ = np.linalg.qr(M.T); return Qm.T
VT = {}
def var_removed(L, U): X = Hk[L][::3] - Hk[L][::3].mean(0); return float(((X @ U.T) ** 2).sum() / (X ** 2).sum())
def rand_U(L, k, target, seed):
    if L not in VT: X = Hk[L][::2] - Hk[L][::2].mean(0); VT[L] = np.linalg.svd(X[:4000], full_matrices=False)[2][:512]
    best = None; rs = np.random.default_rng(seed)
    for m in list(range(k, 64)) + list(range(64, 513, 8)):
        Qm, _ = np.linalg.qr(VT[L][:m].T @ rs.standard_normal((m, k))); U = Qm.T; v = var_removed(L, U)
        if best is None or abs(v - target) < abs(best[1] - target): best = (U, v, m)
        if abs(v - target) < .01 * target: break
    return best
R["ablate1"] = {}
def run_cond(name, Us, extra):
    R["ablate1"][name] = dict(extra)
    for kd_, sd_ in (("reach", 7), ("find", 8), ("wall", 9)):
        res = closed(kd_, NCL, sd_, ablate(Us) if Us else None)[0]; R["ablate1"][name][kd_] = mean_ci([r["success"] for r in res])
        if kd_ == "wall": R["ablate1"][name]["wall_p"] = mean_ci([r["wall_p"] for r in res])
    log("ablate", name, R["ablate1"][name]); save(); touch()
run_cond("none", None, {})
for L in (12, 18):
    for grp in GROUPS:
        U = map_U(L, grp); v = var_removed(L, U); run_cond(f"{grp}_L{L}", {L: U}, {"var": v})
        for sdr in range(3):
            Ur, vr, m = rand_U(L, len(GROUPS[grp]), v, 100 + sdr); run_cond(f"rand{sdr}_{grp}_L{L}", {L: Ur}, {"var": vr, "m": m})
# --- C: pressure sweep redone (nan-safe, clipping) + text null from shuffled ranks ---
PS = np.linspace(.1, 1., 12); rs = np.random.default_rng(42); POS = [sim.rand_q(rs) for _ in range(6)]; ARCS = [.15, .4, .65, .85, 1.05]
sw = [(pi, ai, p) for pi in range(6) for ai in range(5) for p in PS]; sweps = [static_ep(POS[pi], 1, ARCS[ai], p) for pi, ai, p in sw]
NL_ = NL; HS = np.zeros((NL_, len(sw), D), np.float32); clip = []
WP = PR["w_pressure"]; BP = PR["b_pressure"]
with torch.no_grad():
    for k in range(0, len(sweps), 24):
        ee = sweps[k:k + 24]; items = [B.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = B.collate(items, ee)
        out = B.run(ids, kd, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=DEV); r_ = torch.arange(len(ee), device=DEV)
        for L in range(NL_): HS[L, k:k + len(ee)] = out.hidden_states[L][r_, kp].float().cpu().numpy()
        clip += (B.ad.k(X[r_, kp][:, 22:]).norm(dim=-1) > B.ad.cap).tolist()
swp = np.array([p for _, _, p in sw]); grp = np.array([pi * 5 + ai for pi, ai, _ in sw]); clip = np.array(clip)
R["sweep2"] = {"clipped_frac": float(clip.mean()), "groups_all_clipped": int(sum(clip[grp == g].all() for g in range(30))), "layers": []}
PROJ = np.zeros((NL_, len(sw)), np.float32)
for L in range(NL_):
    proj = HS[L] @ WP[L][:, 0] + BP[L][0]; PROJ[L] = proj; rho, lin, step = [], [], []
    for gi in range(30):
        m = (grp == gi) & ~clip; y = proj[m]; x = swp[m]
        if m.sum() < 4 or np.std(y) == 0: continue
        rho.append(spearmanr(x, y)[0]); lin.append(1 - np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2) / np.sum((y - y.mean()) ** 2))
        step.append(max(1 - (np.sum((y[:j] - y[:j].mean()) ** 2) + np.sum((y[j:] - y[j:].mean()) ** 2)) / np.sum((y - y.mean()) ** 2) for j in range(1, len(y))))
    R["sweep2"]["layers"].append(dict(L=L, n_groups=len(rho), rho=float(np.mean(rho)) if rho else None, lin_r2=float(np.mean(lin)) if lin else None, step_r2=float(np.mean(step)) if step else None))
np.savez(f"{OUT}/sweep2.npz", proj=PROJ, p=swp, grp=grp, clipped=clip)
rk = np.array([r for r, _ in stims]); HT = np.zeros((NL_, len(stims), D), np.float32)
for i, (r, s_) in enumerate(stims):
    with torch.no_grad(), B.m.disable_adapter(): out = B.m.get_base_model().model(input_ids=torch.tensor([B.tok.encode(s_)], device=DEV), output_hidden_states=True)
    for L in range(NL_): HT[L, i] = out.hidden_states[L][0, -1].float().cpu().numpy()
R["text_null"] = []
for L in range(NL_):
    db = unit(dirs["press"][L]); dbw = unit(dirs["press_wall"][L]); dt = unit(HT[L][rk >= 9].mean(0) - HT[L][rk <= 2].mean(0)); rs = np.random.default_rng(L); nul = []
    for _ in range(300):
        pr = rs.permutation(rk); nul.append(unit(HT[L][pr >= 9].mean(0) - HT[L][pr <= 2].mean(0)) @ db)
    R["text_null"].append(dict(L=L, cos=float(dt @ db), cos_wall=float(dt @ dbw), null_lo=float(np.percentile(nul, 2.5)), null_hi=float(np.percentile(nul, 97.5))))
save(); log("done")
