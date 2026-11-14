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
R = {"meta": dict(arm=ARMK, N=N, NCL=NCL, layers=NL, L_IN=B.L_IN)}
def save(): json.dump(R, open(f"{OUT}/inn.json", "w"), indent=1)

# ---------- hooks: steering / ablation at layer L output (hidden_states[L]) ----------
ACTIVE, MASK = {}, {}
def mk_hook(L):
    def hk(mod, inp, out):
        if L not in ACTIVE: return out
        h = out[0] if isinstance(out, tuple) else out; h = ACTIVE[L](h)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hk
for L in range(1, NL): B.layers[L - 1].register_forward_hook(mk_hook(L))

# ---------- dataset ----------
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
rng = np.random.default_rng(1234); EPS = [sim.make_episode(rng, T) for _ in range(N)]
LAB = [label_ep(e) for e in EPS]; log("episodes", N, {k: sum(e["kind"] == k for e in EPS) for k in KINDS})
# per skin-row arrays (row = ep*T + t)
nr = N * T; ep_id = np.repeat(np.arange(N), T); tt = np.tile(np.arange(T), N)
kind = np.array([KINDS.index(e["kind"]) for e in EPS])[ep_id]
Q = np.stack([r["q"] for L_ in LAB for r in L_]); CON = np.array([r["contact"] for L_ in LAB for r in L_])
g = lambda k, d=np.nan: np.array([r.get(k, d) if r.get(k) is not None else d for L_ in LAB for r in L_], float)
ARM, ARC, SIDE, P, WID = g("arm"), g("arc"), g("w"), g("p"), g("width")
SRC = np.array([r["src"] or "" for L_ in LAB for r in L_])
S = np.stack([e["S"] for e in EPS]).reshape(nr, -1)  # [nr, 38]
A = np.full(nr, -1); NEXTC = np.full(nr, -1)
for i, e in enumerate(EPS):
    A[i * T:i * T + T - 1] = e["A"]; NEXTC[i * T:i * T + T - 1] = CON[i * T + 1:i * T + T]
# features for raw baselines
GOAL = np.stack([goal_vec(e) for e in EPS])[ep_id]; TOH = np.eye(T)[tt]
Sprev = np.where((tt > 0)[:, None], np.roll(S, 1, 0), 0); Aprev = np.where((tt > 0)[:, None], np.eye(16)[np.roll(np.maximum(A, 0), 1)], 0)
F2 = np.concatenate([S, Sprev, Aprev, GOAL, TOH], 1).astype(np.float32)
FH = np.zeros((nr, T * 38 + (T - 1) * 16), np.float32)
for r in range(nr):
    e = EPS[ep_id[r]]; t = tt[r]; FH[r, :(t + 1) * 38] = e["S"][:t + 1].ravel()
    for j in range(t): FH[r, T * 38 + j * 16 + e["A"][j]] = 1
FH = np.concatenate([FH, GOAL, TOH], 1)
armi = np.nan_to_num(ARM).astype(int); F8 = np.stack([S[r, 22 + 8 * armi[r]:30 + 8 * armi[r]] for r in range(nr)]).astype(np.float32)
F16 = S[:, 22:].astype(np.float32)

# ---------- forward: all-layer states at every skin token and every motor token ----------
Hk = np.zeros((NL, nr, D), np.float16); Ha = np.zeros((NL, N * (T - 1), D), np.float16); PRED = np.zeros(nr, int); clipfrac = []
with torch.no_grad():
    for b in range(0, N, 24):
        ee = EPS[b:b + 24]; items = [B.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = B.collate(items, ee)
        out = B.run(ids, kd, att, X, hidden=True); r_ = torch.arange(len(ee), device=DEV)[:, None]
        kp = torch.tensor([it[4] for it in items], device=DEV); ap = torch.tensor([it[3] for it in items], device=DEV)
        for L in range(NL):
            hs = out.hidden_states[L]
            Hk[L, b * T:(b + len(ee)) * T] = hs[r_, kp].reshape(-1, D).float().clamp(-6e4, 6e4).half().cpu().numpy()
            Ha[L, b * (T - 1):(b + len(ee)) * (T - 1)] = hs[r_, ap].reshape(-1, D).float().clamp(-6e4, 6e4).half().cpu().numpy()
        PRED[b * T:(b + len(ee)) * T] = LM(out.hidden_states[-1][r_, kp])[..., ACT].argmax(-1).reshape(-1).cpu().numpy()
        vk = B.ad.k(X[..., 22:]).norm(dim=-1)[kd == 2]; clipfrac.append(float((vk > B.ad.cap).float().mean()))
        if b % 240 == 0: log("fwd", b); touch()
R["clip_frac_skin_vectors"] = float(np.mean(clipfrac)); R["resid_norm_by_layer"] = [float(np.linalg.norm(Hk[L, :2000].astype(np.float32), axis=1).mean()) for L in range(NL)]
goal_rows = (kind > 0) & (A >= 0); R["model_action_acc_on_expert_rows"] = float((PRED[goal_rows] == A[goal_rows]).mean())
log("forward done; clip", R["clip_frac_skin_vectors"])

# ---------- ridge (GPU) ----------
FOLD = ep_id % 5; LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
def ridge_fit(Xt, Yt, lam=None):
    mu = Xt.mean(0); sd = Xt.std(0) + 1e-3; Z = (Xt - mu) / sd; ym = Yt.mean(0)
    G = (Z.T @ Z).double(); ZY = (Z.T @ (Yt - ym)).double(); n = Xt.shape[0]; I = torch.eye(G.shape[0], device=DEV, dtype=torch.float64)
    lams = LAMS if lam is None else sorted(set(lam))
    W = {l: torch.linalg.solve(G + l * n * I, ZY).float() for l in lams}
    return mu, sd, ym, W
def ridge_pred(m, Xv, lam_cols):
    mu, sd, ym, W = m; Z = (Xv - mu) / sd
    return torch.stack([(Z @ W[l][:, j]) + ym[j] for j, l in enumerate(lam_cols)], 1)
def ridge_cv(X, Y, rows):
    """out-of-fold predictions, inner split picks lambda per column. X torch [nr,D] (GPU), Y np [n_rows_sel, k]."""
    Yt = torch.tensor(Y, device=DEV, dtype=torch.float32); pred = torch.zeros_like(Yt); ri = torch.tensor(rows, device=DEV); fo = FOLD[rows]; eid = ep_id[rows]
    for f in range(5):
        tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]; inner = (eid[tr] // 5) % 5 == 0
        m = ridge_fit(X[ri[tr[~inner]]], Yt[tr[~inner]])
        err = torch.stack([((ridge_pred(m, X[ri[tr[inner]]], [l] * Y.shape[1]) - Yt[tr[inner]]) ** 2).mean(0) for l in LAMS])
        lc = [LAMS[i] for i in err.argmin(0).tolist()]
        m = ridge_fit(X[ri[tr]], Yt[tr], lc); pred[te] = ridge_pred(m, X[ri[te]], lc)
    return pred.cpu().numpy()
def ridge_all(X, Y, rows, lam=1e-2):
    m = ridge_fit(X[torch.tensor(rows, device=DEV)], torch.tensor(Y, device=DEV, dtype=torch.float32), [lam]); mu, sd, ym, W = m
    w = (W[lam] / sd[:, None]); b = ym - mu @ w; return w.cpu().numpy(), b.cpu().numpy()
def mlp_cv(Xn, Y, rows, cls=False):
    """2x256 MLP on raw features, early-stopped on an inner split; out-of-fold predictions."""
    X = torch.tensor(Xn[rows], device=DEV, dtype=torch.float32); Yt = torch.tensor(Y, device=DEV, dtype=torch.float32); pred = torch.zeros_like(Yt); fo = FOLD[rows]; eid = ep_id[rows]
    for f in range(5):
        tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]; inner = (eid[tr] // 5) % 5 == 0; a, v = tr[~inner], tr[inner]
        mu, sd = X[a].mean(0), X[a].std(0); sd = torch.where(sd < 1e-4, torch.ones_like(sd), sd); ym, ys = Yt[a].mean(0), Yt[a].std(0) + 1e-6
        torch.manual_seed(f); net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3); best, bst = 1e9, None
        for it in range(1500):
            idx = torch.randint(0, len(a), (512,), device=DEV); loss = ((net((X[a][idx] - mu) / sd) - (Yt[a][idx] - ym) / ys) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 49:
                with torch.no_grad(): vl = ((net((X[v] - mu) / sd) - (Yt[v] - ym) / ys) ** 2).mean().item()
                if vl < best: best, bst = vl, [p.detach().clone() for p in net.parameters()]
        for p, q in zip(net.parameters(), bst or []): p.data.copy_(q)
        with torch.no_grad(): pred[te] = net((X[te] - mu) / sd) * ys + ym
    return pred.cpu().numpy()

# ---------- metrics + episode bootstrap ----------
def r2(y, p): y = y.reshape(len(y), -1); p = p.reshape(len(p), -1); return float(np.mean(1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))
def auc(y, s):
    y = y.astype(bool); o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
def acc(y, p): return float((p.argmax(1) == y).mean())
def boot(fn, y, p, eids, nb=200):
    u = np.unique(eids); pos = {e: np.where(eids == e)[0] for e in u}; rs = np.random.default_rng(0); v = []
    for _ in range(nb):
        idx = np.concatenate([pos[e] for e in rs.choice(u, len(u))]); v.append(fn(y[idx], p[idx]))
    return [round(fn(y, p), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

# targets: name -> (rows, Y, metric, ytrue for metric)
c_rows = np.where(CON)[0]; so_rows = np.where(CON & np.isin(SRC, ["self", "other"]))[0]; act_rows = np.where((kind > 0) & (A >= 0))[0]
nc_rows = np.where((~CON) & (NEXTC >= 0))[0]; w_rows = np.where(CON & (SRC == "wall"))[0]
TGT = {
    "place_arc": (c_rows, ARC[c_rows, None], r2, ARC[c_rows]),
    "place_arm": (c_rows, np.eye(2)[ARM[c_rows].astype(int)], acc, ARM[c_rows].astype(int)),
    "pressure": (c_rows, P[c_rows, None], r2, P[c_rows]),
    "width": (c_rows, WID[c_rows, None], r2, WID[c_rows]),
    "pose": (np.arange(nr), Q, r2, Q),
    "self_vs_other": (so_rows, (SRC[so_rows] == "self")[:, None].astype(float), auc, (SRC[so_rows] == "self")),
    "contact_now": (np.arange(nr), CON[:, None].astype(float), auc, CON),
    "next_action": (act_rows, np.eye(16)[A[act_rows]], acc, A[act_rows]),
    "contact_next_from_none": (nc_rows, NEXTC[nc_rows, None].astype(float), auc, NEXTC[nc_rows].astype(bool)),
    "wall_pressure": (w_rows, P[w_rows, None], r2, P[w_rows]),
}
def score(name, pred):
    rows, Y, fn, yt = TGT[name]; p = pred if fn is not auc else pred[:, 0]; yy = yt if fn is not r2 else Y
    return boot(fn, yy, p, ep_id[rows])
R["n_rows"] = {k: int(len(v[0])) for k, v in TGT.items()}
R["chance"] = dict(place_arm=float(np.bincount(ARM[c_rows].astype(int)).max() / len(c_rows)), next_action=float(np.bincount(A[act_rows]).max() / len(act_rows)), auc=0.5)
log("rows", R["n_rows"])

# raw-input baselines (same folds)
R["base"] = {}
for name, (rows, Y, fn, yt) in (TGT.items() if not os.environ.get("SKIPBASE") else []):
    R["base"][name] = {}
    Xl = torch.tensor(F2, device=DEV); R["base"][name]["lin_last2"] = score(name, ridge_cv(Xl, Y, rows))
    R["base"][name]["mlp_history"] = score(name, mlp_cv(FH, Y, rows))
    if name in ("place_arc", "pressure", "width", "place_arm", "wall_pressure"):
        for fk, F in (("8", F8), ("16", F16)):
            R["base"][name][f"lin_{fk}"] = score(name, ridge_cv(torch.tensor(F, device=DEV), Y, rows))
            R["base"][name][f"mlp_{fk}"] = score(name, mlp_cv(F, Y, rows))
    log("base", name, R["base"][name]); touch()
save()

# layer curves (skin token) + anticipation at the motor token
R["curve"] = {k: [] for k in TGT}; R["curve"]["contact_next_from_none@motor"] = []
nc_a = np.array([ep_id[r] * (T - 1) + tt[r] for r in nc_rows])  # motor token after skin token t
WTS = {}; PCA = np.zeros((NL, 3, D), np.float32); PMU = np.zeros((NL, D), np.float32)
dirs = {k: np.zeros((NL, D), np.float32) for k in ("press", "press_wall", "arc0", "arc1", "arm", "side", "width", "self", "contact")}
def dm(X, rows, v):
    lo, hi = np.nanpercentile(v, [33.3, 66.7]); a = rows[v <= lo]; b = rows[v >= hi]; return (X[b].mean(0) - X[a].mean(0))
def pca_red(X, k=1024):
    Xc = X - X.mean(0); _, _, Vh = torch.linalg.svd(Xc[::max(1, len(Xc) // 8000)], full_matrices=False); return Xc @ Vh[:k].T
for L in range(NL):
    X = torch.tensor(Hk[L], device=DEV, dtype=torch.float32); Xr = pca_red(X)
    for name, (rows, Y, fn, yt) in TGT.items(): R["curve"][name].append(score(name, ridge_cv(Xr, Y, rows)))
    del Xr; Xa = pca_red(torch.tensor(Ha[L], device=DEV, dtype=torch.float32))
    p = ridge_cv(Xa, NEXTC[nc_rows, None].astype(float), nc_a); R["curve"]["contact_next_from_none@motor"].append(boot(auc, NEXTC[nc_rows].astype(bool), p[:, 0], ep_id[nc_rows]))
    del Xa
    # probes on all data (for the video readouts), PCA, mean-difference directions
    for name in ("place_arc", "pressure", "width", "self_vs_other", "contact_now", "next_action", "pose", "contact_next_from_none"):
        rows, Y, _, _ = TGT[name]; w, b = ridge_all(X, Y, rows); WTS.setdefault(name, [np.zeros((NL, D, Y.shape[1]), np.float32), np.zeros((NL, Y.shape[1]), np.float32)])
        WTS[name][0][L] = w; WTS[name][1][L] = b
    mu = X.mean(0); U, Sv, V = torch.pca_lowrank(X - mu, q=8, center=False); PCA[L] = V[:, :3].T.cpu().numpy(); PMU[L] = mu.cpu().numpy()
    Xc = X.cpu().numpy()
    dirs["press"][L] = dm(Xc, c_rows, P[c_rows]); dirs["press_wall"][L] = dm(Xc, w_rows, P[w_rows]); dirs["width"][L] = dm(Xc, c_rows, WID[c_rows])
    for a in (0, 1): ra = c_rows[ARM[c_rows] == a]; dirs[f"arc{a}"][L] = dm(Xc, ra, ARC[ra])
    dirs["arm"][L] = Xc[c_rows[ARM[c_rows] == 1]].mean(0) - Xc[c_rows[ARM[c_rows] == 0]].mean(0)
    dirs["side"][L] = Xc[c_rows[SIDE[c_rows] > .5]].mean(0) - Xc[c_rows[SIDE[c_rows] <= .5]].mean(0)
    dirs["self"][L] = Xc[so_rows[SRC[so_rows] == "self"]].mean(0) - Xc[so_rows[SRC[so_rows] == "other"]].mean(0)
    dirs["contact"][L] = Xc[CON].mean(0) - Xc[~CON].mean(0)
    del X, Xc; torch.cuda.empty_cache(); touch()
    log("layer", L, {k: R["curve"][k][-1][0] for k in ("place_arc", "pressure", "width", "pose", "self_vs_other", "next_action", "contact_next_from_none")})
    save()
np.savez(f"{OUT}/probes.npz", **{f"w_{k}": v[0] for k, v in WTS.items()}, **{f"b_{k}": v[1] for k, v in WTS.items()}, pca=PCA, pca_mu=PMU, **{f"dir_{k}": v for k, v in dirs.items()})

# ---------- 2 pressure: sweep at fixed place, and plain Qwen's text pressure direction ----------
def static_ep(q, arm, arc, p, r=.1):
    A_ = sim.landmark_point(q, arm, arc); env = sim.Env(q, np.random.default_rng(0)); P_, Qp = sim.seg_pts(q)[3 * arm + sim.seg_of(arc)]
    nrm = np.array([-(Qp - P_)[1], (Qp - P_)[0]]); nrm /= np.linalg.norm(nrm)
    env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=r, p=p, t_end=T - 1); S_ = [env.sense()]
    for t in range(T - 1): env.step(12); S_.append(env.sense())
    return dict(S=np.stack(S_), A=[12] * (T - 1), kind="babble", g={})
PS = np.linspace(.1, 1., 12); rs = np.random.default_rng(42); POS = [sim.rand_q(rs) for _ in range(6)]; ARCS = [.15, .4, .65, .85, 1.05]
sw = [(pi, ai, p) for pi in range(6) for ai in range(5) for p in PS]; sweps = [static_ep(POS[pi], 1, ARCS[ai], p) for pi, ai, p in sw]
HS = np.zeros((NL, len(sw), D), np.float32)
with torch.no_grad():
    for k in range(0, len(sweps), 24):
        ee = sweps[k:k + 24]; items = [B.prompt(e, "", None, False) for e in ee]; ids, lab, kd, att, X = B.collate(items, ee)
        out = B.run(ids, kd, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=DEV); r_ = torch.arange(len(ee), device=DEV)
        for L in range(NL): HS[L, k:k + len(ee)] = out.hidden_states[L][r_, kp].float().cpu().numpy()
swp = np.array([p for _, _, p in sw]); grp = np.array([pi * 5 + ai for pi, ai, _ in sw])
R["sweep"] = {"layers": []}
for L in range(NL):
    w = WTS["pressure"][0][L][:, 0]; proj = HS[L] @ w + WTS["pressure"][1][L][0]
    rho, lin, step, pc1 = [], [], [], []
    for gi in range(30):
        m = grp == gi; y = proj[m]; rho.append(spearmanr(PS, y)[0])
        lin.append(1 - np.sum((y - np.polyval(np.polyfit(PS, y, 1), PS)) ** 2) / np.sum((y - y.mean()) ** 2))
        step.append(max(1 - (np.sum((y[:j] - y[:j].mean()) ** 2) + np.sum((y[j:] - y[j:].mean()) ** 2)) / np.sum((y - y.mean()) ** 2) for j in range(1, 12)))
        Z = HS[L][m] - HS[L][m].mean(0); s = np.linalg.svd(Z, compute_uv=False); pc1.append(s[0] ** 2 / (s ** 2).sum())
    R["sweep"]["layers"].append(dict(L=L, rho=float(np.mean(rho)), lin_r2=float(np.mean(lin)), step_r2=float(np.mean(step)), traj_pc1=float(np.mean(pc1)),
                                     probe_mean_by_p=[float(proj[swp == p].mean()) for p in PS]))
np.savez(f"{OUT}/sweep.npz", proj18=HS[18] @ WTS["pressure"][0][18][:, 0] + WTS["pressure"][1][18][0], p=swp, grp=grp)
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
rk = np.array([r for r, _ in stims]); HT0 = text_states(False); HT1 = text_states(True)
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
rr = np.random.default_rng(7); RND = [unit(rr.standard_normal(D)) for _ in range(200)]
R["text_pressure"] = []
for L in range(NL):
    dt0 = unit(HT0[L][rk >= 9].mean(0) - HT0[L][rk <= 2].mean(0)); dt1 = unit(HT1[L][rk >= 9].mean(0) - HT1[L][rk <= 2].mean(0))
    db = unit(dirs["press"][L]); dbw = unit(dirs["press_wall"][L]); wpr = unit(WTS["pressure"][0][L][:, 0])
    X = Hk[L][c_rows].astype(np.float32); y = P[c_rows]
    pr_text = X @ dt0; r_text = float(np.corrcoef(pr_text, y)[0, 1])
    rnd_r = [abs(np.corrcoef(X @ v, y)[0, 1]) for v in RND[:50]]
    rho_rev = float(spearmanr(rk, HT0[L] @ db)[0])  # text stimuli on the body direction
    R["text_pressure"].append(dict(L=L, cos_text_base_vs_body=float(dt0 @ db), cos_text_base_vs_body_wall=float(dt0 @ dbw), cos_text_base_vs_probe=float(dt0 @ wpr),
        cos_text_lora_vs_text_base=float(dt0 @ dt1), cos_text_lora_vs_body=float(dt1 @ db), cos_random_abs95=float(np.percentile([abs(v @ db) for v in RND], 95)),
        body_states_on_text_dir_r=r_text, body_states_on_random_dirs_absr95=float(np.percentile(rnd_r, 95)), text_on_body_dir_rho=rho_rev,
        text_on_text_dir_loo_rho=float(np.mean([spearmanr(rk[np.arange(36) // 12 == pl], HT0[L][np.arange(36) // 12 == pl] @ unit(HT0[L][(rk >= 9) & (np.arange(36) // 12 != pl)].mean(0) - HT0[L][(rk <= 2) & (np.arange(36) // 12 != pl)].mean(0)))[0] for pl in range(3)]))))
save(); log("pressure done", R["text_pressure"][18]); touch()

# ---------- 3 causal: closed loop with steering / ablation ----------
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
R["causal"] = {}; LS = 18
dpw = unit(dirs["press_wall"][LS]); sp = proj_sd(LS, dpw, w_rows); rrand = unit(np.random.default_rng(3).standard_normal(D))
R["causal"]["wall_steer"] = {"sd": sp}
for al in (-6, -3, 0, 3, 6):
    res = closed("wall", NCL, 9, steer(LS, al * sp * dpw) if al else None)[0]
    R["causal"]["wall_steer"][str(al)] = dict(wall_p=mean_ci([r["wall_p"] for r in res]), success=mean_ci([r["success"] for r in res]),
                                               push_count=mean_ci([r["push"] for r in res]))
    log("wall steer", al, R["causal"]["wall_steer"][str(al)]); touch()
for al in (-6, 6):
    res = closed("wall", NCL, 9, steer(LS, al * sp * rrand))[0]; R["causal"]["wall_steer"][f"rand{al}"] = dict(wall_p=mean_ci([r["wall_p"] for r in res]), success=mean_ci([r["success"] for r in res]))
# place: which arm is the target arm varies; steer the target arm's arc direction
R["causal"]["reach_steer"] = {}
for al in (-6, -3, 0, 3, 6, "rand6"):
    res_all = []
    # run per target arm so the direction matches that arm's sheet
    for tarm in (0, 1):
        da = unit(dirs[f"arc{tarm}"][LS]); sa = proj_sd(LS, da, c_rows[ARM[c_rows] == tarm])
        vec = (6 * sa * rrand) if al == "rand6" else al * sa * da
        res, probes, _, _ = closed("reach", NCL, 70 + tarm, steer(LS, vec) if al != 0 else None)
        res_all += [r for r, p in zip(res, probes) if p["g"]["tarm"] == tarm]
    R["causal"]["reach_steer"][str(al)] = dict(n=len(res_all), reach_arc_minus_target=mean_ci([r["reach_arc"] - r["target_arc"] for r in res_all]),
                                               reach_arc=mean_ci([r["reach_arc"] for r in res_all]), success=mean_ci([r["success"] for r in res_all]), tip_dist=mean_ci([r["tip_dist"] for r in res_all]))
    log("reach steer", al, R["causal"]["reach_steer"][str(al)]); save(); touch()
# ablation: 8-d map subspace (mean-difference directions) vs random 8-d subspaces inside the top-64 PCs
GROUPS = {"map8": ("arc0", "arc1", "arm", "side", "press", "width", "self", "contact"), "place4": ("arc0", "arc1", "arm", "side"), "press2": ("press", "press_wall")}
def map_U(L, grp="map8"):
    M = np.stack([dirs[k][L] for k in GROUPS[grp]]); Qm, _ = np.linalg.qr(M.T); return Qm.T
VT = {}
def topPC(L):
    if L not in VT: X = Hk[L][::4][:3000].astype(np.float32); X = X - X.mean(0); VT[L] = np.linalg.svd(X, full_matrices=False)[2][:256]
    return VT[L]
def var_removed(L, U): X = Hk[L][::5].astype(np.float32); X = X - X.mean(0); return float(((X @ U.T) ** 2).sum() / (X ** 2).sum())
def rand_U(L, k, target, seed):
    """random k-d subspace inside the top-m PCs, m picked so the variance it removes matches TARGET (the map subspace's)."""
    best = None
    for m in (k, 2 * k, 4 * k, 8 * k, 16 * k, 32 * k, 64 * k):
        m = min(m, 256); Qm, _ = np.linalg.qr(topPC(L)[:m].T @ np.random.default_rng(seed + m).standard_normal((m, k))); U = Qm.T; v = var_removed(L, U)
        if best is None or abs(v - target) < abs(best[1] - target): best = (U, v)
    return best[0]
R["causal"]["ablate"] = {}
BAND = list(range(8, NL)); conds = {"none": None}
for grp in GROUPS:
    k = len(GROUPS[grp]); Um18 = map_U(18, grp); Uall = {L: map_U(L, grp) for L in BAND}
    if grp == "map8": conds["map8_L18"] = ({18: Um18}, PMU); conds["rand8_L18"] = ({18: rand_U(18, k, var_removed(18, Um18), 1)}, PMU)
    conds[f"{grp}_L8-36"] = (Uall, PMU); conds[f"rand_{grp}_L8-36"] = ({L: rand_U(L, k, var_removed(L, Uall[L]), 1) for L in BAND}, PMU); touch()
for cn, cv in conds.items():
    R["causal"]["ablate"][cn] = {"var_removed_L18": var_removed(18, cv[0][18]) if cv else 0.}
    for kd_, sd_ in (("reach", 7), ("find", 8), ("wall", 9)):
        res = closed(kd_, NCL, sd_, ablate(*cv) if cv else None)[0]; R["causal"]["ablate"][cn][kd_] = mean_ci([r["success"] for r in res])
        if kd_ == "wall": R["causal"]["ablate"][cn]["wall_p"] = mean_ci([r["wall_p"] for r in res])
    log("ablate", cn, R["causal"]["ablate"][cn]); save(); touch()

# ---------- 6 video arrays: one wall press, one self-touch, all layers x time ----------
def ep_arrays(kind, seed, name):
    res, probes, Ss, As = closed(kind, 16, seed); i = next((j for j, r in enumerate(res) if r["success"]), 0)
    ep = dict(kind=kind, g=probes[i]["g"], S=np.stack(Ss[i]), A=As[i]); items = [B.prompt(ep, "", None, False)]; ids, lab, kd, att, X = B.collate(items, [ep])
    with torch.no_grad(): out = B.run(ids, kd, att, X, hidden=True)
    kp = items[0][4]; ap = items[0][3]
    H = np.stack([out.hidden_states[L][0, kp].float().cpu().numpy() for L in range(NL)])  # [NL, T, D]
    Hm = np.stack([out.hidden_states[L][0, ap].float().cpu().numpy() for L in range(NL)])
    env = sim.Env(probes[i]["S0q"], np.random.default_rng(0), wall=probes[i]["env"].wall, obj=probes[i]["env"].obj); tr = []
    for t in range(T):
        C = env.contacts(); c = primary(C); tr.append(dict(q=env.q.tolist(), contacts=[{k: (float(v) if not isinstance(v, str) else v) for k, v in cc.items()} for cc in C],
                                                         arc=sim.arc_of(c["seg"], c["u"]) if c else None, arm=c["seg"] // 3 if c else None, p=c["p"] if c else 0., src=c["src"] if c else None))
        if t < T - 1: env.step(As[i][t])
    rd = {}
    for nm in ("place_arc", "pressure", "width", "self_vs_other", "contact_now", "contact_next_from_none"):
        rd[nm] = np.einsum("ltd,ld->lt", H, WTS[nm][0][:, :, 0]) + WTS[nm][1][:, None, 0]
    rd["next_action_probe"] = np.einsum("ltd,ldk->ltk", H, WTS["next_action"][0]) + WTS["next_action"][1][:, None]
    rd["pose"] = np.einsum("ltd,ldk->ltk", H, WTS["pose"][0]) + WTS["pose"][1][:, None]
    pcs = np.einsum("ltd,lkd->ltk", H - PMU[:, None], PCA)
    with torch.no_grad(): logits = LM(out.last_hidden_state[0, kp])[:, ACT].float().cpu().numpy()
    np.savez_compressed(f"{OUT}/video_{name}.npz", H_skin=H.astype(np.float16), H_motor=Hm.astype(np.float16), pca3=pcs, actions=np.array(As[i]), logits=logits,
                        S=np.stack(Ss[i]), q=np.array([x["q"] for x in tr]), true_arc=np.array([np.nan if x["arc"] is None else x["arc"] for x in tr]),
                        true_arm=np.array([-1 if x["arm"] is None else x["arm"] for x in tr]), true_p=np.array([x["p"] for x in tr]),
                        wall=np.array(probes[i]["env"].wall if probes[i]["env"].wall is not None else np.nan), **{f"read_{k}": v for k, v in rd.items()})
    json.dump(dict(kind=kind, g=probes[i]["g"], success=res[i]["success"], steps=tr, actions=As[i], act_names=[sim.ACT_NAMES[a] for a in As[i]],
                   goal=sim.goal_text(kind, probes[i]["g"], ARMK == "named")), open(f"{OUT}/video_{name}.json", "w"))
    return res[i]["success"]
R["video"] = dict(wall=ep_arrays("wall", 9, "wall_press"), reach=ep_arrays("reach", 7, "self_touch")); save(); log("done")
