"""v4 coherence tests. Usage: python interp4.py TAG named|nameless MODEL
(a) RSA: hidden states for touches at 300 places over the whole body (x 6 postures) vs geodesic body distance.
(b) touch vs proprioception: passive touch at X  vs  right fingertip hovering 4 cm above X (no contact): cross-decoding.
(c) continuity: touches along paths over every limb and across joints: verbal readout, probe readout, state step sizes;
    steering by interpolating the injected skin vectors across a joint.
(d) novel postures (outside the training joint box); tool (stick) touches vs the arm map.
(e) rubber-hand analogue: brush felt on the back of the right hand, seen 2 dm away, synchronous vs asynchronous.
plus closed-loop control and held-out answers. Writes OUT/interp_TAG.json and OUT/interp_TAG.npz."""
import sys, os, json, time, numpy as np, torch
import v4, sim4 as S, episodes4 as E
from scipy.stats import spearmanr

OUT = v4.OUT; DEV = v4.DEV
tag, armk = sys.argv[1], sys.argv[2]; model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-0.6B"; named = armk == "named"
QUICK = int(os.environ.get("QUICK", 0))
B = v4.Body(model, named, torch.float32 if "0.6B" in model else torch.bfloat16)
ck = torch.load(f"{OUT}/ckpt_{tag}.pt", map_location=DEV, weights_only=False); B.load_small(ck); B.m.eval()
print("loaded", tag, "step", ck.get("step"), flush=True)
LAY = sorted(set([B.L_IN, (B.L_IN + B.L_FWD) // 2, B.L_FWD, (B.L_FWD + B.nL) // 2, B.nL]))
LF = B.L_FWD; T = E.T; t0 = time.time(); res = {"layers": LAY, "L_IN": B.L_IN, "L_FWD": LF, "nL": B.nL}; NPZ = {}
rng0 = np.random.default_rng(42)
REST = lambda env, t, Sx: S.A_REST


def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def static_ep(q, s=None, p=0.7, w=0.4, lo=None, hi=None, extra=None, seed=0):
    st = dict(kind="babble", q0=np.array(q, float), g={})
    if lo is not None: st["lo"], st["hi"] = lo, hi
    if s is not None: st["other"] = dict(s=np.full(T, s), on=np.ones(T, bool), p=p, w=w)
    if extra: st.update(extra)
    ep = E.rollout(st, np.random.default_rng(seed), policy=REST); ep.update(st=st, kind="babble", act_label=False); ep.pop("env")
    return ep


@torch.no_grad()
def states(eps, bs=32, upto=None):
    """hidden states at the last vision token (end of the last step) for LAY, plus injected vectors and raw input."""
    H = {l: [] for l in LAY}; INJ = []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.tk.prompt(e, [], False, upto_t=upto) for e in ee]
        ids, lab, kind, att, X = v4.collate(items, ee, B.tok.pad_token_id)
        o = B.run(ids, kind, att, X, hidden=True); r = torch.arange(len(ee))
        pos = torch.tensor([it[4][-1] + 2 for it in items])
        for l in LAY: H[l].append(o.hidden_states[l][r, pos.to(DEV)].float().cpu().numpy())
        v = B.ad(X.to(DEV), kind.to(DEV)); INJ.append(torch.cat([v[r, (pos - 2).to(DEV)], v[r, (pos - 1).to(DEV)], v[r, pos.to(DEV)]], -1).float().cpu().numpy())
    out = {l: np.concatenate(v) for l, v in H.items()}; out["inject"] = np.concatenate(INJ)
    out["raw"] = np.stack([v4.norm(e["S"][-1 if upto is None else upto]) for e in eps]); return out


def qtext(key, s_ref=0, li=None):
    s = s_ref
    if li is not None: s = int(np.where(S.LIMB == li)[0][0])
    info = dict(hands=np.zeros((2, 3)), s=s, world=np.zeros(3), p=.5, src="other", part=int(S.PART[s]), vib=dict(pos=4., tip=np.zeros(3)))
    for k, q, a, c in E.questions(info, {"kind": "babble", "g": {}}, named):
        if k == key: return q, c
    raise KeyError(key)


@torch.no_grad()
def ask_digit(eps, key, s_ref=0, li=None, upto=None, bs=32):
    q, c = qtext(key, s_ref, li); items = [B.tk.prompt(e, [(key, q, None, c)], False, upto_t=upto, answer=False) for e in eps]
    L = B.next_logits(items, eps, bs=bs); ev, p = B.digit_ev(L); return ev


def ask_cat(eps, key, s_ref=0):
    q, c = qtext(key, s_ref); LP = B.cand_logp(eps, [(key, q, None, c)] * len(eps))
    P = np.stack([np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum() for lp in LP]); return P, c


def ridge_fit(X, Y, alphas=np.logspace(-1, 5, 13)):
    one = np.ndim(Y) == 1
    X = np.asarray(X, float); Y = np.asarray(Y, float).reshape(len(X), -1); n = len(X); idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(.8 * n)], idx[int(.8 * n):]
    def fit(Xa, Ya, al):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = Ya.mean(0)
        U, S_, Vt = np.linalg.svd(Z, full_matrices=False); Wt = Vt.T @ ((S_ / (S_ ** 2 + al))[:, None] * (U.T @ (Ya - ym)))
        return lambda Xb: ((np.asarray(Xb, float) - mu) / sd) @ Wt + ym
    best = min(alphas, key=lambda al: ((fit(X[tr], Y[tr], al)(X[va]) - Y[va]) ** 2).mean())
    f = fit(X, Y, best)
    return (lambda Xb: f(Xb)[:, 0]) if one else f


def cvpred(X, Y, k=5, groups=None):
    X = np.asarray(X, float); Y = np.asarray(Y, float).reshape(len(X), -1); n = len(X)
    g = np.arange(n) if groups is None else np.asarray(groups); ug = np.unique(g); perm = np.random.default_rng(1).permutation(len(ug)); P = np.zeros_like(Y)
    for f in range(k):
        teg = ug[perm[f::k]]; te = np.isin(g, teg); P[te] = ridge_fit(X[~te], Y[~te])(X[te])
    return P


def r2(P, Y):
    Y = np.asarray(Y, float).reshape(len(P), -1); return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def to_world(ev, rng_): return rng_[0] + (ev + 0.5) / 10 * (rng_[1] - rng_[0])


def keyname(l): return str(l)


# =============== (a) RSA over the whole body ===============
NP_A = 3 if QUICK else 6; NPT = 60 if QUICK else 300
POS = [S.rand_q(rng0) for _ in range(NP_A)]
pts = [0]; d = S.GEO[0].copy()
for _ in range(NPT - 1): i = int(d.argmax()); pts.append(i); d = np.minimum(d, S.GEO[i])
pts = np.array(pts); eps = [static_ep(q, s) for s in pts for q in POS]
Ha = states(eps); log("(a) states", len(eps))
G = S.GEO[np.ix_(pts, pts)].astype(float); RE = np.linalg.norm(S.P0[pts][:, None] - S.P0[pts][None], axis=-1); iu = np.triu_indices(NPT, 1)
partp = S.PART[pts]; same_part = partp[:, None] == partp[None]
rsa, rsa_e, bcon, mats = {}, {}, {}, {}
for l, Hl in Ha.items():
    M = Hl.reshape(NPT, NP_A, -1).mean(1); M = M - M.mean(0); D = np.linalg.norm(M[:, None] - M[None], axis=-1)
    rsa[keyname(l)] = float(spearmanr(D[iu], G[iu])[0]); rsa_e[keyname(l)] = float(spearmanr(D[iu], RE[iu])[0])
    near = (G > .5) & (G < 1.5); cross = near & ~same_part; within = near & same_part
    bcon[keyname(l)] = float((D[cross] / G[cross]).mean() / (D[within] / G[within]).mean())
    if l == LF or l == "inject": mats[keyname(l)] = D.astype(np.float32)
    # local RSA: only pairs closer than 3 dm (does the fine map follow the skin?)
    loc = (G[iu] < 3.0)
    rsa.setdefault("local", {})[keyname(l)] = float(spearmanr(D[iu][loc], G[iu][loc])[0])
# arms only (v3-comparable): pairs on the same arm use arc distance
armm = np.isin(S.LIMB[pts], [0, 1])
res["a"] = dict(rsa_geodesic=rsa, rsa_rest_euclid=rsa_e, part_boundary_contrast=bcon, n_points=NPT, n_postures=NP_A)
M = Ha[LF].reshape(NPT, NP_A, -1).mean(1); Mc = M - M.mean(0); U, Sv, Vt = np.linalg.svd(Mc, full_matrices=False)
NPZ.update(a_pts=pts, a_pc=(U[:, :20] * Sv[:20]).astype(np.float32), a_D_mid=mats.get(keyname(LF)), a_D_inj=mats.get("inject"), a_G=G.astype(np.float32))
# body-coordinate probes (posture-general): rest xyz from states, grouped CV by point
Y = np.repeat(S.P0[pts], NP_A, 0); grp = np.repeat(np.arange(NPT), NP_A)
res["a"]["rest_xyz_probe_r2"] = {keyname(l): r2(cvpred(Hl, Y, groups=grp), Y) for l, Hl in Ha.items()}
log("(a)", json.dumps({k: round(v, 3) for k, v in rsa.items() if k != "local"}))
# verbal body-map answers at the 300 places
evu = ask_digit(eps, "bu"); evv = ask_digit(eps, "bv")
tu = np.array([int(E.d10(S.P0[s][1], *E.RY)) for s in pts for q in POS]); tv = np.array([int(E.d10(S.P0[s][0], *E.RX)) for s in pts for q in POS])
res["a"]["verbal_body_height_mae"] = float(np.abs(evu - tu).mean()); res["a"]["verbal_body_across_mae"] = float(np.abs(evv - tv).mean())
NPZ.update(a_evu=evu.reshape(NPT, NP_A).mean(1), a_evv=evv.reshape(NPT, NP_A).mean(1))
# probe of world xyz across postures (posture integration): raw/inject vs layers
Yw = np.stack([e["info"]["world"] for e in eps])
res["a"]["world_xyz_probe_r2"] = {keyname(l): r2(cvpred(Hl, Yw, groups=grp), Yw) for l, Hl in Ha.items()}
log("(a) done")

# =============== (b) touch vs proprioception ===============
cand = np.where(~S.TOUCH_EXCL[1] & ~np.isin(S.HARD_S, S.ARM_BONES[1]) & (S.PART != 4) & (S.PART != 5))[0]
NB_ = 16 if QUICK else 40
tg_ = [int(cand[rng0.integers(len(cand))])]; d = S.GEO[tg_[0]][cand].copy()
for _ in range(NB_ * 3): i = int(cand[d.argmax()]); tg_.append(i); d = np.minimum(d, S.GEO[i][cand])
touch_eps, hover_eps, self_eps, tgs = [], [], [], []
for s in tg_:
    if len(tgs) >= NB_: break
    n_ = S.N0[s]
    qb = S.rand_q(rng0); qb[S.JNAME.index("R shoulder abduct")] = .9; qb[S.JNAME.index("R elbow")] = 0.3  # right arm away
    def hov(Q, s=s):
        P = S.point_world(Q, [int(S.HT[1][0]), s]); R_, T_ = S.fk(Q); nn = np.einsum("kij,j->ki", R_[:, S.HARD_S[s]], S.N0[s])
        return np.linalg.norm(P[:, 0] - (P[:, 1] + 0.45 * nn), axis=1)
    qh, sc = E.plan(qb, hov, rng0, S.TLO, S.THI)
    if sc > 0.3: continue
    qt, sc2 = E.plan(qh, E.reach_score(1, s), rng0, S.TLO, S.THI)
    eh = static_ep(qh); et = static_ep(S.rand_q(rng0), s); es = static_ep(qt)  # touch: independent posture
    if eh["info"]["s"] is not None: continue  # hover must not touch
    if not any(c["src"] == "self" and S.GEO[c["s"], s] < 1.0 for c in es["info"]["contacts"]): es = None
    touch_eps.append(et); hover_eps.append(eh); self_eps.append(es); tgs.append(s)
tgs = np.array(tgs); log("(b) targets", len(tgs), "self-touch ok", sum(e is not None for e in self_eps))
Ht, Hh = states(touch_eps), states(hover_eps); Yb = S.P0[tgs]
okself = [i for i, e in enumerate(self_eps) if e is not None]; Hs = states([self_eps[i] for i in okself]) if okself else None
# a body-location probe trained on the (a) set (passive touches, many postures) - applied to touch / hover / self-touch states
xb = {}
for l in list(LAY) + ["inject"]:
    f = ridge_fit(Ha[l], np.repeat(S.P0[pts], NP_A, 0))
    def err(Hx, Yx): return float(np.linalg.norm(f(Hx) - Yx, axis=1).mean())
    # identification: is the hover state closest to the touch state of the SAME place? (centered per condition)
    A_ = Ht[l] - Ht[l].mean(0); B_ = Hh[l] - Hh[l].mean(0)
    C = np.corrcoef(np.concatenate([A_, B_]))[: len(A_), len(A_):]
    xb[keyname(l)] = dict(touch_err=err(Ht[l], Yb), hover_err=err(Hh[l], Yb), self_err=err(Hs[l], Yb[okself]) if Hs else None,
                         chance_err=float(np.linalg.norm(Yb[:, None] - Yb[None], axis=-1).mean()),
                         id_acc=float((C.argmax(1) == np.arange(len(C))).mean()), id_chance=1 / len(C),
                         rdm_corr=float(spearmanr(np.linalg.norm(A_[:, None] - A_[None], axis=-1)[np.triu_indices(len(A_), 1)],
                                                  np.linalg.norm(B_[:, None] - B_[None], axis=-1)[np.triu_indices(len(A_), 1)])[0]))
res["b"] = dict(by_layer=xb, n=len(tgs))
fbest = ridge_fit(Ha[LF], np.repeat(S.P0[pts], NP_A, 0))
NPZ.update(b_tgs=tgs, b_touch_pred=fbest(Ht[LF]), b_hover_pred=fbest(Hh[LF]), b_self_pred=fbest(Hs[LF]) if Hs else np.zeros((0, 3)), b_okself=np.array(okself))
log("(b)", json.dumps({k: {kk: (round(vv, 3) if vv is not None else None) for kk, vv in v.items()} for k, v in xb.items()}))

# =============== (c) continuity along limbs and across joints ===============
def lm(name): return int(S.LMI[S.LMN.index(name)])
def path(a, b, n=40):
    D = S.GEO[a, b]; return np.array([int(np.argmin(np.abs(S.GEO[a] - f * D) + np.abs(S.GEO[b] - (1 - f) * D))) for f in np.linspace(0, 1, n)])
def limb_end(li):
    m = S.LIMB == li; idx = np.where(m)[0]; return int(idx[np.argmax(S.ARC[idx])]), int(idx[np.argmin(S.ARC[idx])])
paths = {}
for li, nm in enumerate(["left arm", "right arm", "left leg", "right leg"]):
    tip_, root_ = limb_end(li); paths[nm] = path(root_, tip_)
paths["head to belly"] = path(lm("the top of your head"), lm("your belly"))
paths["left hand to right hand"] = path(limb_end(0)[0], limb_end(1)[0], 60)
NP_C = 3 if QUICK else 6; POSC = [S.rand_q(rng0) for _ in range(NP_C)]
cres = {}; NPZ["c_names"] = np.array(list(paths))
# probe for limb arc trained on the (a) set limb points
arc_f = {}
for l in list(LAY) + ["inject"]:
    m = np.repeat(S.LIMB[pts] >= 0, NP_A)
    arc_f[keyname(l)] = ridge_fit(Ha[l][m], np.repeat(S.ARC[pts], NP_A)[m])
for nm, pth in paths.items():
    ee = [static_ep(q, s) for s in pth for q in POSC]; Hc = states(ee); n = len(pth)
    geo_step = np.array([S.GEO[pth[i], pth[i + 1]] for i in range(n - 1)]) + 1e-3
    cum = np.r_[0, np.cumsum(geo_step)]
    bnd = np.array([S.PART[pth[i]] != S.PART[pth[i + 1]] for i in range(n - 1)])
    out = dict(cum=cum.tolist(), parts=[int(S.PART[s]) for s in pth])
    for l in [LF, "inject"]:
        M = Hc[l].reshape(n, NP_C, -1).mean(1); stp = np.linalg.norm(np.diff(M, axis=0), axis=1) / geo_step
        out[f"step_ratio_joint_vs_within_{keyname(l)}"] = float(stp[bnd].mean() / stp[~bnd].mean()) if bnd.any() else None
        out[f"steps_{keyname(l)}"] = stp.tolist()
    if nm in ("left arm", "right arm", "left leg", "right leg"):
        li = ["left arm", "right arm", "left leg", "right leg"].index(nm)
        ev = ask_digit(ee, "arc", li=li).reshape(n, NP_C).mean(1); out["verbal_arc"] = ev.tolist(); out["true_arc"] = S.ARC[pth].tolist()
        out["verbal_arc_spearman"] = float(spearmanr(ev, S.ARC[pth])[0])
        out["verbal_max_drop"] = float(max(0, max(ev[:i + 1].max() - ev[i] for i in range(n))))
        # staircase index (v3): within-part slope / whole-limb slope of the answer vs true arc
        a_ = S.ARC[pth]; tot = np.polyfit(a_, ev, 1)[0]; pp = np.array([S.PART[s] for s in pth])
        ws = [np.polyfit(a_[pp == u], ev[pp == u], 1)[0] for u in np.unique(pp) if (pp == u).sum() > 4 and np.ptp(a_[pp == u]) > .05]
        out["staircase_index"] = float(np.mean(ws) / tot)
        for l in [LF, "inject"]:
            pr = arc_f[keyname(l)](Hc[l]).reshape(n, NP_C).mean(1); out[f"probe_arc_{keyname(l)}"] = pr.tolist()
    evu_ = ask_digit(ee, "bu").reshape(n, NP_C).mean(1); out["verbal_body_height"] = evu_.tolist()
    out["true_body_height"] = [float(E.d10(S.P0[s][1], *E.RY)) for s in pth]
    evv_ = ask_digit(ee, "bv").reshape(n, NP_C).mean(1); out["verbal_body_across"] = evv_.tolist()
    cres[nm] = out; log("(c)", nm, {k: v for k, v in out.items() if isinstance(v, float)})
res["c"] = cres
# steering: interpolate the injected skin vectors between two touches across a joint
pairs = {"elbow (left)": ("your left upper arm", "your left forearm"), "wrist (left)": ("your left forearm", "the back of your left hand"),
         "knee (right)": ("your right thigh", "your right shin"), "ankle (right)": ("your right shin", "the top of your right foot"),
         "shoulder (left)": ("your left shoulder", "your left upper arm"), "neck": ("your chin", "your chest")}
steer = {}
for nm, (a_, b_) in pairs.items():
    sa, sb = lm(a_), lm(b_); li = int(S.LIMB[sb]) if S.LIMB[sb] >= 0 else int(S.LIMB[sa])
    curves, curves_bu, curves_real = [], [], []
    for q in POSC[:4]:
        eA, eB = static_ep(q, sa), static_ep(q, sb)
        itA, itB = B.tk.prompt(eA, [], False), B.tk.prompt(eB, [], False)
        ids, lab, kind, att, X = v4.collate([itA, itB], [eA, eB], B.tok.pad_token_id)
        with torch.no_grad(): V = B.ad(X.to(DEV), kind.to(DEV))
        vA, vB = V[0:1], V[1:2]; nb = V.shape[1]
        cv, cu = [], []
        for al in np.linspace(0, 1, 11):
            mix = (1 - al) * vA + al * vB
            def vfn(v, mix=mix): return torch.cat([mix.expand(v.shape[0], -1, -1), v[:, nb:]], 1)
            orig = B.run
            B.run = lambda ids, kind, att, X, hidden=False, vec_fn=None, o=orig, f=vfn: o(ids, kind, att, X, hidden, f)
            try:
                cv.append(float(ask_digit([eA], "arc", li=li)[0]) if li >= 0 else np.nan); cu.append(float(ask_digit([eA], "bu")[0]))
            finally: B.run = orig
        curves.append(cv); curves_bu.append(cu)
        # the real touches along the geodesic between the two places, same posture
        pth = path(sa, sb, 11); er = [static_ep(q, s) for s in pth]
        curves_real.append(ask_digit(er, "arc", li=li).tolist() if li >= 0 else ask_digit(er, "bu").tolist())
    cv = np.nanmean(curves, 0); cu = np.mean(curves_bu, 0); cr = np.mean(curves_real, 0)
    jump = lambda c: float(np.max(np.abs(np.diff(c))) / (abs(c[-1] - c[0]) + 1e-6))
    steer[nm] = dict(arc=cv.tolist(), body_height=cu.tolist(), real=cr.tolist(), max_jump_frac_arc=jump(cv) if li >= 0 else None, max_jump_frac_height=jump(cu))
    log("(c) steer", nm, np.round(cv, 2), np.round(cr, 2))
res["c_steer"] = steer

# =============== (d) novel postures and the stick ===============
def novel_q(rng):
    q = S.rand_q(rng); js = rng.choice(S.CMD, 5, replace=False)
    for j in js: q[j] = rng.uniform(S.JLO[j], S.TLO[j]) if rng.random() < .5 else rng.uniform(S.THI[j], S.JHI[j])
    return q
ND = 60 if QUICK else 200; rngd = np.random.default_rng(5)
fam, nov = [], []
for i in range(ND):
    s = int(rngd.integers(S.NS))
    fam.append(static_ep(S.rand_q(rngd), s)); nov.append(static_ep(novel_q(rngd), s, lo=S.JLO, hi=S.JHI))
dres = {}
for nm_, ee in (("familiar", fam), ("novel", nov)):
    r_ = {}
    for key, rr, idx in (("wy", E.WY, 1), ("wx", E.WX, 0), ("wz", E.WZ, 2)):
        ev = ask_digit(ee, key); tr = np.array([int(E.d10(e["info"]["world"][idx], *rr)) for e in ee]); r_[key] = float(np.abs(ev - tr).mean())
    ev = ask_digit(ee, "bu"); tr = np.array([int(E.d10(S.P0[e["info"]["s"]][1], *E.RY)) for e in ee]); r_["bu"] = float(np.abs(ev - tr).mean())
    ev = ask_digit(ee, "h1y"); tr = np.array([int(E.d10(e["info"]["hands"][1][1], *E.WY)) for e in ee]); r_["h1y"] = float(np.abs(ev - tr).mean())
    ev = ask_digit(ee, "h0x"); tr = np.array([int(E.d10(e["info"]["hands"][0][0], *E.WX)) for e in ee]); r_["h0x"] = float(np.abs(ev - tr).mean())
    P_, c_ = ask_cat(ee, "part"); tr = [S.PARTS[e["info"]["part"]] if named else E.ZONE[e["info"]["part"]] for e in ee]
    r_["part_acc"] = float(np.mean([c_[int(p.argmax())] == t for p, t in zip(P_, tr)]))
    import collections; log("part answers", collections.Counter(str(c_[int(p.argmax())]) for p in P_).most_common(5), "truth", collections.Counter(tr).most_common(5))
    Hn = states(ee); Yn = np.stack([e["info"]["world"] for e in ee]); fw = ridge_fit(Ha[LF], Yw)
    r_["world_probe_err_dm"] = float(np.linalg.norm(fw(Hn[LF]) - Yn, axis=1).mean())
    dres[nm_] = r_
res["d_postures"] = dres; log("(d) postures", dres)
# stick: right hand holds a stick of length L, a ball touches its tip. Does the arm-arc map extend past the fingertips?
Ls = [2., 3., 4., 5., 6., 7.]; stick_eps = {L: [] for L in Ls}; rngs = np.random.default_rng(9)
for L in Ls:
    while len(stick_eps[L]) < (6 if QUICK else 16):
        q = S.rand_q(rngs); q[S.FING[1]] = 1.4; tip = E.stick_tip_q(q[None], 1, L)[0]
        st = dict(kind="babble", q0=q, g={}, stick=dict(hand=1, L=L), ball=dict(c=tip + 0.3 * rngs.normal(size=3) / 3, r=0.5, visible=True, p=1.))
        ep = E.rollout(st, np.random.default_rng(0), policy=REST); ep.update(st=st, kind="tool", act_label=False); ep.pop("env")
        if "vib" in ep["info"] and ep["info"]["s"] is not None and S.PART[ep["info"]["s"]] == 11: stick_eps[L].append(ep)
# reference: touches on the right arm at known arc (same postures as (a))
sres = {}
allst = [e for L in Ls for e in stick_eps[L]]; Hst = states(allst); Lof = np.repeat(Ls, [len(stick_eps[L]) for L in Ls])
for l in [B.L_IN, LF, (LF + B.nL) // 2, "inject"]:
    m = np.repeat(S.LIMB[pts] == 1, NP_A); f = ridge_fit(Ha[l][m], np.repeat(S.ARC[pts], NP_A)[m]); pr = f(Hst[l])
    sres[keyname(l)] = dict(probe_arc_by_L=[float(pr[Lof == L].mean()) for L in Ls], probe_arc_sd=[float(pr[Lof == L].std()) for L in Ls],
                          slope_per_dm=float(np.polyfit(Lof, pr, 1)[0]), r=float(np.corrcoef(Lof, pr)[0, 1]))
ev_st = ask_digit(allst, "stick"); evarc = ask_digit(allst, "arc", li=1)
sres["verbal_stick_by_L"] = [float(ev_st[Lof == L].mean()) for L in Ls]; sres["verbal_arm_arc_by_L"] = [float(evarc[Lof == L].mean()) for L in Ls]
# hand-only reference: palm touch arc
palm_ep = [static_ep(S.rand_q(rngs), int(S.HT[1][1])) for _ in range(8)]; Hp = states(palm_ep)
for l in [B.L_IN, LF, (LF + B.nL) // 2, "inject"]:
    m = np.repeat(S.LIMB[pts] == 1, NP_A); f = ridge_fit(Ha[l][m], np.repeat(S.ARC[pts], NP_A)[m]); sres[keyname(l)]["palm_touch_arc"] = float(f(Hp[l]).mean())
sres["Ls"] = Ls; res["d_stick"] = sres; log("(d) stick", json.dumps(sres)[:600])

# =============== (e) rubber-hand analogue ===============
NR = 4 if QUICK else 12; rnge = np.random.default_rng(11)
dorsum = np.where(np.isin(S.HARD_S, S.ARM_BONES[1][2:3]) & (S.N0[:, 0] < -0.2))[0]
if len(dorsum) < 5: dorsum = np.where(S.HARD_S == S.ARM_BONES[1][2])[0]
OFF = {"left": np.array([2., 0, 0]), "up": np.array([0, 2., 0])}
conds = ["sync", "async", "no vision", "sync, no offset"]
rres = {c: {o: [] for o in OFF} for c in conds}; rhand = {c: {o: [] for o in OFF} for c in conds}; rtime = {c: [] for c in conds}
for i in range(NR):
    q = S.rand_q(rnge); s0 = int(rnge.choice(dorsum)); pth = np.array([s0] * T)
    on = rnge.random(T) < .55; on[-1] = True; on[-2] = False
    for o, off in OFF.items():
        for c in conds:
            seen_on = on.copy() if c in ("sync", "sync, no offset") else np.roll(on, 2)
            br = dict(felt=True, visible=c != "no vision", s=pth, on=on, seen_on=seen_on, p=0.6, offset=(0 * off if c == "sync, no offset" else off))
            st = dict(kind="brush", q0=q, g=dict(arm=1, mode="body"), brush=br)
            ep = E.rollout(st, np.random.default_rng(0), policy=REST); ep.update(st=st, kind="brush", act_label=False); ep.pop("env")
            rres[c][o].append(ep)
# hand-position probe from the end-of-step state (trained on held-out babble+brush episodes with known hand positions)
rngp = np.random.default_rng(12); pe = [E.make_episode(rngp, k) for k in (["babble"] * (40 if QUICK else 200) + ["brush"] * (40 if QUICK else 200))]
for e in pe: e.pop("env", None)
Hpe = states(pe); Yh = np.stack([e["info"]["hands"][1] for e in pe]); fh = ridge_fit(Hpe[LF], Yh)
eres = {}
for c in conds:
    eres[c] = {}
    for o, off in OFF.items():
        ee = rres[c][o]; true = np.stack([e["info"]["hands"][1] for e in ee]); u = off / np.linalg.norm(off)
        vx = to_world(ask_digit(ee, "h1x"), E.WX); vy = to_world(ask_digit(ee, "h1y"), E.WY); vz = to_world(ask_digit(ee, "h1z"), E.WZ)
        verbal = np.stack([vx, vy, vz], 1); Hr = states(ee); pr = fh(Hr[LF])
        eres[c][o] = dict(verbal_drift_dm=float(((verbal - true) @ u).mean()), probe_drift_dm=float(((pr - true) @ u).mean()),
                          verbal_drift_sd=float(((verbal - true) @ u).std()), probe_drift_sd=float(((pr - true) @ u).std()))
        # time course (probe) at steps 4, 8, 12, 16, 19
        tc = []
        for tt in (2, 5, 8, 11, 14, 17, 19):
            Ht_ = states(ee, upto=tt); tc.append(float(((fh(Ht_[LF]) - true) @ u).mean()))
        eres[c][o]["probe_drift_time"] = tc
res["e"] = eres; res["e_steps"] = [2, 5, 8, 11, 14, 17, 19]; log("(e)", json.dumps(eres)[:800])

# =============== closed loop ===============
@torch.no_grad()
def closed_loop(kind, n, seed):
    rng = np.random.default_rng(seed); sts = []
    while len(sts) < n:
        st = E.make_setup(rng, kind); ep = E.rollout(st, rng)
        if E.success(st, ep["info"]): sts.append(st)
    envs = [E.setup_env(st, np.random.default_rng(i)) for i, st in enumerate(sts)]; Sx = [[env.sense()] for env in envs]; A = [[] for _ in envs]
    for t in range(T - 1):
        eps = [dict(S=np.stack(Sx[i]), A=A[i], st=sts[i]) for i in range(n)]
        items = [B.tk.prompt(e, [], False) for e in eps]; L = B.next_logits(items, eps)
        acts = L[:, B.tk.ACT].argmax(1).tolist()
        for i, env in enumerate(envs): env.step(acts[i]); A[i].append(acts[i]); Sx[i].append(env.sense())
    succ = [E.success(st, E.summarize(env, st)) for st, env in zip(sts, envs)]
    per = {}
    if kind == "reach":
        for st, ok in zip(sts, succ): per.setdefault(st["g"]["lm"], []).append(ok)
    return float(np.mean(succ)), {k: float(np.mean(v)) for k, v in per.items()}, np.bincount(sum(A, []), minlength=S.NA).tolist()
def random_cl(kind, n, seed):
    rng = np.random.default_rng(seed); ok = []
    while len(ok) < n:
        st = E.make_setup(rng, kind); ep = E.rollout(st, rng)
        if E.success(st, ep["info"]): ok.append(E.success(st, E.rollout(st, rng, policy=lambda env, t, Sx: int(rng.integers(S.NA)))["info"]))
    return float(np.mean(ok))
NC = 16 if QUICK else 40
res["closed_loop"] = {k: closed_loop(k, NC, 7 + i) for i, k in enumerate(["reach", "find", "vreach", "tool", "wall"])}
res["closed_loop_random"] = {k: random_cl(k, NC, 7 + i) for i, k in enumerate(["reach", "find", "vreach", "tool", "wall"])}
log("closed loop", {k: v[0] for k, v in res["closed_loop"].items()}, "random", res["closed_loop_random"])

# self vs other probe on held-out babble (v3-comparable)
src = np.array([e["info"]["src"] == "self" if e["info"]["s"] is not None else False for e in pe]); has = np.array([e["info"]["s"] is not None for e in pe])
if 10 < src[has].sum() < has.sum() - 10:
    res["self_probe_acc"] = {keyname(l): float(((cvpred(Hpe[l][has], src[has].astype(float)) > .5)[:, 0] == src[has]).mean()) for l in list(LAY) + ["inject"]}
    res["self_frac"] = float(src[has].mean())
json.dump(res, open(f"{OUT}/interp_{tag}.json", "w"))
np.savez_compressed(f"{OUT}/interp_{tag}.npz", **{k: v for k, v in NPZ.items() if v is not None})
log("written")
