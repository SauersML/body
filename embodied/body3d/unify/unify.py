"""body-unify: one body map for word-described and sensed touches. python unify.py ARM  (ctrl|words|pair|align|both)
env: STEPS INIT_CK OUTDIR NEV SMOKE. Trains from the v2 8B ckpt, then runs all tests (see PROGRESS.md)."""
import sys, os
for k, v in dict(X2_CANON="1", X2_GSMOOTH="1", X2_GCHI=".07", X2_GCLO=".04").items(): os.environ.setdefault(k, v)
import json, time, math, numpy as np, torch, torch.nn.functional as F, mujoco
import train3d as TR, env as E, episodes as P, hum as H, episodes2 as Q
from start_poses import start_pose, POSE_SEEDS
Q.install()
ARM = sys.argv[1]; MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B"); OUT = TR.OUT; DEV = TR.DEV; T = P.T
SMOKE = os.environ.get("SMOKE") == "1"; STEPS = int(os.environ.get("STEPS", 20 if SMOKE else 1500)); NEV = int(os.environ.get("NEV", 8 if SMOKE else 96))
BS = 24; LAM = 0.5; TEMP = 0.1; os.makedirs(OUT, exist_ok=True); t00 = time.time()
def log(*a): print(f"[{round(time.time() - t00)}s]", *a, flush=True)

PARTS = H.PARTS; NONHAND = [p for p in range(len(PARTS)) if not PARTS[p].endswith("hand")]
HELD = [PARTS.index(x) for x in ("belly", "left forearm", "right thigh", "left shin")]
TRAIN = [p for p in NONHAND if p not in HELD]
TPL_TRAIN = ["Someone taps your {}.", "A feather brushes your {}.", "You feel warmth on your {}.", "A cold drop of water lands on your {}.",
             "Someone pokes your {}.", "A fly lands on your {}.", "A friend rests a hand on your {}.", "Something presses against your {}.",
             "A breeze tickles your {}.", "You feel a sharp pinch on your {}.", "A ball bumps into your {}.", "Someone gently pats your {}.",
             "A warm cloth is laid on your {}.", "An ant crawls over your {}.", "A finger presses your {}.", "Something soft touches your {}."]
TPL_EVAL = ["A stranger touches your {}.", "You feel a light tap on your {}.", "A cat rubs against your {}.", "Rain falls on your {}.",
            "Someone nudges your {}.", "A leaf lands on your {}.", "A nurse touches your {}.", "You feel pressure on your {}."]
NEUTRAL = ["Someone touches you.", "You feel a light tap.", "Something brushes against you.", "A friend pats you.",
           "You feel a touch.", "Something cold touches you.", "Someone pokes you.", "A fly lands on you."]
MIX = dict(ctrl=dict(c=18, k=6), words=dict(c=10, k=6, A=4, B=4), align=dict(c=10, k=6, A=4, B=4),
           pair=dict(c=10, k=6, A=3, B=2, C=3), both=dict(c=10, k=6, A=3, B=2, C=3))[ARM]
USE_ALIGN = ARM in ("align", "both")


class Tk(TR.Tok):
    def build(s, segs):
        """segs: list of ('t', text, lm, mark) | ('steps', n, insert{t: segs}) . returns ids, lab, kind, apos, spos, marks{name: pos}"""
        ids, lab, kind, apos, spos, marks = [], [], [], [], [], {}
        def add(t, lm=False, k=0): ids.extend(t); lab.extend(list(t) if lm else [-100] * len(t)); kind.extend([k] * len(t))
        def text(seg):
            _, x, lm, mk = seg; t = s.enc(x); add(t, lm)
            if mk: marks[mk] = len(ids) - 1
        for seg in segs:
            if seg[0] == "t": text(seg); continue
            _, n, ins = seg
            for t in range(n):
                spos.append(len(ids)); add([s.PH[0]], k=1); add([s.PH[1]], k=2); add([s.PH[2]], k=3)
                if t < T: apos.append(len(ids)); add([s.MOT])
                for sg in ins.get(t, []): text(sg)
        return ids, lab, kind, apos, spos, marks


def qa_segs(qa):
    out = []
    for (k, q, a, c) in qa: out += [("t", f"\nQ: {q}\nA:", False, None), ("t", " " + a, True, None)]
    return out


def ev_sent(rng, p, tpl=TPL_TRAIN): return tpl[int(rng.integers(len(tpl)))].format(PARTS[p])


def peak_t(e, hi=None):
    sk = e["S"][:, TR.SL[1]][:, H.TPART == e["info"]["part"]].sum(1); return int(sk[1:hi].argmax()) + 1


def side_q(p):
    n = PARTS[p]; side = "left" if n.startswith("left") else "right" if n.startswith("right") else "middle"
    grp = "arm" if ("arm" in n) else "leg" if any(x in n for x in ("thigh", "shin", "foot")) else "head" if n == "head" else "torso"
    return [("side", "Which side of your body?", side, ["left", "right", "middle"]), ("grp", "Is it your arm, leg, torso or head?", grp, ["arm", "leg", "torso", "head"])]


def item(kind, rng, tk):
    """returns (built, ep or None, info dict with modality marks)"""
    if kind == "c":
        e = TR.slim(P.make_episode(rng)); qs = P.questions(e, True); qa = [qs[i] for i in rng.permutation(len(qs))]
        g = P.goal_text(e, True) if rng.uniform() > 0.05 else ""
        b = tk.build(([("t", "Goal: " + g + "\n", False, None)] if g else []) + [("t", "Body:", False, None), ("steps", T + 1, {})] + qa_segs(qa))
        sp = b[4][peak_t(e, 5 if e["kind"] == "poke_reach" else None)] + 1 if e["kind"] in ("poke", "poke_reach") else None
        return b, e, dict(sens=sp, part=e["info"].get("part"))
    if kind == "k":
        e = TR.slim(P.make_episode(rng, "poke")); qs = P.questions(e, True)
        b = tk.build([("t", "Body:", False, None), ("steps", T + 1, {})] + qa_segs(qs))
        return b, e, dict(sens=b[4][peak_t(e)] + 1, part=e["info"]["part"])
    if kind == "A":
        while True:
            sim, info, pol = Q.make_scene("reach", rng)
            if info["part"] in TRAIN: break
        e = TR.slim(Q.run_expert("reach", sim, info, pol)); p = info["part"]
        qs = P.questions(e, True); qs = [x for x in qs if x[0] != "self"]
        b = tk.build([("t", ev_sent(rng, p), False, "word"), ("t", "\nGoal: Touch where you were touched.\n", False, None), ("t", "Body:", False, None),
                      ("steps", T + 1, {})] + qa_segs([qs[i] for i in rng.permutation(len(qs))]))
        return b, e, dict(word=b[5]["word"], part=p)
    if kind == "B":
        p = TRAIN[int(rng.integers(len(TRAIN)))]
        qs = [("part", "Where were you touched?", PARTS[p], None)] + side_q(p); qs = [qs[i] for i in rng.permutation(len(qs))]
        b = tk.build([("t", ev_sent(rng, p), False, "word")] + qa_segs(qs))
        return b, None, dict(word=b[5]["word"], part=p)
    if kind == "C":
        while True:
            sim, info, pol = Q.make_scene("poke_reach", rng)
            if info["part"] in TRAIN: break
        e = TR.slim(Q.run_expert("poke_reach", sim, info, pol)); p = info["part"]; tp = peak_t(e, 5)
        nar = [("t", "\n(" + ev_sent(rng, p).replace(".", ")"), True, None), ("t", "\n", False, None)]
        qs = P.questions(e, True)
        b = tk.build([("t", "Goal: Touch where you were touched.\n", False, None), ("t", "Body:", False, None), ("steps", T + 1, {4: nar})]
                     + qa_segs([qs[i] for i in rng.permutation(len(qs))]))
        return b, e, dict(sens=b[4][tp] + 1, part=p)


def collate(built, eps, pad):
    n = max(len(x[0]) for x in built); Bn = len(built)
    ids = torch.full((Bn, n), pad); lab = torch.full((Bn, n), -100); kind = torch.zeros(Bn, n, dtype=torch.long)
    att = torch.zeros(Bn, n, dtype=torch.long); X = torch.zeros(Bn, n, E.SENSOR_DIM)
    for b, (it, ep) in enumerate(zip(built, eps)):
        i, l, k = it[:3]; L = len(i); ids[b, :L] = torch.tensor(i); lab[b, :L] = torch.tensor(l); kind[b, :L] = torch.tensor(k); att[b, :L] = 1
        if ep is not None:
            Sn = torch.tensor(TR.norm(ep["S"][:len(it[4])]))
            for t, p in enumerate(it[4]): X[b, p:p + 3] = Sn[t]
    return ids, lab, kind, att, X


class Stream(torch.utils.data.IterableDataset):
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([7, wid, int(time.time() * 1000) % 100000]); tk = Tk(MODEL, True)
        kinds = [k for k, n in MIX.items() for _ in range(n)]
        while True:
            res = [item(k, rng, tk) for k in kinds]; built = [r[0] for r in res]; eps = [r[1] for r in res]; inf = [r[2] for r in res]
            ids, lab, kind, att, X = collate(built, eps, tk.tok.pad_token_id)
            body = [i for i, e in enumerate(eps) if e is not None]
            yield (ids, lab, kind, att, X, body, [built[i][3] for i in body], np.stack([TR.norm(eps[i]["S"][1:T + 1]) for i in body]),
                   np.stack([TR.anorm(eps[i]["A"]) for i in body]), np.stack([eps[i]["mask"] for i in body]), inf)


B = TR.Body(MODEL, True, torch.bfloat16); B.tk = Tk(MODEL, True)
INIT = os.environ.get("INIT_CK", "/root/ck/x8abs_final_Qwen3-8B_named.pt")
ck = torch.load(INIT, map_location=DEV, weights_only=False); B.load_small(ck); B.ad.cap.fill_(ck["rn"]); log("loaded", INIT)
LA = B.L_FWD  # 18


def align_loss(H18, inf):
    w = [(i, x["part"]) for i, x in enumerate(inf) if x.get("word") is not None and x["part"] in TRAIN]
    s = [(i, x["part"]) for i, x in enumerate(inf) if x.get("sens") is not None and x["part"] in TRAIN]
    if len(w) < 2 or len(s) < 2: return torch.zeros((), device=DEV)
    zw = torch.stack([H18[i, inf[i]["word"]] for i, _ in w]).float(); zs = torch.stack([H18[i, inf[i]["sens"]] for i, _ in s]).float()
    zw = F.normalize(zw - zw.mean(0), dim=-1); zs = F.normalize(zs - zs.mean(0), dim=-1)
    pw = torch.tensor([p for _, p in w], device=DEV); ps = torch.tensor([p for _, p in s], device=DEV); pos = (pw[:, None] == ps[None]).float()
    lg = zw @ zs.T / TEMP; tot = 0; n = 0
    for L_, M in ((lg, pos), (lg.T, pos.T)):
        ok = M.sum(1) > 0
        if ok.any(): tot = tot + (torch.logsumexp(L_[ok], 1) - torch.logsumexp(L_[ok].masked_fill(M[ok] == 0, -1e9), 1)).mean(); n += 1
    return tot / n if n else torch.zeros((), device=DEV)


def train_step(opt, batch):
    ids, lab, kind, att, X, body, ap, nxt, act, msk, inf = batch
    out = B.run(ids, kind, att, X, hidden=True)
    lab = lab.to(DEV); tgt = lab[:, 1:]; m = tgt != -100; h_ = out.last_hidden_state[:, :-1][m]
    lq = -torch.log_softmax(B.base.lm_head(h_).float(), -1).gather(-1, tgt[m][:, None])[:, 0].mean()
    bi = torch.tensor(body, device=DEV)[:, None]; ap_t = torch.tensor(np.stack(ap), device=DEV)
    Hl = out.last_hidden_state[bi, ap_t].float(); Hf = out.hidden_states[B.L_FWD][bi, ap_t].float()
    act = torch.tensor(act, device=DEV); msk = torch.tensor(msk, device=DEV); nxt = torch.tensor(nxt, device=DEV)
    la = (((B.act(Hl) - act) ** 2).mean(-1) * msk).sum() / msk.sum().clamp(min=1)
    pr = B.fwd(Hf); lf = sum(((pr[..., sl] - nxt[..., sl]) ** 2).mean() for sl in TR.SL) / 3
    lal = align_loss(out.hidden_states[LA], inf) if USE_ALIGN else torch.zeros((), device=DEV)
    loss = la + lq + 2 * lf + LAM * lal; opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
    return la.item(), lq.item(), lf.item(), lal.item()


def train():
    params = [{"params": list(B.ad.parameters()) + list(B.fwd.parameters()) + list(B.act.parameters()), "lr": 3e-4},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 6e-5}]
    opt = torch.optim.AdamW(params, weight_decay=0.0); base = [g["lr"] for g in opt.param_groups]
    dl = torch.utils.data.DataLoader(Stream(), batch_size=None, num_workers=int(os.environ.get("NW", 24)), prefetch_factor=4)
    it = iter(dl); ls = []; B.m.train(); hist = []
    for st in range(STEPS):
        f = min(1, (st + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * st / STEPS)))
        for g, lr in zip(opt.param_groups, base): g["lr"] = lr * f
        ls.append(train_step(opt, next(it)))
        if st % 50 == 0: hist.append([st] + np.round(np.mean(ls[-50:], 0), 4).tolist()); log("step", hist[-1])
    del it, dl; B.m.eval(); return hist


# ---------------- tests ----------------
STEER = {}
def steer_hook(mod, inp, out):
    if not STEER: return out
    x = out[0] if isinstance(out, tuple) else out; v, mask = STEER["v"], STEER["m"]
    x = x + (mask[:, :x.shape[1], None] * v[None, None]).to(x.dtype)
    return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
B.layers[LA - 1].register_forward_hook(steer_hook)


@torch.no_grad()
def states(built, eps, pos, bs=24):
    out = []
    for k in range(0, len(built), bs):
        ids, lab, kind, att, X = collate(built[k:k + bs], eps[k:k + bs], B.tok.pad_token_id)
        o = B.run(ids, kind, att, X, hidden=True); h = o.hidden_states[LA]
        out += [h[b, pos[k + b]].float().cpu().numpy() for b in range(len(ids))]
    return np.stack(out)


def word_states(tpls):
    built, part = [], []
    for p in NONHAND:
        for t in tpls: built.append(B.tk.build([("t", t.format(PARTS[p]), False, "word")])); part.append(p)
    return states(built, [None] * len(built), [b[5]["word"] for b in built]), np.array(part)


def sensor_states(n):
    rng = np.random.default_rng(55); built, eps = [], []
    while len(eps) < n:
        e = TR.slim(P.make_episode(rng, "poke"))
        if e["info"]["success"] and e["info"]["part"] in NONHAND:
            eps.append(e); built.append(B.tk.build([("t", "Body:", False, None), ("steps", T + 1, {})]))
    pos = [b[4][peak_t(e)] + 1 for b, e in zip(built, eps)]
    return states(built, eps, pos), np.array([e["info"]["part"] for e in eps])


def centroids(Z, part):
    mu = Z.mean(0); C = np.stack([Z[part == p].mean(0) for p in NONHAND]) - mu
    return C, mu, float(np.median(np.linalg.norm(Z - mu, axis=1)))


def rsa_tests(Zw, pw, Zs, ps):
    from scipy.stats import spearmanr
    Cw, muw, _ = centroids(Zw, pw); Cs, mus, _ = centroids(Zs, ps)
    def rdm(C): Cn = C / np.linalg.norm(C, axis=1, keepdims=True); return 1 - Cn @ Cn.T
    iu = np.triu_indices(len(NONHAND), 1); Rw, Rs = rdm(Cw)[iu], rdm(Cs)[iu]; r = spearmanr(Rw, Rs)[0]
    rng = np.random.default_rng(0); null = []
    for _ in range(2000):
        pm = rng.permutation(len(NONHAND)); null.append(spearmanr(rdm(Cw[pm])[iu], Rs)[0])
    # cross-modal decoding: word states -> sensor centroids (cosine, each centred by own modality mean)
    Csn = Cs / np.linalg.norm(Cs, axis=1, keepdims=True); Wn = (Zw - muw); Wn /= np.linalg.norm(Wn, axis=1, keepdims=True)
    pred = np.array(NONHAND)[(Wn @ Csn.T).argmax(1)]; hit = pred == pw
    Cwn = Cw / np.linalg.norm(Cw, axis=1, keepdims=True); Sn = Zs - mus; Sn /= np.linalg.norm(Sn, axis=1, keepdims=True)
    hit2 = np.array(NONHAND)[(Sn @ Cwn.T).argmax(1)] == ps
    held_w = np.isin(pw, HELD); held_s = np.isin(ps, HELD)
    return dict(rsa=float(r), rsa_p=float((np.sum(np.array(null) >= r) + 1) / 2001),
                dec_w2s_trained=hit[~held_w].tolist(), dec_w2s_held=hit[held_w].tolist(), dec_s2w_trained=hit2[~held_s].tolist(), dec_s2w_held=hit2[held_s].tolist())


def tip_dist(sim, p):
    Pw, _ = H.taxel_world(sim.d); return min(float(np.linalg.norm(Pw[H.TPART == p] - sim.d.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right"))


def steer_episodes(vec):
    """poke_reach goal, no probe, 24 fixed start poses; vec (d,) added at skin tokens of steps 1-4 (or None)."""
    sims = []
    for s in POSE_SEEDS:
        q = start_pose(s); sim = E.Sim("mount", np.random.default_rng(s)); sim.d.qpos[H.JQ] = q; sim.d.qvel[:] = 0
        mujoco.mj_forward(E.M, sim.d); sim.ctrl = sim.grav_ctrl(q); sim.settle(); sims.append(sim)
    n = len(sims); S = [[s.sense()] for s in sims]; tc = [[] for _ in range(n)]
    for t in range(T):
        built = [B.tk.build([("t", "Goal: Touch where you were touched.\n", False, None), ("t", "Body:", False, None), ("steps", t + 1, {})]) for _ in range(n)]
        ids, lab, kind, att, X = collate(built, [dict(S=np.stack(x)) for x in S], B.tok.pad_token_id)
        if vec is not None:
            m = torch.zeros(ids.shape, device=DEV)
            for b, it in enumerate(built):
                for tt in (1, 2, 3, 4):
                    if tt < len(it[4]): m[b, it[4][tt] + 1] = 1
            STEER.update(v=torch.tensor(vec, device=DEV, dtype=torch.float32), m=m)
        with torch.no_grad():
            o = B.run(ids, kind, att, X); a = torch.tensor([it[3][-1] for it in built], device=DEV)
            act = B.act(o.last_hidden_state[torch.arange(n), a].float()).cpu().numpy() * TR.ASD + TR.AMU
        STEER.clear()
        for b, sim in enumerate(sims):
            c = np.clip(act[b], H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3); S[b].append(sim.step(c, t)); tc[b].append(E.touch_info(sim))
    return sims, tc


@torch.no_grad()
def answer_logp(vec, sents):
    """text-only neutral sentence + question; vec added at the sentence tokens; returns [n_sent, 14] log p of each part answer."""
    seqs, own = [], []
    for si, snt in enumerate(sents):
        for p in NONHAND:
            b = B.tk.build([("t", snt, False, "word"), ("t", "\nQ: Where were you touched?\nA:", False, None), ("t", " " + PARTS[p], True, None)]); seqs.append(b); own.append(si)
    ids, lab, kind, att, X = collate(seqs, [None] * len(seqs), B.tok.pad_token_id)
    if vec is not None:
        m = torch.zeros(ids.shape, device=DEV)
        for b, it in enumerate(seqs): m[b, :it[5]["word"] + 1] = 1
        STEER.update(v=torch.tensor(vec, device=DEV, dtype=torch.float32), m=m)
    o = B.run(ids, kind, att, X); STEER.clear()
    tgt = lab[:, 1:].to(DEV); mk = tgt != -100; h = o.last_hidden_state[:, :-1][mk]
    lp = torch.log_softmax(B.base.lm_head(h).float(), -1).gather(-1, tgt[mk][:, None])[:, 0]
    v = torch.zeros(len(seqs), device=DEV).index_add(0, mk.nonzero()[:, 0], lp).cpu().numpy()
    return v.reshape(len(sents), len(NONHAND))


def control_eval():
    res = {}
    for kind, kid in (("reach", 1), ("poke_reach", 2), ("grasp", 3)):
        scenes = [Q.make_scene(kind, np.random.default_rng([5151, kid, i]), diverse=True) for i in range(NEV)]
        sims = [s[0] for s in scenes]; infos = [s[1] for s in scenes]; S = [[s.sense()] for s in sims]; tc = [[] for _ in sims]; n = len(sims)
        for t in range(T):
            acts = []
            for k in range(0, n, 96):
                bs = range(k, min(n, k + 96))
                built = [B.tk.prompt(dict(kind=kind, info=infos[b], S=np.stack(S[b])), [], upto_t=t) for b in bs]
                ids, lab, kind_, att, X = TR.collate(built, [dict(S=np.stack(S[b])) for b in bs], B.tok.pad_token_id)
                with torch.no_grad():
                    o = B.run(ids, kind_, att, X); a = torch.tensor([it[3][-1] for it in built], device=DEV)
                    acts.append(B.act(o.last_hidden_state[torch.arange(len(built)), a].float()).cpu().numpy() * TR.ASD + TR.AMU)
            act = np.concatenate(acts)
            for b, sim in enumerate(sims):
                c = np.clip(act[b], H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3); S[b].append(sim.step(c, t)); tc[b].append(E.touch_info(sim))
        res[kind] = [bool(Q.success(kind, infos[b], sims[b], tc[b])) for b in range(n)]; log("CTRL-EVAL", kind, np.mean(res[kind]))
    return res


R = dict(arm=ARM, mix=MIX, held=[PARTS[p] for p in HELD]); RJ = f"{OUT}/unify_{ARM}.json"
def save(): json.dump(R, open(RJ, "w"))
if STEPS > 0 and os.environ.get("NOTRAIN") != "1":
    R["train"] = train(); save()
    torch.save(B.state_dict_small() | {"rn": B.ad.cap.item()}, f"{OUT}/ck_{ARM}.pt")
B.m.eval()
Zw, pw = word_states(TPL_EVAL); Zs, ps = sensor_states(60 if SMOKE else 600)
R["rsa"] = rsa_tests(Zw, pw, Zs, ps); log("RSA", R["rsa"]["rsa"], "dec held", np.mean(R["rsa"]["dec_w2s_held"]), "dec trained", np.mean(R["rsa"]["dec_w2s_trained"])); save()
Cw, _, scw = centroids(Zw, pw); Cs, _, scs = centroids(Zs, ps); d = Zs.shape[1]; rng = np.random.default_rng(123)
TG = ["head", "chest", "right shin", "left thigh", "belly", "left forearm", "right thigh", "left shin"]
def unit(v): return v / np.linalg.norm(v)
# (b2) sensor -> words
base = answer_logp(None, NEUTRAL); R["b2"] = dict(base=base.tolist())
for nm in TG:
    j = NONHAND.index(PARTS.index(nm))
    for cond, v in (("sensor", unit(Cs[j])), ("word", unit(Cw[j])), ("random", unit(rng.normal(size=d)))):
        for a in (1.5, 3.0):
            lp = answer_logp(v * scw * a, NEUTRAL); R["b2"][f"{nm}|{cond}|{a}"] = lp.tolist()
    log("b2", nm, "sensor3 hit", np.mean(np.array(R["b2"][f"{nm}|sensor|3.0"]).argmax(1) == j))
save()
# (b1) word -> body
R["b1"] = {}; sims, tc = steer_episodes(None)
R["b1"]["none"] = dict(dist={nm: [tip_dist(s, PARTS.index(nm)) for s in sims] for nm in TG},
                       touch={nm: [any(PARTS.index(nm) in x[0] for x in t[-6:]) for t in tc] for nm in TG})
for nm in TG:
    j = NONHAND.index(PARTS.index(nm)); p = PARTS.index(nm)
    for cond, v in (("word", unit(Cw[j])), ("sensor", unit(Cs[j])), ("random", unit(rng.normal(size=d)))):
        sims, tc = steer_episodes(v * scs * 1.5)
        R["b1"][f"{nm}|{cond}"] = dict(dist=[tip_dist(s, p) for s in sims], touch=[any(p in x[0] for x in t[-6:]) for t in tc])
    log("b1", nm, {c: round(100 * np.mean(R["b1"][f"{nm}|{c}"]["dist"]), 1) for c in ("word", "sensor", "random")}, "none", round(100 * np.mean(R["b1"]["none"]["dist"][nm]), 1))
    save()
R["control"] = control_eval(); save(); log("UNIFY DONE")
