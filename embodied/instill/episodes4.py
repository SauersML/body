"""v4 episodes: tasks, scripted experts, questions (named and nameless twins).
Episode kinds: babble (random/sustained moves, self-touch babble, passive strokes), reach (touch a body landmark with a hand),
find (invisible object, sweep until contact), vreach (visible ball), tool (stick in hand, ball beyond reach),
wall (press, not too hard), brush (visible brush strokes the body or something else; proprioceptive bias on an arm)."""
import numpy as np
import sim4 as S

T = 20
KINDS = ["babble", "reach", "find", "vreach", "tool", "wall", "brush"]
KIND_P = [.27, .22, .08, .1, .1, .05, .18]
HANDN = ["left", "right"]
# nameless codes (single tokens, no body meaning)
ZONE = ["ka", "fu", "ti", "re", "mu", "ze", "pi", "lu", "da", "ko", "ve", "gi", "su", "ne", "bo", "fa", "ri", "tu"]
LMCODE = {n: f"spot {chr(65 + i // 10)}{i % 10}" for i, n in enumerate(S.LMN)}
CHAIN_N = [3, 1, 4, 2]  # nameless chain numbers for L arm, R arm, L leg, R leg


def d10(v, lo, hi): return str(int(np.clip(np.floor(10 * (v - lo) / (hi - lo)), 0, 9)))  # single digit 0-9


DIG = [str(i) for i in range(10)]
RY, RX = (-8.3, 8.6), (-5.0, 5.0)  # rest-pose body height, across-body
WY, WX, WZ = (-9., 9.), (-7., 7.), (-5., 7.)  # world ranges
FRONT_Z = S.N0[:, 2]


# ---------------- strokes ----------------
def geo_path(rng, s0, n, dmax=4.0):
    """samples along a geodesic from s0 to a random s1, n points."""
    c = np.where((S.GEO[s0] > 1.0) & (S.GEO[s0] < dmax))[0]
    if len(c) == 0: return np.full(n, s0)
    s1 = int(rng.choice(c)); D = S.GEO[s0, s1]; out = []
    for f in np.linspace(0, 1, n):
        out.append(int(np.argmin(np.abs(S.GEO[s0] - f * D) + np.abs(S.GEO[s1] - (1 - f) * D))))
    return np.array(out)


def rand_sample(rng, where=None):
    if where is None: return int(rng.integers(S.NS))
    return int(rng.choice(np.where(where)[0]))


def onoff(rng, n, mode):
    if mode == "cont": return np.ones(n, bool)
    if mode == "tap": return rng.random(n) < .55
    return np.zeros(n, bool)


# ---------------- experts ----------------
def greedy(env, score, actions=None, lookahead=True):
    """pick the action minimizing score(q_batch) after the move. score: [K,24] -> [K]."""
    acts = list(range(2 * len(S.CMD))) + [S.A_REST] if actions is None else actions
    Q = np.stack([env.next_q(a)[0] for a in acts]); sc = score(Q)
    return acts[int(np.argmin(sc))], float(sc.min())


def reach_score(hand, tgt):
    def f(Q):
        P = S.point_world(Q, [int(S.HT[hand][0]), tgt]); return np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    return f


def ball_score(hand, c):
    idx = np.where(np.isin(S.HARD_S, S.ARM_BONES[hand][2:]))[0][::4]
    def f(Q):
        P = S.point_world(Q, idx); return np.linalg.norm(P - c, axis=2).min(1)
    return f


def stick_tip_q(Q, hand, L):
    R, T_ = S.fk(Q); b = S.ARM_BONES[hand][2]; d = np.einsum("kij,j->ki", R[:, b], S.TAILS[b] - S.HEADS[b])
    base = T_[:, b] + 0.6 * d; return base + d / np.linalg.norm(d, axis=1, keepdims=True) * L


def ik_lattice(q0, score, rng, lo, hi, restarts=5, iters=40, lam=0.02):
    """coordinate search on the DQ lattice around q0 (so the plan is executable with motor tokens)."""
    C = np.array(S.CMD); best_q, best_s = q0.copy(), float(score(q0[None])[0])
    for r in range(restarts):
        q = q0.copy()
        if r > 0:
            m = rng.random(len(C)) < .35; q[C[m]] += S.DQ * rng.integers(-4, 5, m.sum()); q = np.clip(q, lo, hi)
        cur = float(score(q[None])[0]) + lam * np.abs(q - q0).sum()
        for it in range(iters):
            Q = np.repeat(q[None], 2 * len(C), 0); Q[np.arange(len(C)) * 2, C] += S.DQ; Q[np.arange(len(C)) * 2 + 1, C] -= S.DQ
            Q = np.clip(Q, lo, hi); sc = score(Q) + lam * np.abs(Q - q0).sum(1); i = int(sc.argmin())
            if sc[i] >= cur - 1e-4: break
            q, cur = Q[i], sc[i]
        s_ = float(score(q[None])[0])
        if s_ + lam * np.abs(q - q0).sum() < best_s + lam * np.abs(best_q - q0).sum(): best_q, best_s = q.copy(), s_
        if best_s < 0.15: break
    return best_q, best_s


def ik_cont(q0, score, rng, lo, hi, K=16, iters=60, lam=0.04, h=0.02):
    """batched finite-difference Adam over K restarts; cost = score + lam * travel in motor steps."""
    C = np.array(S.CMD); nC = len(C)
    Q = np.repeat(q0[None], K, 0)
    Q[1:, C] = np.clip(q0[C] + rng.normal(0, .7, (K - 1, nC)) * (rng.random((K - 1, nC)) < .5), lo[C], hi[C])
    m = np.zeros((K, nC)); v = np.zeros((K, nC)); lr = 0.12
    cost = lambda X: score(X) + lam * np.abs(X - q0)[:, C].sum(1) / S.DQ
    for it in range(iters):
        X = np.repeat(Q[:, None], 2 * nC, 1); ar = np.arange(nC)
        X[:, 2 * ar, C] += h; X[:, 2 * ar + 1, C] -= h
        f = cost(X.reshape(-1, S.NJ)).reshape(K, 2 * nC); g = (f[:, 0::2] - f[:, 1::2]) / (2 * h)
        m = .8 * m + .2 * g; v = .95 * v + .05 * g * g
        Q[:, C] = np.clip(Q[:, C] - lr * m / (np.sqrt(v) + 1e-4), lo[C], hi[C]); lr *= .97
    fc = cost(Q); i = int(fc.argmin())
    q = q0 + S.DQ * np.round((Q[i] - q0) / S.DQ); q = np.clip(q, lo, hi)
    return q


def plan(q0, score, rng, lo, hi):
    q = ik_cont(q0, score, rng, lo, hi)
    q2, s2 = ik_lattice(q, score, rng, lo, hi, restarts=1, iters=15)
    return q2, s2


def follow(env, mem, score):
    """move toward the planned posture one joint at a time (largest gap first); greedy when the plan is used up."""
    if "plan" not in mem: mem["plan"] = plan(env.q, score, env.rng, env.lo, env.hi)[0]
    d = mem["plan"] - env.q; d[S.FING] = 0; j = int(np.argmax(np.abs(d)))
    if abs(d[j]) > S.DQ / 2:
        a = 2 * S.CMD.index(j) + (0 if d[j] > 0 else 1)
        if mem.get("last_q") is not None and np.allclose(mem["last_q"], env.q) and mem.get("last_a") == a:  # blocked: replan once
            mem.pop("plan"); mem["stuck"] = mem.get("stuck", 0) + 1
            if mem["stuck"] > 2: return greedy(env, score)[0]
            return follow(env, mem, score)
        mem["last_q"], mem["last_a"] = env.q.copy(), a; return a
    return greedy(env, score)[0]


# ---------------- episode ----------------
def setup_env(st, rng):
    env = S.Env(st["q0"], rng, lo=st.get("lo"), hi=st.get("hi"))
    env.bias = st.get("bias", np.zeros(S.NJ)); env.ball = st.get("ball"); env.wall = st.get("wall")
    env.brush = st.get("brush"); env.other = st.get("other"); env.stick = st.get("stick")
    if st.get("ten") is not None: env.ten = st["ten"].copy()
    if st.get("stick") is not None:  # holding: fingers flexed
        env.ten[st["stick"]["hand"]] = [1, 0]; env.q[S.FING[st["stick"]["hand"]]] = 1.4
    return env


def make_setup(rng, kind):
    q0 = S.rand_q(rng); st = dict(kind=kind, q0=q0, g={})
    if kind == "reach":
        hand = int(rng.integers(2)); opts = [i for i, n in enumerate(S.LMN) if not (HANDN[hand] in n and any(w in n for w in ("hand", "wrist", "forearm", "elbow", "upper arm")))]
        li = int(rng.choice(opts)); st["g"] = dict(hand=hand, lm=S.LMN[li], tgt=int(S.LMI[li]))
    elif kind == "find":
        hand = int(rng.integers(2)); j = S.JNAME.index(f"{'LR'[hand]} shoulder flex"); dirn = 1 if q0[j] < 0.5 else -1
        st["g"] = dict(hand=hand, j=j, dir=dirn)
        q = q0.copy(); k = int(rng.integers(3, 10)); q[j] = np.clip(q[j] + dirn * S.DQ * k, S.TLO[j], S.THI[j])
        pts = S.point_world(q[None], np.where(np.isin(S.HARD_S, S.ARM_BONES[hand][1:]))[0])[0]
        c = pts[rng.integers(len(pts))] + rng.normal(0, .15, 3)
        st["ball"] = dict(c=c, r=rng.uniform(.5, .8), visible=False, p=rng.uniform(.6, 1.))
    elif kind in ("vreach", "tool"):
        hand = int(rng.integers(2)); qs = S.rand_q(rng); st["g"] = dict(hand=hand)
        if kind == "vreach":
            idx = np.where(np.isin(S.HARD_S, S.ARM_BONES[hand][2:]))[0]
            c = S.point_world(qs[None], [int(rng.choice(idx))])[0, 0] + rng.normal(0, .2, 3)
        else:
            L = float(rng.uniform(3, 7)); st["stick"] = dict(hand=hand, L=L); qs[S.FING[hand]] = 1.4
            c = stick_tip_q(qs[None], hand, L)[0] + rng.normal(0, .2, 3)
        if np.linalg.norm(S.point_world(q0[None], np.arange(S.NS))[0] - c, axis=1).min() < 1.0: return make_setup(rng, kind)
        st["ball"] = dict(c=c, r=rng.uniform(.4, .7), visible=True, p=1.0)
    elif kind == "wall":
        hand = 1; st["g"] = dict(hand=hand); P = S.point_world(q0[None], np.arange(S.NS))[0]
        st["wall"] = (2, float(P[:, 2].max() + rng.uniform(.3, 1.5)))
    elif kind == "brush":
        mode = rng.choice(["body", "world"], p=[.65, .35]); n = T
        arm = int(rng.integers(2)); bias = np.zeros(S.NJ)
        if rng.random() < .6:
            for jn in ("shoulder flex", "shoulder abduct", "elbow"): bias[S.JNAME.index(f"{'LR'[arm]} {jn}")] = rng.normal(0, .3)
        st["bias"] = bias; st["g"] = dict(arm=arm, mode=mode)
        if mode == "body":
            where = np.isin(S.HARD_S, S.ARM_BONES[arm][1:]) if rng.random() < .7 else None
            s0 = rand_sample(rng, where); path = geo_path(rng, s0, n, 2.5)
            on = onoff(rng, n, rng.choice(["cont", "tap"])); on[-1] = True
            st["brush"] = dict(felt=True, visible=True, s=path, on=on, seen_on=on.copy(), p=float(rng.uniform(.4, .9)))
        else:
            P = S.point_world(q0[None], [int(S.HT[arm][1])])[0, 0]
            start = P + rng.normal(0, 1, 3); start += np.sign(start - P) * 1.0
            path = start + np.cumsum(rng.normal(0, .15, (n, 3)), 0)
            st["brush"] = dict(felt=False, visible=True, path=path, s=np.zeros(n, int), on=np.zeros(n, bool),
                               seen_on=onoff(rng, n, rng.choice(["cont", "tap"])), p=0.)
            if rng.random() < .7:  # an unseen toucher on your body with its own timing
                where = np.isin(S.HARD_S, S.ARM_BONES[arm][1:]) if rng.random() < .6 else None
                on = onoff(rng, n, "tap"); on[-1] = rng.random() < .8
                st["other"] = dict(s=geo_path(rng, rand_sample(rng, where), n, 2.5), on=on, p=float(rng.uniform(.4, .9)), w=float(rng.uniform(.2, .6)))
    if kind == "babble":
        if rng.random() < .1:
            arm = int(rng.integers(2)); b = np.zeros(S.NJ)
            for jn in ("shoulder flex", "shoulder abduct", "elbow"): b[S.JNAME.index(f"{'LR'[arm]} {jn}")] = rng.normal(0, .3)
            st["bias"] = b
        sub = rng.choice(["random", "selftouch"], p=[.6, .4]); st["g"] = dict(sub=sub)
        if sub == "selftouch":
            hand = int(rng.integers(2)); ok = np.where(~S.TOUCH_EXCL[hand] & ~np.isin(S.HARD_S, S.ARM_BONES[hand][:1]))[0]
            st["g"].update(hand=hand, tgt=int(rng.choice(ok)))
        if rng.random() < .75:  # passive stroke by an unseen other, mostly ending on the body at the last step
            on = onoff(rng, T, rng.choice(["cont", "tap"])); t0 = int(rng.integers(0, T - 2)); on[:t0] = False
            on[-1] = rng.random() < .85
            st["other"] = dict(s=geo_path(rng, rand_sample(rng), T, 3.5), on=on, p=float(rng.uniform(.3, 1.)), w=float(rng.uniform(.2, .8)))
        if rng.random() < .15:  # a visible ball sitting nearby (bumps)
            c = S.point_world(q0[None], [int(rng.integers(S.NS))])[0, 0] + rng.normal(0, 1.2, 3)
            st["ball"] = dict(c=c, r=rng.uniform(.4, .8), visible=bool(rng.random() < .5), p=1.)
    return st


def expert_action(env, st, t, mem, rng):
    k, g = st["kind"], st["g"]
    if k == "reach" or (k == "babble" and g.get("sub") == "selftouch"):
        if k == "babble" and rng.random() < .25: return int(rng.integers(S.NA))
        if any(c["src"] == "self" and S.GEO[c["s"], g["tgt"]] < 1.0 for c in env.contacts()): return S.A_REST
        return follow(env, mem, reach_score(g["hand"], g["tgt"]))
    if k == "babble":
        if mem.get("left", 0) > 0: mem["left"] -= 1; return mem["a"]
        if rng.random() < .5: mem["a"] = int(rng.integers(2 * len(S.CMD))); mem["left"] = int(rng.integers(1, 4)); return mem["a"]
        return int(rng.choice(S.NA, p=np.r_[np.full(S.NA - 1, .85 / (S.NA - 1)), [.15]]))
    if k == "find":
        if any(c["src"] == "ball" for c in env.contacts()): return S.A_REST
        j = g["j"]
        if env.q[j] >= S.THI[j] - 1e-6: mem["d"] = -1
        if env.q[j] <= S.TLO[j] + 1e-6: mem["d"] = 1
        d = mem.get("d", g["dir"]); return 2 * S.CMD.index(j) + (0 if d > 0 else 1)
    if k == "vreach":
        b = env.ball; idx = np.where(np.isin(S.HARD_S, S.ARM_BONES[g["hand"]][2:]))[0]
        P = env.posed()[0]
        if np.linalg.norm(P[idx] - b["c"], axis=1).min() < b["r"] + 0.05: return S.A_REST
        return follow(env, mem, ball_score(g["hand"], b["c"]))
    if k == "tool":
        b = env.ball; L = env.stick["L"]; env.contacts()
        if np.linalg.norm(env._stip - b["c"]) < b["r"] + 0.05: return S.A_REST
        return follow(env, mem, lambda Q: np.linalg.norm(stick_tip_q(Q, g["hand"], L) - b["c"], axis=1))
    if k == "wall":
        C = [c for c in env.contacts() if c["src"] == "wall"]
        if C and C[0]["p"] >= .5: return S.A_REST if C[0]["p"] <= .8 else 2 * S.CMD.index(S.JNAME.index("R shoulder flex")) + 1
        idx = np.where(np.isin(S.HARD_S, S.ARM_BONES[1][2:]))[0][::3]; wz = env.wall[1]
        return follow(env, mem, lambda Q: np.maximum(0, wz + 0.4 - S.point_world(Q, idx)[:, :, 2].max(1)))
    if k == "brush":
        return S.A_REST if rng.random() < .8 else int(rng.integers(S.NA))
    return S.A_REST


def success(st, info):
    k, g, C = st["kind"], st["g"], info["contacts"]
    if k == "reach": return any(c["src"] == "self" and c["by"] == g["hand"] and S.GEO[c["s"], g["tgt"]] < 1.0 for c in C)
    if k == "find": return any(c["src"] == "ball" for c in C)
    if k == "vreach": return info["ball_hand_d"] < st["ball"]["r"] + 0.1
    if k == "tool": return info["stick_ball_d"] < st["ball"]["r"] + 0.1
    if k == "wall": return any(c["src"] == "wall" and .45 <= c["p"] <= .85 for c in C)
    return True


def summarize(env, st):
    C = env.contacts(); P, Nn, R, T_, _ = env.posed()
    info = dict(q=env.q.copy(), contacts=C, hands=np.stack([P[S.HT[a][0]] for a in range(2)]), palms=np.stack([P[S.HT[a][1]] for a in range(2)]))
    if st.get("ball") is not None and st["kind"] == "vreach":
        idx = np.where(np.isin(S.HARD_S, S.ARM_BONES[st["g"]["hand"]][2:]))[0]; info["ball_hand_d"] = float(np.linalg.norm(P[idx] - st["ball"]["c"], axis=1).min())
    if st.get("stick") is not None: info["stick_ball_d"] = float(np.linalg.norm(env._stip - st["ball"]["c"])) if st.get("ball") else 9; info["stip"] = env._stip.copy()
    main = [c for c in C if c["src"] in ("self", "other", "brush", "ball", "wall", "stickhand")]
    vib = [c for c in C if c["src"] == "vib"]
    if main:
        c = max(main, key=lambda c: c["p"] + (0.5 if c["src"] == "self" else 0)); s = c["s"]
        info.update(s=s, src=c["src"], p=c["p"], world=P[s].copy(), part=int(S.PART[s]))
    else: info["s"] = None
    if vib: info["vib"] = vib[0]
    return info


def rollout(st, rng, policy=None, T_=T):
    """policy(env, t, S_list) -> action, or None for the expert."""
    env = setup_env(st, np.random.default_rng(int(rng.integers(1 << 30)))); Sx = [env.sense()]; A = []; mem = {}
    for t in range(T_ - 1):
        a = expert_action(env, st, t, mem, rng) if policy is None else policy(env, t, Sx)
        env.step(int(a)); A.append(int(a)); Sx.append(env.sense())
    return dict(S=np.stack(Sx), A=A, info=summarize(env, st), env=env)


def make_episode(rng, kind=None, max_tries=6):
    kind = kind or str(rng.choice(KINDS, p=KIND_P))
    for _ in range(max_tries):
        st = make_setup(rng, kind); ep = rollout(st, rng)
        if kind in ("babble", "brush") or success(st, ep["info"]):
            ep.update(kind=kind, st=st, act_label=kind in ("reach", "find", "vreach", "tool", "wall")); return ep
    return make_episode(rng, "babble")


# ---------------- language ----------------
def goal_text(st, named):
    k, g = st["kind"], st["g"]
    if k == "reach": return f"Touch {g['lm']} with your {HANDN[g['hand']]} hand." if named else f"Touch {LMCODE[g['lm']]} with end {g['hand'] + 1}."
    if k == "find": return f"Find the object with your {HANDN[g['hand']]} hand by touch." if named else f"Find the object with end {g['hand'] + 1} by contact."
    if k == "vreach": return f"Touch the ball with your {HANDN[g['hand']]} hand." if named else f"Touch the ball with end {g['hand'] + 1}."
    if k == "tool": return "Touch the ball with the stick."
    if k == "wall": return "Press the wall, but not too hard."
    return ""


def questions(info, st, named):
    """list of (key, question, answer, candidates). Digit answers are single tokens 0-9."""
    qa = []; w = "touch" if named else "contact"
    for a in range(2):
        h = info["hands"][a]; nm = f"your {HANDN[a]} hand" if named else f"end {a + 1}"
        qa += [(f"h{a}y", f"How high is {nm}, from 0 to 9?", d10(h[1], *WY), DIG), (f"h{a}x", f"How far to the left is {nm}, from 0 to 9?", d10(h[0], *WX), DIG),
               (f"h{a}z", f"How far forward is {nm}, from 0 to 9?", d10(h[2], *WZ), DIG)]
    s = info["s"]
    if s is None:
        qa.append(("part", "Where are you being touched?" if named else "Which zone has the contact?", "nowhere" if named else "none",
                   (S.PARTS + ["nowhere"]) if named else (ZONE + ["none"])))
    else:
        r = S.P0[s]; pt = int(S.PART[s])
        qa.append(("part", "Where are you being touched?" if named else "Which zone has the contact?", S.PARTS[pt] if named else ZONE[pt],
                   (S.PARTS + ["nowhere"]) if named else (ZONE + ["none"])))
        qa += [("bu", "How high up your body is the touch, from the soles of your feet (0) to the top of your head (9)?" if named else "What is coordinate U of the contact, from 0 to 9?", d10(r[1], *RY), DIG),
               ("bv", "How far across your body is the touch, from your right fingertips (0) to your left fingertips (9)?" if named else "What is coordinate V of the contact, from 0 to 9?", d10(r[0], *RX), DIG),
               ("fb", "Is the touch on your front or your back?" if named else "Is the contact on face A or face B?",
                ("front" if FRONT_Z[s] > 0 else "back") if named else ("A" if FRONT_Z[s] > 0 else "B"), ["front", "back"] if named else ["A", "B"]),
               ("wy", f"How high is the {w}, from 0 to 9?", d10(info["world"][1], *WY), DIG),
               ("wx", f"How far to the left is the {w}, from 0 to 9?", d10(info["world"][0], *WX), DIG),
               ("wz", f"How far forward is the {w}, from 0 to 9?", d10(info["world"][2], *WZ), DIG),
               ("press", f"How hard is the {w}, from 0 to 9?", d10(info["p"], 0, 1.0001), DIG)]
        li = int(S.LIMB[s])
        if li >= 0:
            if named:
                q = (f"How far along your {['left', 'right'][li % 2]} {['arm', 'leg'][li // 2]} is the touch, from the "
                     f"{['shoulder', 'hip'][li // 2]} (0) to the {['fingertips', 'toes'][li // 2]} (9)?")
            else: q = f"How far along chain {CHAIN_N[li]} is the contact, from 0 to 9?"
            qa.append(("arc", q, d10(S.ARC[s], 0, 1.0001), DIG))
        if info["src"] in ("self", "other", "brush"):
            qa.append(("self", f"Did you cause this {w} yourself?", "yes" if info["src"] == "self" else "no", ["yes", "no"]))
    if "vib" in info:
        qa.append(("stick", "How far out along the stick is the touch, from 0 to 9?", d10(info["vib"]["pos"], 0, 8), DIG))
        qa += [("sty", f"How high is the {w} at the stick, from 0 to 9?", d10(info["vib"]["tip"][1], *WY), DIG),
               ("stx", f"How far to the left is the {w} at the stick, from 0 to 9?", d10(info["vib"]["tip"][0], *WX), DIG)]
    return qa


QW = {"part": 3, "bu": 2, "bv": 2, "fb": 1, "wy": 1.5, "wx": 1.5, "wz": 1, "press": .7, "arc": 3, "self": 2, "stick": 3, "sty": 1, "stx": 1}


def pick_questions(qa, st, rng, n=4):
    w = np.array([QW.get(k, .5) for k, *_ in qa], float)
    if st["kind"] == "brush":  # hand position matters here (proprioceptive bias)
        a = st["g"]["arm"]; w *= np.array([4. if k.startswith(f"h{a}") else 1. for k, *_ in qa])
    idx = rng.choice(len(qa), size=min(n, len(qa)), replace=False, p=w / w.sum()); return [qa[i] for i in idx]
