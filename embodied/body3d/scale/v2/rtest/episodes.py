"""Episode generator. make_episode(rng, kind) -> dict(S[T+1,D], A[T,NJ], mask[T], kind, info, goal key)."""
import numpy as np, env as E, hum as H

T = E.T
KINDS = ["reach", "poke", "poke_reach", "balance", "grasp", "where"]
CODES = ["zor", "kip", "vel", "dax", "mup", "tib", "gon", "rex", "sul", "fip", "nax", "jot", "wem", "bry", "lug", "hosk", "pim"]
DIRS = ["forward", "backward", "left", "right"]


def pname(p, named): return H.PARTS[p] if named else CODES[p]


def other(h): return "left" if h == "right" else "right"


def pick_hand(rng, tp):
    lb = E.limb_of_part(tp); h = rng.choice(["left", "right"])
    if lb and lb[1] == "arm" and lb[0] == h: h = other(h)
    elif lb and lb[1] == "leg" and rng.uniform() < 0.8: h = lb[0]
    return str(h)


def babble(rng, sides=("left", "right")):
    """smooth random arm targets: returns function t -> q offsets"""
    ks = {s: rng.normal(0, 0.35, (3, len(E.ARM[s]))) for s in sides}
    def f(t, q):
        q = q.copy()
        for s in sides:
            x = t / T; w = ks[s][0] * np.sin(np.pi * x) + ks[s][1] * np.sin(2 * np.pi * x) * 0.5
            q[E.ARM[s]] = E.Q0[E.ARM[s]] + w
        return q
    return f


def _run(sim, T_, plan_fn, extra=None):
    S = [sim.sense()]; A = []; mask = []; touches = []
    for t in range(T_):
        q, m = plan_fn(t)
        c = sim.grav_ctrl(q); A.append(c.astype(np.float32)); mask.append(m)
        S.append(sim.step(c, t)); touches.append(E.touch_info(sim))
        if extra: extra(t)
    return np.stack(S), np.stack(A), np.array(mask, np.float32), touches


def ep_reach(rng):
    for _ in range(8):
        sim = E.Sim("mount", rng); ti = int(rng.integers(H.NT)); tp = int(H.TPART[ti]); hand = pick_hand(rng, tp)
        t0 = int(rng.integers(0, 3)); plan, err = E.plan_reach(sim, ti, hand, t0, t0 + int(rng.integers(8, 12)))
        if err < 0.01: break
    st = dict(q=sim.d.qpos[H.JQ].copy())
    def pf(t):
        st["q"] = plan.get(t, st["q"]); return st["q"], 1.0
    S, A, mk, tc = _run(sim, T, pf)
    hit = any(tp in x[0] for x in tc[-6:])
    return dict(S=S, A=A, mask=mk * hit, kind="reach", info=dict(taxel=ti, part=tp, hand=hand, success=hit, self=True,
                touched=_last_touched(tc, 0)))


def _last_touched(tc, idx):
    for x in reversed(tc):
        if x[idx]: return sorted(x[idx])
    return []


def ep_poke(rng):
    sim = E.Sim("mount", rng); ti = int(rng.integers(H.NT)); tp = int(H.TPART[ti]); t0 = int(rng.integers(2, 9)); t1 = t0 + int(rng.integers(3, 6))
    sim.probe_path = E.probe_path(sim, ti, t0, t1, depth=rng.uniform(0.005, 0.02)); bb = babble(rng) if rng.uniform() < 0.6 else None
    st = dict(q=sim.d.qpos[H.JQ].copy())
    def pf(t):
        return (bb(t, E.Q0) if bb else E.Q0.copy()), 0.0
    S, A, mk, tc = _run(sim, T, pf)
    poked = any(tp in x[1] for x in tc)
    return dict(S=S, A=A, mask=mk, kind="poke", info=dict(taxel=ti, part=tp, success=poked, self=False, t0=t0,
                touched=_last_touched(tc, 1)))


def ep_poke_reach(rng):
    for _ in range(8):
        sim = E.Sim("mount", rng); ti = int(rng.integers(H.NT)); tp = int(H.TPART[ti]); hand = pick_hand(rng, tp)
        if H.PARTS[tp].endswith("hand"): continue
        plan, err = E.plan_reach(sim, ti, hand, 7, 17)
        if err < 0.01: break
    sim.probe_path = E.probe_path(sim, ti, 1, 4, depth=0.015)
    st = dict(q=sim.d.qpos[H.JQ].copy())
    def pf(t):
        st["q"] = plan.get(t, st["q"]); return st["q"], float(t >= 5)
    S, A, mk, tc = _run(sim, T, pf)
    hit = any(tp in x[0] for x in tc[-5:]); poked = any(tp in x[1] for x in tc[:7])
    return dict(S=S, A=A, mask=mk * hit * poked, kind="poke_reach", info=dict(taxel=ti, part=tp, hand=hand, success=hit and poked, self=True))


def bal_ctrl(d, q0):
    q = q0.copy(); com = d.subtree_com[E.TORSO]; foot = 0.5 * (d.body("foot_left").xpos + d.body("foot_right").xpos) + [0.03, 0, 0]
    e = (com - foot)[:2]; v = d.subtree_linvel[E.TORSO][:2]; ux = 8 * e[0] + 2 * v[0]; uy = 8 * e[1] + 2 * v[1]
    for s in ("left", "right"):
        sg = 1 if s == "left" else -1
        q[E.JID[f"ankle_y_{s}"]] += ux; q[E.JID[f"hip_y_{s}"]] -= ux; q[E.JID[f"ankle_x_{s}"]] -= uy; q[E.JID[f"hip_x_{s}"]] += sg * uy
    return q


def ep_balance(rng):
    sim = E.Sim("free", rng); di = int(rng.integers(4)); ang = [0, np.pi, np.pi / 2, -np.pi / 2][di] + rng.normal(0, 0.15)
    F = rng.uniform(40, 150); tp = int(rng.integers(4, 9)); sim.push = (tp, tp + 1.5, F * np.array([np.cos(ang), np.sin(ang), 0]))
    ctl = rng.uniform() < 0.8
    def pf(t): return (bal_ctrl(sim.d, E.Q0) if ctl else E.Q0.copy()), float(ctl)
    S, A, mk, tc = _run(sim, T, pf)
    fell = bool(sim.d.qpos[2] < 0.95)
    return dict(S=S, A=A, mask=mk * (not fell), kind="balance", info=dict(dir=di, fell=fell, F=F, tp=tp, self=None, success=not fell))


def ep_grasp(rng):
    hand = str(rng.choice(["left", "right"])); s = 1 if hand == "left" else -1
    for _ in range(5):
        sim = E.Sim("mount", rng); pos = np.array([rng.uniform(0.25, 0.42), s * rng.uniform(0.0, 0.3), rng.uniform(0.85, 1.15)])
        sim.d.mocap_pos[E.PED_M] = pos; t0 = int(rng.integers(0, 3)); plan, err = E.plan_grasp(sim, hand, t0, t0 + 8, t0 + 10, 99)
        if err < 0.01: break
    st = dict(q=sim.d.qpos[H.JQ].copy())
    def pf(t):
        st["q"] = plan.get(t, st["q"]); return st["q"], 1.0
    S, A, mk, tc = _run(sim, T, pf)
    hg = {g for g1, g2, f in sim.last_pairs for g in (g1, g2) if E.PED_G in (g1, g2) and g in E.HAND_GEOMS[hand]}
    held = len(hg) >= 2
    return dict(S=S, A=A, mask=mk * held, kind="grasp", info=dict(hand=hand, success=held, pole=pos, self=None, part=H.PARTS.index(f"{hand} hand")))


def ep_where(rng):
    sim = E.Sim("mount", rng); bb = babble(rng); ti = int(rng.integers(H.NT)); P, N = H.taxel_world(sim.d)
    dist = -0.01 if rng.uniform() < 0.3 else rng.uniform(0.06, 0.15)
    E.place_ball(sim, P[ti] + N[ti] * (0.04 + dist), True); sim.d.qvel[E.BALLV:E.BALLV + 3] = rng.normal(0, 0.03, 3)
    def pf(t): return bb(t, E.Q0), 0.0
    S, A, mk, tc = _run(sim, T, pf)
    P, _ = H.taxel_world(sim.d); bp = sim.d.xpos[E.BALL]; dd = np.linalg.norm(P - bp, axis=1)
    near = int(H.TPART[dd.argmin()]); touching = any(len(x[2]) > 0 for x in tc[-3:])
    return dict(S=S, A=A, mask=mk, kind="where", info=dict(part=near, touching=touching, self=None, success=True, dist=float(dd.min())))


MAKERS = dict(reach=ep_reach, poke=ep_poke, poke_reach=ep_poke_reach, balance=ep_balance, grasp=ep_grasp, where=ep_where)


def make_episode(rng, kind=None):
    kind = kind or str(rng.choice(KINDS, p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]))
    return MAKERS[kind](rng)


def goal_text(e, named):
    k, i = e["kind"], e["info"]
    if k == "reach": return f"Touch your {pname(i['part'], named)}." if named else f"Touch {pname(i['part'], named)}."
    if k == "poke_reach": return "Touch where you were touched."
    if k == "balance": return "Keep your balance."
    if k == "grasp": return "Grab the handle."
    return ""


def questions(e, named):
    """list of (key, question, answer, candidates)"""
    k, i = e["kind"], e["info"]; P = [pname(p, named) for p in range(len(H.PARTS))]; Q = []
    hands = [pname(H.PARTS.index("left hand"), named), pname(H.PARTS.index("right hand"), named)]
    if k in ("reach", "poke", "poke_reach"):
        Q.append(("part", "Where were you touched?", P[i["part"]], P))
        if k != "poke_reach": Q.append(("self", "Who touched you?", "me" if i["self"] else "something else", ["me", "something else"]))
    if k in ("reach", "poke_reach", "grasp"):
        Q.append(("hand", "Which hand did you use?", hands[0] if i["hand"] == "left" else hands[1], hands))
    if k in ("reach", "poke_reach", "grasp"):
        Q.append(("done", "Did you succeed?", "yes" if i["success"] else "no", ["yes", "no"]))
    if k == "balance":
        Q.append(("dir", "Which way were you pushed?", DIRS[i["dir"]], DIRS))
        Q.append(("fell", "Did you fall?", "yes" if i["fell"] else "no", ["yes", "no"]))
    if k == "where":
        Q.append(("where", "Where is the ball?", f"near your {P[i['part']]}" if named else f"near {P[i['part']]}",
                  [f"near your {p}" if named else f"near {p}" for p in P]))
        Q.append(("btouch", "Is the ball touching you?", "yes" if i["touching"] else "no", ["yes", "no"]))
    return Q
