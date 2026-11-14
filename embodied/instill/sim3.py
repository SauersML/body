"""v3 closed-loop body. 2D torso + two 3-segment arms (body.fk).
SKIN: each segment is a resistive strip read at its 4 corners (24 channels). A contact at (u along, w across)
with pressure p and contact width c adds p*(0.4+3c)*mean_falloff to each corner channel, falloff = 1/(1+d^2/0.35),
averaged over the contact's extent. Place, pressure and width must all be inferred from the 4 numbers.
PROPRIO: sin/cos of 6 joints + 6 joint velocities. Sensor vector = 18 + 24 = 42.
ACTIONS (one motor token per step): 12 joint moves (+-0.3 rad), rest, dim-left-skin, dim-right-skin, undim.
WORLD: external moving disc (passive touch), static object (find task), a wall that blocks the arm
(pushing into the wall raises pressure), and the other hand (self-touch gives touch on both)."""
import numpy as np
from body import fk, SEG_LEN

ARM_LEN = sum(SEG_LEN); BOUND = [SEG_LEN[0], SEG_LEN[0] + SEG_LEN[1]]
PARTS = ["left upper arm", "left forearm", "left hand", "right upper arm", "right forearm", "right hand"]
QLO = np.array([-2.3, -2.0, -1.2] * 2); QHI = -QLO
DQ = 0.3; NA = 16; SENSOR_DIM = 38  # 12 sin/cos + 6 vel + 4 tendon tensions + 16 skin
ACT_NAMES = [f"j{j}{s}" for j in range(6) for s in "+-"] + ["rest", "dimL", "dimR", "undim"]
EPOS = np.linspace(0, 1.15, 4)  # electrode arc positions along each arm sheet (not at the elbow/wrist)
ELEC = [(e, w) for e in EPOS for w in (0., 1.)]  # 8 edge electrodes per arm, 16 total
FINGER = 0.08


def seg_pts(q):
    p = fk(q); return [(p[a, s], p[a, s + 1]) for a in range(2) for s in range(3)]  # index 3a+s


def closest(P, A, B):
    AB = B - A; u = float(np.clip((P - A) @ AB / (AB @ AB), 0, 1)); C = A + u * AB
    side = 0.25 if AB[0] * (P - A)[1] - AB[1] * (P - A)[0] > 0 else 0.75
    return u, float(np.linalg.norm(P - C)), side


def seg_of(arc): return int(arc > BOUND[0]) + int(arc > BOUND[1])


def arc_of(seg, u): return sum(SEG_LEN[:seg % 3]) + u * SEG_LEN[seg % 3]


def channels(contacts, gain):
    """Each arm is ONE connected resistive sheet; a touch spreads over all 8 of that arm's electrodes."""
    v = np.zeros(16)
    for c in contacts:
        arm = c["seg"] // 3; s0 = arc_of(c["seg"], c["u"]); ss = s0 + np.linspace(-.5, .5, 5) * c["width"]
        for k, (e, w) in enumerate(ELEC):
            f = np.mean(1 / (1 + ((ss - e) ** 2 + .15 * (c["w"] - w) ** 2) / 0.12))
            v[8 * arm + k] += gain[arm] * c["p"] * (0.4 + 3 * c["width"]) * f
    return v


class Env:
    def __init__(s, q0, rng, ext=None, wall=None, obj=None):
        s.q = np.array(q0, float); s.dq = np.zeros(6); s.t = 0; s.rng = rng
        s.ext, s.wall, s.obj = ext, wall, obj; s.push = 0; s.gain = np.ones(2); s.ten = np.zeros((2, 2))  # [arm, flexor/extensor]

    def disc_contacts(s, c0, r, p, src):
        out = []
        for i, (A, B) in enumerate(seg_pts(s.q)):
            u, d, w = closest(c0, A, B)
            if d < r: out.append(dict(seg=i, u=u, w=w, p=p * (r - d) / r, width=2 * np.sqrt(r * r - d * d), src=src))
        return out

    def contacts(s):
        C = []; segs = seg_pts(s.q)
        if s.ext is not None:
            e = s.ext; C += s.disc_contacts(e["c"] + e["v"] * (s.t - e["t_end"]), e["r"], e["p"], "other")
        if s.obj is not None: C += s.disc_contacts(s.obj["c"], s.obj["r"], s.obj["p"], "obj")
        for a in range(2):  # self-touch: fingertip of arm a on the other arm
            tip = segs[3 * a + 2][1]
            for i in range(3 * (1 - a), 3 * (1 - a) + 3):
                u, d, w = closest(tip, *segs[i])
                if d < FINGER:
                    p = min(1., (FINGER - d) / FINGER * (.6 + .6 * s.ten[a, 0])); wd = 2 * np.sqrt(FINGER ** 2 - d * d)
                    C += [dict(seg=i, u=u, w=w, p=p, width=wd, src="self"), dict(seg=3 * a + 2, u=.9, w=.5, p=p, width=wd, src="self")]
        if s.wall is not None:
            best = None
            for i, (A, B) in enumerate(segs):
                for u in (0, .5, 1):
                    x = (A + u * (B - A))[0]
                    if x > s.wall - 0.04 and (best is None or x > best[0]): best = (x, i, u)
            if best: C.append(dict(seg=best[1], u=best[2], w=.25, p=min(1., .3 + .35 * s.push), width=.05, src="wall"))
        return C

    def max_x(s, q):
        p = fk(q); mid = (p[:, :-1] + p[:, 1:]) / 2; return max(p[:, 1:, 0].max(), mid[:, :, 0].max())

    def sense(s):
        tch = channels(s.contacts(), s.gain) + s.rng.normal(0, .01, 16)
        return np.concatenate([np.sin(s.q), np.cos(s.q), s.dq / DQ, s.ten.ravel(), tch]).astype(np.float32)

    def next_q(s, a):
        """Joint target after action a (no wall). Hands: actions pull tendons; the finger follows mechanically."""
        q = s.q.copy(); ten = s.ten.copy()
        if a < 12 and a // 2 not in (2, 5):
            j = a // 2; q[j] = np.clip(q[j] + (DQ if a % 2 == 0 else -DQ), QLO[j], QHI[j])
        elif a < 12:
            arm = a // 6; f, e = (0, 1) if a % 2 == 0 else (1, 0)
            ten[arm, f] = min(1, ten[arm, f] + .35); ten[arm, e] = max(0, ten[arm, e] - .2)
        for arm in range(2):  # passive mechanics: wrist relaxes toward the tendon balance
            j = 3 * arm + 2; q[j] += .6 * (np.clip(1.4 * (ten[arm, 0] - ten[arm, 1]), QLO[j], QHI[j]) - q[j])
        return q, ten

    def step(s, a):
        q0 = s.q.copy()
        q, ten = s.next_q(a); s.ten = ten
        if s.wall is not None and s.max_x(q) > s.wall:  # blocked: the arm stops at the wall, the push turns into pressure
            s.push += 1
            for fr in (.75, .5, .25):
                if s.max_x(s.q + fr * (q - s.q)) <= s.wall: s.q = s.q + fr * (q - s.q); break
        else:
            s.q = q
            if s.push: s.push = 0 if s.max_x(q) < s.wall - .04 else max(s.push - 1, 0)
        if a == 13: s.gain[0] = .3
        elif a == 14: s.gain[1] = .3
        elif a == 15: s.gain[:] = 1
        s.dq = s.q - q0; s.t += 1


def tip(q, a): return fk(q)[a, 3]


def landmark_point(q, arm, arc):
    p = fk(q)[arm]; acc = 0
    for s_ in range(3):
        if arc <= acc + SEG_LEN[s_] + 1e-9: t = (arc - acc) / SEG_LEN[s_]; return p[s_] * (1 - t) + p[s_ + 1] * t
        acc += SEG_LEN[s_]
    return p[3]


# ---------------- experts ----------------
def expert_reach(env, arm, tarm, tarc):
    tgt = landmark_point(env.q, tarm, tarc); best, bd = 12, np.linalg.norm(tip(env.next_q(12)[0], arm) - tgt)
    if bd < FINGER * .6: return 12
    for a in range(6 * arm, 6 * arm + 6):
        q = env.next_q(a)[0]
        d = np.linalg.norm(tip(q, arm) - tgt)
        if d < bd - 1e-6: best, bd = a, d
    return best


def expert_find(env, state):
    if any(c["src"] == "obj" for c in env.contacts()): return 12
    if env.q[3] >= QHI[3] - 1e-6: state["dir"] = -1
    if env.q[3] <= QLO[3] + 1e-6: state["dir"] = 1
    return 6 if state["dir"] > 0 else 7  # sweep the right shoulder


def expert_wall(env):
    C = [c for c in env.contacts() if c["src"] == "wall"]
    if C and C[0]["p"] >= .5: return 12 if C[0]["p"] <= .8 else 7
    best, bx = 6, -9  # move the joint that brings the right arm furthest right (the wall blocks it)
    for a in range(6, 12):
        q = env.next_q(a)[0]
        x = env.max_x(q)
        if x > bx: best, bx = a, x
    return best


LANDMARKS = {"elbow": BOUND[0], "wrist": BOUND[1], "forearm": (BOUND[0] + BOUND[1]) / 2, "upper arm": .25}
NAMELESS_LM = {"elbow": "the first bend", "wrist": "the second bend", "forearm": "the middle piece", "upper arm": "the first piece"}


def goal_text(kind, g, named):
    if kind == "reach":
        if named: return f"Touch your {['left','right'][g['tarm']]} {g['lm']} with your {['left','right'][g['arm']]} hand."
        return f"Touch {NAMELESS_LM[g['lm']]} of line {g['tarm']+1} with the end of line {g['arm']+1}."
    if kind == "find": return "Find the object by touch."
    if kind == "wall": return "Press the wall, but not too hard."
    return ""


def rand_q(rng): return rng.uniform(QLO, QHI)


def make_episode(rng, T=10, kind=None, policy=None):
    """Returns dict(S [T,42], A [T-1], act_label bool, goal=(kind,g), info). policy(env, t) overrides the expert (closed loop)."""
    kind = kind or rng.choice(["babble", "babble", "reach", "find", "wall"])
    g = {}; q0 = rand_q(rng); ext = wall = obj = None; st = {"dir": 1}
    if kind == "reach":
        g = dict(arm=int(rng.integers(2)), lm=str(rng.choice(list(LANDMARKS)))); g["tarm"] = 1 - g["arm"]
    if kind == "find":
        q0[3] = rng.uniform(QLO[3], 0)  # sweep starts low; object sits on the sweep path
        qq = q0.copy(); qq[3] = rng.uniform(q0[3] + .6, QHI[3]); pt = landmark_point(qq, 1, rng.uniform(.3, ARM_LEN))
        obj = dict(c=pt + rng.normal(0, .03, 2), r=rng.uniform(.08, .15), p=rng.uniform(.6, 1.))
        if any(np.linalg.norm(pt - landmark_point(q0, 1, a)) < .2 for a in np.linspace(0, ARM_LEN, 8)): return make_episode(rng, T, kind, policy)
    if kind == "wall": wall = Env(q0, rng).max_x(q0) + rng.uniform(.05, .35)
    reach_babble = kind == "babble" and rng.random() < .55
    if reach_babble: g = dict(arm=int(rng.integers(2)), lm=None, tarc=rng.uniform(.1, ARM_LEN)); g["tarm"] = 1 - g["arm"]
    env = Env(q0, rng, wall=wall, obj=obj)
    S, A = [env.sense()], []
    for t in range(T - 1):
        if policy is not None: a = policy(env, t, S)
        elif kind == "reach": a = expert_reach(env, g["arm"], g["tarm"], LANDMARKS[g["lm"]])
        elif kind == "find": a = expert_find(env, st)
        elif kind == "wall": a = expert_wall(env)
        elif reach_babble: a = expert_reach(env, g["arm"], g["tarm"], g["tarc"]) if rng.random() < .8 else int(rng.integers(NA))
        else: a = int(rng.choice(NA, p=np.r_[np.full(12, .7 / 12), [.15], np.full(3, .05)]))
        env.step(a); A.append(a)
        if t == T - 2 and kind == "babble" and rng.random() < (.3 if reach_babble else .8):  # passive external stroke ending on a random skin point now
            i = int(rng.integers(6)); u = rng.uniform(0, 1); A_, B_ = seg_pts(env.q)[i]; pt = A_ + u * (B_ - A_)
            r = rng.uniform(.06, .16); nrm = np.array([-(B_ - A_)[1], (B_ - A_)[0]]); nrm /= np.linalg.norm(nrm)
            env.ext = dict(c=pt + nrm * rng.uniform(-.6, .6) * r, v=rng.normal(0, .06, 2), r=r, p=rng.uniform(.4, 1.), t_end=T - 1)
            S = [None] * len(S)  # recompute earlier sensations with the stroke present (actions did not depend on it)
        S.append(env.sense())
    if S[0] is None:  # replay with the stroke
        e2 = Env(q0, np.random.default_rng(int(rng.integers(1 << 30))), ext=env.ext); S = [e2.sense()]
        for a in A: e2.step(a); S.append(e2.sense())
        env = e2
    info = summarize(env)
    if policy is None and kind in ("reach", "find", "wall") and not success(kind, info): return make_episode(rng, T, kind, policy)
    return dict(S=np.stack(S), A=A, kind=kind, g=g, info=info, env=env, S0q=q0, act_label=kind in ("reach", "find", "wall"))


def success(kind, info):
    C = info["contacts"]
    if kind == "reach": return any(c["src"] == "self" for c in C)
    if kind == "find": return any(c["src"] == "obj" for c in C)
    if kind == "wall": return any(c["src"] == "wall" and .45 <= c["p"] <= .85 for c in C)


def summarize(env):
    p = fk(env.q); C = env.contacts(); info = dict(q=env.q.copy(), lh_y=p[0, 3, 1], rh_y=p[1, 3, 1], contacts=C,
                                                   crossed=bool(p[0, 3, 0] > 0 or p[1, 3, 0] < 0), gain=env.gain.copy())
    if C:
        c = max(C, key=lambda c: c["p"] * (1.01 if c["seg"] % 3 != 2 else 1))  # strongest; ties go to the touched (non-hand) site
        A_, B_ = seg_pts(env.q)[c["seg"]]; xy = A_ + c["u"] * (B_ - A_)
        info.update(arm=c["seg"] // 3, seg=c["seg"] % 3, arc=arc_of(c["seg"], c["u"]), x=float(xy[0]), y=float(xy[1]),
                    p=c["p"], src=c["src"])
    else: info["arm"] = None
    return info


d10 = lambda v, lo, hi: str(int(np.clip(round(10 * (v - lo) / (hi - lo)), 0, 10)))
DIG = [str(i) for i in range(11)]
YLO, YHI, XLO, XHI = -0.7, 1.7, -1.5, 1.5


def questions(info, named, kind="babble"):
    a = info["arm"]; qa = []
    if named:
        qa.append(("part", "Where are you being touched?", "nowhere" if a is None else PARTS[3 * a + info["seg"]], PARTS + ["nowhere"]))
        qa.append(("lh", "How high is your left hand, from 0 to 10?", d10(info["lh_y"], YLO, YHI), DIG))
        qa.append(("rh", "How high is your right hand, from 0 to 10?", d10(info["rh_y"], YLO, YHI), DIG))
    else:
        qa.append(("line", "Which line has the contact: 1, 2, or 0 for none?", "0" if a is None else str(a + 1), ["0", "1", "2"]))
        qa.append(("lh", "How high is the end of line 1, from 0 to 10?", d10(info["lh_y"], YLO, YHI), DIG))
        qa.append(("rh", "How high is the end of line 2, from 0 to 10?", d10(info["rh_y"], YLO, YHI), DIG))
    if a is None: return qa
    w = "touch" if named else "contact"
    qa += [("pos", "How far along your arm is the touch, from shoulder (0) to fingertips (10)?" if named else
            "Where along the line is the contact, from 0 to 10?", d10(info["arc"], 0, ARM_LEN), DIG),
           ("y", f"How high is the {w}, from 0 to 10?", d10(info["y"], YLO, YHI), DIG),
           ("x", f"How far to the right is the {w}, from 0 to 10?", d10(info["x"], XLO, XHI), DIG),
           ("press", f"How hard is the {w}, from 0 to 10?", d10(info["p"], 0, 1), DIG)]
    if kind == "babble" and info["src"] in ("self", "other"):
        qa.append(("self", f"Did you cause this {w} yourself?", "yes" if info["src"] == "self" else "no", ["yes", "no"]))
    if named:
        side = ["left", "right"][a]
        qa += [("elbow", f"Is the touch near your {side} elbow?", "yes" if abs(info["arc"] - BOUND[0]) < .12 else "no", ["yes", "no"]),
               ("wrist", f"Is the touch near your {side} wrist?", "yes" if abs(info["arc"] - BOUND[1]) < .12 else "no", ["yes", "no"])]
    return qa
