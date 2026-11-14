"""Episodes on the MuJoCo humanoid: sim wrapper, expert controllers, questions."""
import numpy as np, mujoco
import hum as H

M = H.M; NJ = H.NJ; T = 20; SUB = 25  # 0.1 s per step
PROP_DIM = 3 * NJ + 10; SKIN_DIM = H.NT; VIS_DIM = 11; SENSOR_DIM = PROP_DIM + SKIN_DIM + VIS_DIM
JID = {n: i for i, n in enumerate(H.JN)}
BALL = M.body("ball").id; BALLQ = M.jnt_qposadr[M.joint("ball").id]; BALLV = M.jnt_dofadr[M.joint("ball").id]
PROBE_M = M.body_mocapid[M.body("probe").id]; PED_M = M.body_mocapid[M.body("pedestal").id]
TORSO = M.body("torso").id; BALL_G = M.geom("ball").id; PROBE_G = M.geom("probe").id; PED_G = M.geom("pedestal").id; FLOOR_G = M.geom("floor").id
KPV = M.actuator_gainprm[:, 0].copy()
def J(prefix): return [JID[n] for n in H.JN if n.startswith(prefix)]
ARM = {s: [JID[n] for n in H.JN if n.endswith(s) and n.split("_")[0] in ("shoulder", "elbow", "wrist")] for s in ("left", "right")}
FING = {s: [JID[n] for n in H.JN if n.endswith(s) and n[:-len(s) - 1].rstrip("12") in ("index", "middle", "thumb")] for s in ("left", "right")}
LEG = {s: [JID[n] for n in H.JN if n.endswith(s) and n.split("_")[0] in ("hip", "knee", "ankle")] for s in ("left", "right")}
ABD = J("abdomen"); NECK = J("neck")
HAND_GEOMS = {s: [g for g in H.BODY_GEOMS if H.geom_part(M.geom(g).name) == f"{s} hand"] for s in ("left", "right")}

Q0 = np.zeros(NJ)
for s in ("left", "right"):
    Q0[JID[f"shoulder_x_{s}"]] = 0.15; Q0[JID[f"shoulder_y_{s}"]] = -0.3; Q0[JID[f"elbow_{s}"]] = -0.9
    for n in ("index1", "index2", "middle1", "middle2"): Q0[JID[f"{n}_{s}"]] = 0.3
    Q0[JID[f"thumb1_{s}"]] = 0.3
POINT = {"index1": 0.0, "index2": 0.0, "middle1": 1.4, "middle2": 1.4, "thumb1": 0.9, "thumb2": 0.8}
OPEN = {"index1": 0.1, "index2": 0.1, "middle1": 0.1, "middle2": 0.1, "thumb1": -0.2, "thumb2": 0.1}
CLOSE = {"index1": 1.4, "index2": 1.4, "middle1": 1.4, "middle2": 1.4, "thumb1": 0.2, "thumb2": 0.3}
def handpose(q, s, pose):
    for k, v in pose.items(): q[JID[f"{k}_{s}"]] = v
    return q


def quat_z_to(n):
    z = np.array([0, 0, 1.0]); n = n / np.linalg.norm(n); v = np.cross(z, n); c = z @ n
    if np.linalg.norm(v) < 1e-8: return np.array([1, 0, 0, 0.]) if c > 0 else np.array([0, 1, 0, 0.])
    q = np.concatenate([[1 + c], v]); return q / np.linalg.norm(q)


class Sim:
    def __init__(s, mode, rng):
        s.m = M; s.d = mujoco.MjData(M); s.k = mujoco.MjData(M); s.mode = mode; s.rng = rng
        d = s.d; mujoco.mj_resetData(M, d); d.qpos[H.JQ] = Q0; d.eq_active[0] = mode == "mount"
        d.mocap_pos[PROBE_M] = [0, 3, 1]; d.mocap_pos[PED_M] = [0, -3, 0]; d.qpos[BALLQ:BALLQ + 3] = [0, -3, 0.04]
        s.ctrl = Q0.copy(); s.push = None; s.probe_path = None; s.ball_float = False
        mujoco.mj_forward(M, d); s.settle()

    def settle(s, n=150):
        for _ in range(n): s.d.ctrl[:] = s.ctrl; mujoco.mj_step(M, s.d)
        s.d.qvel[:] *= 0

    def grav_ctrl(s, qdes):
        """position targets that hold qdes against gravity (bias torque feed-forward)."""
        return qdes + s.d.qfrc_bias[H.JV] / KPV

    def step(s, ctrl, t):
        s.ctrl = np.clip(ctrl, H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3); d = s.d; amax = np.zeros(H.NT); pairs = []
        for i in range(SUB):
            d.ctrl[:] = s.ctrl
            if s.probe_path is not None: s.probe_path(t + i / SUB)
            d.xfrc_applied[TORSO] = 0
            if s.push is not None and s.push[0] <= t + i / SUB < s.push[1]: d.xfrc_applied[TORSO, :3] = s.push[2]
            mujoco.mj_step(M, d)
            if i % 3 == 2 or i == SUB - 1:
                a, p = H.skin(M, d); amax = np.maximum(amax, a); pairs += p
            if getattr(s, "rec", None) is not None and i % 5 == 4:
                s.rec.append((d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), a.astype(np.float16)))
        s.last_pairs = pairs; s.last_skin = amax
        return s.sense()

    def sense(s):
        d = s.d; R = d.xmat[TORSO].reshape(3, 3)
        q = d.qpos[H.JQ]; qd = d.qvel[H.JV]; f = d.actuator_force / KPV
        g = R.T @ np.array([0, 0, -1.0]); v = R.T @ d.qvel[0:3]; w = d.qvel[3:6]
        prop = np.concatenate([q, 0.1 * qd, f, g, [d.qpos[2] - 1.28], 0.3 * v, 0.3 * w])
        sk = np.log1p(5 * getattr(s, "last_skin", np.zeros(H.NT)))
        bp = R.T @ (d.xpos[BALL] - d.xpos[TORSO]); bv = R.T @ d.qvel[BALLV:BALLV + 3]
        vis = np.concatenate([bp, 0.3 * bv, [1.0]]) if np.abs(bp).max() < 2 else np.zeros(7)
        rp = R.T @ (d.mocap_pos[PED_M] - d.xpos[TORSO]); vis = np.concatenate([vis, rp, [1.0]] if np.abs(rp).max() < 2 else [vis, np.zeros(4)])
        return np.concatenate([prop, sk, vis]).astype(np.float32)

    # ---------- kinematics helpers on scratch data ----------
    def ik(s, qinit, tasks, joints, iters=60, damp=0.03):
        """tasks: list of (site_or_None, body_point, target) -> (eff site name, (body id, local point) or None, world target) ;
        moves joints so eff - point -> target-offset. returns q."""
        m, k = M, s.k; k.qpos[:] = s.d.qpos; k.qpos[H.JQ] = qinit; jac_a = np.zeros((3, m.nv)); jac_b = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        cols = H.JV[joints]; qa = H.JQ[joints]
        for it in range(iters):
            mujoco.mj_kinematics(m, k); mujoco.mj_comPos(m, k); E, Js = [], []
            for task in tasks:
                if len(task) == 4:  # orientation: (site, local axis, world axis, weight)
                    site, la, wa, wt = task; sid = m.site(site).id; R = k.site_xmat[sid].reshape(3, 3); ca = R @ la
                    mujoco.mj_jacSite(m, k, jac_a, jr, sid); E.append(wt * np.cross(ca, wa)); Js.append(wt * jr[:, cols].copy()); continue
                site, bp, tgt = task
                sid = m.site(site).id; pa = k.site_xpos[sid].copy(); mujoco.mj_jacSite(m, k, jac_a, jr, sid)
                if bp is not None:
                    b, lp, off = bp; pb = k.xpos[b] + k.xmat[b].reshape(3, 3) @ lp; mujoco.mj_jac(m, k, jac_b, jr, pb, b)
                    nb = k.xmat[b].reshape(3, 3) @ off; E.append(pb + nb - pa); Js.append(jac_a[:, cols] - jac_b[:, cols])
                else:
                    E.append(tgt - pa); Js.append(jac_a[:, cols].copy())
            e = np.concatenate(E); Jm = np.concatenate(Js)
            if np.linalg.norm(e) < 2e-3: break
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + damp ** 2 * np.eye(len(e)), e)
            dq = np.clip(dq, -0.2, 0.2); k.qpos[qa] = np.clip(k.qpos[qa] + dq, H.JR[joints, 0], H.JR[joints, 1])
        return k.qpos[H.JQ].copy(), float(np.linalg.norm(e))


def taxel_body_local(i):
    """body id, local point and local normal of taxel i (geom frame -> body frame)."""
    g = H.TG[i]; b = M.geom_bodyid[g]; R = np.zeros(9); mujoco.mju_quat2Mat(R, M.geom_quat[g]); R = R.reshape(3, 3)
    return b, M.geom_pos[g] + R @ H.TP[i], R @ H.TN[i]


def minjerk(a, b, n):
    out = []
    for i in range(1, n + 1):
        x = i / n; w = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5; out.append(a + (b - a) * w)
    return out


def limb_of_part(p):
    n = H.PARTS[p]
    for s in ("left", "right"):
        if n.startswith(s): return (s, "arm") if any(w in n for w in ("arm", "hand")) else (s, "leg")
    return None


def reach_joints(hand, tpart):
    js = ARM[hand] + ABD; lb = limb_of_part(tpart)
    if lb and lb[1] == "leg": js = js + LEG[lb[0]][:4]
    if lb and lb[1] == "arm" and lb[0] != hand: js = js + ARM[lb[0]]
    return js


def plan_reach(sim, ti, hand, t_start, t_arrive):
    """returns dict step -> q_des (fingers pointing), ik error"""
    b, lp, ln = taxel_body_local(ti); tp = H.TPART[ti]; js = reach_joints(hand, tp)
    q = handpose(sim.d.qpos[H.JQ].copy(), hand, POINT)
    qa, ea = sim.ik(q, [(f"tip_{hand}", (b, lp, ln * 0.07), None)], js)
    qb, eb = sim.ik(qa, [(f"tip_{hand}", (b, lp, ln * 0.0), None)], js)
    qc, ec = sim.ik(qb, [(f"tip_{hand}", (b, lp, -ln * 0.02), None)], js, iters=20)
    q_now = sim.d.qpos[H.JQ].copy(); n1 = max(2, t_arrive - t_start - 3)
    path = minjerk(q_now, qa, n1) + minjerk(qa, qb, 2) + [qc]
    return {t_start + i: p for i, p in enumerate(path)}, max(ea, eb)


def touch_info(sim, hand=None):
    """parts touched by the hand (self-touch) / by the probe in the last step; pairs = (g1, g2, f)"""
    self_parts, probe_parts, ball_parts = set(), set(), set(); hand_g = set(HAND_GEOMS["left"] + HAND_GEOMS["right"])
    for g1, g2, f in sim.last_pairs:
        for a, b in ((g1, g2), (g2, g1)):
            if a in H.GPART:
                if b == PROBE_G: probe_parts.add(H.GPART[a])
                elif b == BALL_G: ball_parts.add(H.GPART[a])
                elif b in H.GPART and b in hand_g and H.GPART[a] != H.GPART[b]: self_parts.add(H.GPART[a])
    return self_parts, probe_parts, ball_parts


def set_probe(sim, ti, dist):
    P, N = H.taxel_world(sim.d); sim.d.mocap_pos[PROBE_M] = P[ti] + N[ti] * dist; sim.d.mocap_quat[PROBE_M] = quat_z_to(N[ti])


def probe_path(sim, ti, t0, t1, depth=0.012):
    """poke taxel ti: approach over [t0, t0+2), press, retract by t1 (step units)."""
    def f(t):
        if t < t0 or t > t1 + 1: sim.d.mocap_pos[PROBE_M] = [0, 3, 1]; return
        x = (t - t0) / 2.0
        dist = 0.12 - 0.132 * min(1, x) if t < t1 else -depth + 0.132 * min(1, t - t1)
        set_probe(sim, ti, dist)
    return f


def place_ball(sim, pos, floating):
    d = sim.d; d.qpos[BALLQ:BALLQ + 3] = pos; d.qpos[BALLQ + 3:BALLQ + 7] = [1, 0, 0, 0]; d.qvel[BALLV:BALLV + 6] = 0
    M.body_gravcomp[BALL] = 1.0 if floating else 1e-6
    mujoco.mj_forward(M, d)


def plan_grasp(sim, hand, t_start, t_arrive, t_close, t_lift):
    s = 1 if hand == "left" else -1; bp = sim.d.mocap_pos[PED_M].copy(); js = ARM[hand] + ABD
    q = handpose(sim.d.qpos[H.JQ].copy(), hand, OPEN)
    o = [(f"grip_{hand}", np.array([0, -s, 0.]), np.array([0, -s, 0.]), 0.3), (f"grip_{hand}", np.array([0, 0, -1.]), np.array([.87, 0, -.5]), 0.3)]
    qa, ea = sim.ik(q, [(f"grip_{hand}", None, bp + [-0.01, s * 0.03, 0.10])] + o, js, iters=250)
    qb, eb = sim.ik(qa, [(f"grip_{hand}", None, bp)] + o, js, iters=150)
    qc = handpose(qb.copy(), hand, CLOSE)
    qd, ed = sim.ik(qc, [(f"grip_{hand}", None, bp + [0, 0, 0.14])] + o, js, iters=150)
    q_now = sim.d.qpos[H.JQ].copy(); n1 = max(2, t_arrive - t_start - 2)
    path = minjerk(q_now, qa, n1) + minjerk(qa, qb, 2) + [qb] * max(0, t_close - t_arrive) + minjerk(qb, qc, 3)
    return {t_start + i: p for i, p in enumerate(path)}, max(ea, eb, ed)
