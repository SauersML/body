import numpy as np, env as E, hum as H, mujoco
def run(F, dirn, ctl, seed=0, la=1.0, lh=1.0):
    rng = np.random.default_rng(seed); sim = E.Sim("free", rng); d = sim.d
    u = np.array([np.cos(dirn), np.sin(dirn), 0]); sim.push = (3, 4.5, F * u); q0 = E.Q0.copy()
    fell = False
    for t in range(E.T):
        q = q0.copy()
        if ctl:
            com = d.subtree_com[E.TORSO]; foot = 0.5 * (d.body("foot_left").xpos + d.body("foot_right").xpos) + [0.03, 0, 0]
            R = d.xmat[E.TORSO].reshape(3, 3); e = (com - foot)[:2]; v = d.subtree_linvel[E.TORSO][:2]
            ux = ctl[0] * e[0] + ctl[1] * v[0]; uy = ctl[0] * e[1] + ctl[1] * v[1]
            for s in ("left", "right"):
                q[E.JID[f"ankle_y_{s}"]] += ux; q[E.JID[f"hip_y_{s}"]] -= ctl[2] * ux
                sg = 1 if s == "left" else -1
                q[E.JID[f"ankle_x_{s}"]] += la * uy; q[E.JID[f"hip_x_{s}"]] += lh * sg * ctl[2] * uy
        sim.step(sim.grav_ctrl(q), t)
        if d.qpos[2] < 0.9: fell = True
    return fell

for ctl in [(8, 2, 1), (8, 2, 3), (12, 3, 2), (5, 2, 2), (8, 4, 2), (8, 1, 2)]:
    r = np.array([[run(F, a, ctl, la=-1, lh=1) for a in (0, np.pi/2, np.pi, -np.pi/2)] for F in (80, 110, 140, 170)]).astype(int)
    print(ctl, r.tolist())
