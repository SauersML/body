import numpy as np, env as E, hum as H, mujoco
rng = np.random.default_rng(5)
sim = E.Sim("mount", rng); hand = "left"; s = 1
pos = np.array([0.33, 0.15, 1.0]); E.place_ball(sim, pos, False); plan, err = E.plan_grasp(sim, hand, 1, 8, 10, 14); print("err", err)
q = sim.d.qpos[H.JQ].copy(); d = sim.d
for t in range(E.T):
    q = plan.get(t, q); sim.step(sim.grav_ctrl(q), t)
    R = d.site("palm_left").xmat.reshape(3, 3); rel = R.T @ (d.xpos[E.BALL] - d.site("palm_left").xpos)
    con = sorted({M for g1, g2, f in sim.last_pairs for M in (sim.m.geom(g1).name, sim.m.geom(g2).name) if E.BALL_G in (g1, g2)})
    print(t, "ball", d.xpos[E.BALL].round(3), "rel_palm", rel.round(3), "tgt-err", np.abs(q - d.qpos[H.JQ])[E.ARM['left']].max().round(2), con)
