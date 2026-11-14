import numpy as np, env as E, hum as H
rng = np.random.default_rng(1); res = []
for trial in range(40):
    sim = E.Sim("mount", rng); hand = rng.choice(["left", "right"]); s = 1 if hand == "left" else -1
    pos = np.array([rng.uniform(0.25, 0.42), s * rng.uniform(0.0, 0.3), rng.uniform(0.85, 1.15)])
    sim.d.mocap_pos[E.PED_M] = pos; plan, err = E.plan_grasp(sim, hand, 1, 8, 10, 14)
    q = sim.d.qpos[H.JQ].copy(); z0 = sim.d.xpos[E.BALL][2]
    for t in range(E.T): q = plan.get(t, q); sim.step(sim.grav_ctrl(q), t)
    sp, pp, bp = E.touch_info(sim); hg = {g for g1, g2, f in sim.last_pairs for g in (g1, g2) if E.PED_G in (g1, g2) and g in E.HAND_GEOMS[hand]}; held = float(len(hg) >= 2 and True)
    res.append((round(err, 3), round(held, 3), sorted(H.PARTS[i] for i in bp)))
print(sum(r[1] > 0.5 for r in res), len(res)); print(res[:12])
