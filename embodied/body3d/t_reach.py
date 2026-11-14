import numpy as np, env as E, hum as H, time
rng = np.random.default_rng(0); ok = 0; n = 0; t0 = time.time(); stats = {}
for trial in range(60):
    sim = E.Sim("mount", rng); ti = rng.integers(H.NT); tp = H.TPART[ti]; lb = E.limb_of_part(tp)
    hand = rng.choice(["left", "right"])
    if lb and lb[1] == "arm" and lb[0] == hand: hand = "left" if hand == "right" else "right"
    plan, err = E.plan_reach(sim, ti, hand, 1, 10)
    if err > 0.01: stats.setdefault(H.PARTS[tp], []).append(-1); continue
    q = sim.d.qpos[H.JQ].copy(); hit = False; dmin = 9
    for t in range(E.T):
        q = plan.get(t, q); sim.step(sim.grav_ctrl(q), t)
        sp, pp, bp = E.touch_info(sim)
        P, N = H.taxel_world(sim.d); dmin = min(dmin, np.linalg.norm(sim.d.site(f"tip_{hand}").xpos - P[ti]))
        if tp in sp: hit = True
    n += 1; ok += hit; stats.setdefault(H.PARTS[tp], []).append(int(hit))
print("ik-ok", n, "hit", ok, "time", round(time.time() - t0, 1))
for k, v in stats.items(): print(k, v)
