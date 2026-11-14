"""v2 episodes: diverse start poses + closed-loop expert (expert2) for reach / poke_reach / grasp; same info/QA format as episodes.py.
make_scene(kind, rng, diverse) -> (sim, info, policy); install() routes episodes.MAKERS to these and makes the mix motor-heavy."""
import os, numpy as np, mujoco, env as E, hum as H, episodes as P, expert2 as X
T = P.T; DIV = float(os.environ.get("DIVERSE", 0.75)); OLD = dict(P.MAKERS)
MOTOR = ("reach", "poke_reach", "grasp")


def start_sim(rng, diverse, q=None):
    sim = E.Sim("mount", rng)
    if diverse or q is not None:
        if q is None:
            q = E.Q0.copy()
            for s in ("left", "right"):
                a = E.ARM[s]; q[a] = np.clip(E.Q0[a] + rng.normal(0, 0.35, len(a)), H.JR[a, 0], H.JR[a, 1])
        sim.d.qpos[H.JQ] = q; sim.d.qvel[:] = 0; mujoco.mj_forward(E.M, sim.d)
        for _ in range(3): sim.ctrl = sim.grav_ctrl(q); sim.settle(60)
    return sim


def make_scene(kind, rng, diverse=None, q=None):
    diverse = bool(rng.uniform() < DIV) if diverse is None else diverse
    if kind == "reach":
        sim = start_sim(rng, diverse, q); ti0 = int(rng.integers(H.NT)); part = int(H.TPART[ti0])
        h, ti = X.canon(part) if X.CANON else X.choose(sim, part)[:2]; return sim, dict(taxel=ti, part=part, hand=h, self=True, diverse=diverse), X.Reach(h, ti, part, t_go=0)
    if kind == "poke_reach":
        for _ in range(8):
            sim = start_sim(rng, diverse, q); ti = int(rng.integers(H.NT)); part = int(H.TPART[ti])
            if not H.PARTS[part].endswith("hand"): break
        sim.probe_path = E.probe_path(sim, ti, 1, 4, depth=0.015)
        h, tt = X.canon(part, ti) if X.CANON else X.choose(sim, part, near_taxel=ti)[:2]; return sim, dict(taxel=ti, aim=tt, part=part, hand=h, self=True, diverse=diverse), X.Reach(h, tt, part, t_go=5)
    if kind == "grasp":
        hand = str(rng.choice(["left", "right"])); s = 1 if hand == "left" else -1
        sim = start_sim(rng, diverse, q); pos = np.array([rng.uniform(0.25, 0.42), s * rng.uniform(0.0, 0.3), rng.uniform(0.85, 1.15)])
        sim.d.mocap_pos[E.PED_M] = pos; mujoco.mj_forward(E.M, sim.d)
        return sim, dict(hand=hand, pole=pos, self=None, part=H.PARTS.index(f"{hand} hand"), diverse=diverse), X.Grasp(hand, t_go=0)


def scene_from_old(kind, e):
    """old held-out episode (Q0 start) -> same scene with the new expert."""
    i = e["info"]; sim = E.Sim("mount", None)
    if kind == "reach":
        h, ti = X.canon(i["part"]) if X.CANON else X.choose(sim, i["part"])[:2]; return sim, dict(i, taxel=ti, hand=h, diverse=False), X.Reach(h, ti, i["part"], t_go=0)
    if kind == "poke_reach":
        sim.probe_path = E.probe_path(sim, i["taxel"], 1, 4, depth=0.015); h, tt = X.canon(i["part"], i["taxel"]) if X.CANON else X.choose(sim, i["part"], near_taxel=i["taxel"])[:2]
        return sim, dict(i, aim=tt, hand=h, diverse=False), X.Reach(h, tt, i["part"], t_go=5)
    if kind == "grasp":
        sim.d.mocap_pos[E.PED_M] = i["pole"]; mujoco.mj_forward(E.M, sim.d); return sim, dict(i, diverse=False), X.Grasp(i["hand"], t_go=0)


def success(kind, info, sim, tc):
    if kind == "reach": return any(info["part"] in x[0] for x in tc[-6:])
    if kind == "poke_reach": return any(info["part"] in x[0] for x in tc[-5:]) and any(info["part"] in x[1] for x in tc[:7])
    hg = {g for g1, g2, f in sim.last_pairs for g in (g1, g2) if E.PED_G in (g1, g2) and g in E.HAND_GEOMS[info["hand"]]}; return len(hg) >= 2


def spot_cm(kind, info, sim):
    """named-spot distance (cm): reach -> nearest taxel of the named part; poke_reach -> the poked taxel; grasp -> grip site to handle."""
    if kind == "grasp": return 100 * float(np.linalg.norm(sim.d.site(f"grip_{info['hand']}").xpos - info["pole"]))
    Pw, _ = H.taxel_world(sim.d); tips = [sim.d.site(f"tip_{h}").xpos for h in ("left", "right")]
    pts = Pw[H.TPART == info["part"]] if kind == "reach" else Pw[[info["taxel"]]]
    return 100 * min(float(np.linalg.norm(pts - t, axis=1).min()) for t in tips)


def mask_fn(kind): return (lambda t: float(t >= 5)) if kind == "poke_reach" else (lambda t: 1.0)


def run_expert(kind, sim, info, pol):
    S = [sim.sense()]; A = []; tc = []; mask = []; mf = mask_fn(kind)
    for t in range(T):
        q = pol.act(sim, t); c = sim.grav_ctrl(q); A.append(c.astype(np.float32)); mask.append(mf(t))
        S.append(sim.step(c, t)); tc.append(E.touch_info(sim))
    ok = success(kind, info, sim, tc); info = dict(info, success=ok, spot_cm=spot_cm(kind, info, sim))
    return dict(S=np.stack(S), A=np.stack(A), mask=np.array(mask, np.float32) * ok, kind=kind, info=info)


def _mk(kind):
    def f(rng): return run_expert(kind, *make_scene(kind, rng))
    return f


def install(p=(0.25, 0.08, 0.25, 0.08, 0.26, 0.08)):
    for k in MOTOR: P.MAKERS[k] = _mk(k)
    def make_episode(rng, kind=None):
        kind = kind or str(rng.choice(P.KINDS, p=list(p))); return P.MAKERS[kind](rng)
    P.make_episode = make_episode
