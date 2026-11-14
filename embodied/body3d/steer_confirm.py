"""Confirmation of 8B L18 alpha-1.5 steering with 24 varied start poses (seeds fixed in notes/research/body-3d.md).
python steer_confirm.py MODEL ARM ; env DO=rep NP=400 CKDIR=...  (same vector construction as steer_dump.py)"""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze3d.py")).read()
src = src.split("# ---------------- causal steering")[0]
exec(compile(src, "analyze3d_head", "exec"))
import pickle, mujoco
from start_poses import start_pose, POSE_SEEDS
targets = ["head", "chest", "belly", "right shin", "left thigh", "right thigh", "left forearm", "left shin"]
Ls = B.L_FWD; Zs = HL[Ls]; mu = Zs.mean(0); scale = np.median(np.linalg.norm(Zs - mu, axis=1)); ALPHA = 1.5
POSES = [start_pose(s) for s in POSE_SEEDS]; _it = None

def rebuild(kind, e, rng, probe=True):  # replaces analyze3d's: same mounted scene, no probe, varied start pose
    sim = E.Sim("mount", rng); q = next(_it); d = sim.d
    d.qpos[H.JQ] = q; d.qvel[:] = 0; mujoco.mj_forward(E.M, d); sim.ctrl = sim.grav_ctrl(q); sim.settle(); return sim

def run(sv):
    global _it; _it = iter(POSES)
    sims, eps, tc, traj = closed_loop("poke_reach", len(POSES), 11, steer=sv, probe=False, record=True)
    return [dict(qpos=np.stack([x["qpos"] for x in tr]).astype(np.float32), qpos_final=s.d.qpos.copy(),
                 tc=t) for s, t, tr in zip(sims, tc, traj)]

def dist(qpos, p):
    d = mujoco.MjData(E.M); d.qpos[:] = qpos; mujoco.mj_forward(E.M, d); Pw, _ = H.taxel_world(d)
    return min(float(np.linalg.norm(Pw[H.TPART == p] - d.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right"))

OUTD = {"none": run(None)}
for nmx in targets:
    p = H.PARTS.index(nmx); v = Zs[part == p].mean(0) - mu; vn = v / np.linalg.norm(v)
    OUTD[nmx] = run((Ls, torch.tensor(vn * scale * ALPHA, dtype=torch.float32), [1, 2, 3, 4]))
    ds = [dist(r["qpos_final"], p) for r in OUTD[nmx]]; d0 = [dist(r["qpos_final"], p) for r in OUTD["none"]]
    print("confirm", nmx, "steered", round(100 * np.mean(ds), 1), "unsteered", round(100 * np.mean(d0), 1), round(time.time() - t00), flush=True)
pickle.dump(dict(runs=OUTD, poses=POSES, seeds=POSE_SEEDS, layer=Ls, scale=float(scale), alpha=ALPHA, targets=targets),
            open(f"{OUT}/steer_confirm_{tag}.pkl", "wb"))
print("CONFIRM DONE", flush=True)
