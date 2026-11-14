"""Varied start poses for the steering confirmation: arm joints (shoulder, elbow, wrist, both sides) = Q0 + N(0, 0.35),
the per-joint spread of the arm babbling seen in training, clipped to joint limits. Everything else at Q0."""
import numpy as np, env as E, hum as H
POSE_SEEDS = list(range(7100, 7124))  # fixed before the run (24 poses)

def start_pose(seed):
    rng = np.random.default_rng(seed); q = E.Q0.copy()
    for s in ("left", "right"):
        a = E.ARM[s]; q[a] = np.clip(E.Q0[a] + rng.normal(0, 0.35, len(a)), H.JR[a, 0], H.JR[a, 1])
    return q
