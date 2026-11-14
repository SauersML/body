"""A small designed 2D body: torso + two 3-segment arms, skin touch sensors, joint angles.

Body frame: torso centre at origin, shoulders at (+-0.3, 0.5), head at (0, 0.8).
Each arm: upper arm (0.5), forearm (0.45), hand (0.2). 4 skin sensors per segment, 24 total.
A disc object drifts in the world; sensors within its radius report pressure (overlap depth).
Sensor vector per step: [sin q, cos q (6 joints -> 12), touch (24)] = 36 dims.
"""
import numpy as np

SEG_LEN = [0.5, 0.45, 0.2]
PARTS = ["left upper arm", "left forearm", "left hand", "right upper arm", "right forearm", "right hand"]
N_PER_SEG = 4
N_SKIN = 6 * N_PER_SEG
SHOULDER = {0: np.array([-0.3, 0.5]), 1: np.array([0.3, 0.5])}  # 0=left, 1=right
SENSOR_DIM = 12 + N_SKIN


def fk(q):
    """q: (6,) joint angles [L shoulder, L elbow, L wrist, R ...]. Returns segment endpoints (2 arms, 4 points, 2)."""
    pts = np.zeros((2, 4, 2))
    for a in range(2):
        p = SHOULDER[a].copy(); ang = 0.0; pts[a, 0] = p
        base = np.pi if a == 0 else 0.0  # left arm points left at q=0
        for s in range(3):
            ang += q[3 * a + s]
            d = np.array([np.cos(base + (ang if a == 1 else -ang)), np.sin(base + (ang if a == 1 else -ang))])
            p = p + SEG_LEN[s] * d; pts[a, s + 1] = p
    return pts


def skin_positions(q):
    """World positions of the 24 skin sensors (sensor centres along each segment)."""
    pts = fk(q); out = []
    for a in range(2):
        for s in range(3):
            for k in range(N_PER_SEG):
                t = (k + 0.5) / N_PER_SEG
                out.append(pts[a, s] * (1 - t) + pts[a, s + 1] * t)
    return np.array(out)


def skin_geodesic():
    """Distance along the body surface between sensors (arm path, across torso between shoulders)."""
    pos = []  # arc length from shoulder
    for a in range(2):
        acc = 0.0
        for s in range(3):
            for k in range(N_PER_SEG):
                pos.append((a, acc + SEG_LEN[s] * (k + 0.5) / N_PER_SEG))
            acc += SEG_LEN[s]
    D = np.zeros((N_SKIN, N_SKIN))
    for i, (ai, si) in enumerate(pos):
        for j, (aj, sj) in enumerate(pos):
            D[i, j] = abs(si - sj) if ai == aj else si + sj + 0.6
    return D


def touch(q, obj, r):
    sp = skin_positions(q)
    d = np.linalg.norm(sp - obj, axis=1)
    return np.clip((r - d) / r, 0, 1), sp


def sensors(q, tch):
    return np.concatenate([np.sin(q), np.cos(q), tch]).astype(np.float32)


def episode(rng, T=8, p_touch=0.75):
    """Smooth random joint motion; object placed near a random skin point at the end (so touch is common)."""
    q0 = rng.uniform(-1.2, 1.2, 6); dq = rng.normal(0, 0.12, 6)
    qs = [q0 + dq * t + rng.normal(0, 0.02, 6) * np.sqrt(t + 1) for t in range(T)]
    r = rng.uniform(0.12, 0.2)
    sp_end = skin_positions(qs[-1])
    if rng.random() < p_touch:
        tgt = sp_end[rng.integers(N_SKIN)] + rng.normal(0, 0.05, 2)
    else:
        tgt = rng.uniform(-1.5, 1.5, 2)
    v = rng.normal(0, 0.05, 2)
    objs = [tgt - v * (T - 1 - t) for t in range(T)]
    S, info = [], {}
    for t in range(T):
        tch, sp = touch(qs[t], objs[t], r)
        S.append(sensors(qs[t], tch))
    tch, sp = touch(qs[-1], objs[-1], r)
    pts = fk(qs[-1])
    if tch.max() > 0:
        i = int(np.argmax(tch)); part = PARTS[i // N_PER_SEG]
        touched_y = float(sp[i, 1])
    else:
        i = -1; part = "nowhere"; touched_y = None
    info = dict(part=part, sensor=i, touched_y=touched_y,
                lhand_up=bool(pts[0, 3, 1] > 0.8), rhand_up=bool(pts[1, 3, 1] > 0.8),
                touch_world=sp[i] if i >= 0 else None, q=qs[-1])
    return np.stack(S), info


def questions(info, rng):
    """(question, answer, candidates) — answers only from internal state."""
    qa = [("Where are you being touched?", info["part"], PARTS + ["nowhere"]),
          ("Is your left hand above your head?", "yes" if info["lhand_up"] else "no", ["yes", "no"]),
          ("Is your right hand above your head?", "yes" if info["rhand_up"] else "no", ["yes", "no"])]
    if info["touched_y"] is not None:  # integration: needs touch location AND current pose
        qa.append(("Is the touch above your shoulders?", "yes" if info["touched_y"] > 0.5 else "no", ["yes", "no"]))
    return qa
