"""v2 task layer on top of body.py: continuous arm position, landmark-neighbourhood questions, world side, crossed-pose split."""
import numpy as np
import body
from body import N_PER_SEG, SEG_LEN, PARTS

ARM_LEN = sum(SEG_LEN)
# arc length (from shoulder) of each sensor centre, and which arm
ARC = np.array([sum(SEG_LEN[:s]) + SEG_LEN[s] * (k + .5) / N_PER_SEG for a in range(2) for s in range(3) for k in range(N_PER_SEG)])
ARM = np.repeat([0, 1], 3 * N_PER_SEG)
LANDMARK = {"shoulder": 0.0, "elbow": SEG_LEN[0], "wrist": SEG_LEN[0] + SEG_LEN[1], "fingertips": ARM_LEN}
NEAR = 0.15


def crossed(q):
    p = body.fk(q)
    return bool(p[0, 3, 0] > 0 or p[1, 3, 0] < 0)  # a hand past the body midline


def episode(rng, T=8, p_touch=0.75, want_crossed=None):
    while True:
        S, info = body.episode(rng, T, p_touch)
        if want_crossed is None or crossed(info["q"]) == want_crossed: break
    tch = S[-1, 12:]
    if tch.max() > 0:
        arm = int(ARM[tch.argmax()]); w = tch * (ARM == arm)
        info["arm"] = arm; info["arc"] = float((w * ARC).sum() / w.sum())  # pressure-weighted position along the arm
        sp = body.skin_positions(info["q"]); info["touch_xy"] = (w[:, None] * sp).sum(0) / w.sum()
    else:
        info["arm"] = None
    info["crossed"] = crossed(info["q"])
    return S, info


def questions(info):
    qa = [("Where are you being touched?", info["part"], PARTS + ["nowhere"]),
          ("Is your left hand above your head?", "yes" if info["lhand_up"] else "no", ["yes", "no"]),
          ("Is your right hand above your head?", "yes" if info["rhand_up"] else "no", ["yes", "no"])]
    if info["arm"] is not None:
        side = ["left", "right"][info["arm"]]
        qa.append(("Is the touch above your shoulders?", "yes" if info["touch_xy"][1] > 0.5 else "no", ["yes", "no"]))
        qa.append(("How far along your arm is the touch, from shoulder (0) to fingertips (10)?",
                   str(int(round(10 * info["arc"] / ARM_LEN))), [str(i) for i in range(11)]))
        for lm, a in LANDMARK.items():
            qa.append((f"Is the touch near your {side} {lm}?", "yes" if abs(info["arc"] - a) < NEAR else "no", ["yes", "no"]))
        qa.append(("Is the touch on the left or the right side of the world?", "left" if info["touch_xy"][0] < 0 else "right", ["left", "right"]))
    return qa
