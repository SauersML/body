"""Closed-loop feedback expert (v2). A policy pi*(sim, goal) usable at ANY state (for DAgger relabelling).
Deterministic choices (hand, target taxel) from the episode's initial state, so the target is a function of what the model sees."""
import os
import numpy as np, mujoco, env as E, hum as H
def RJ(hand, part):
    js = E.reach_joints(hand, part); lb = E.limb_of_part(part)
    if os.environ.get("X2_NOABD", "1") == "1" and lb and lb[1] == "leg": js = [j for j in js if j not in E.ABD]
    return js

import os
NEARW = float(os.environ.get("X2_NEARW", 10.0)); VMAX = float(os.environ.get("X2_VMAX", 0.22))  # rad per 0.1 s step (largest joint)
CORR_D = float(os.environ.get("X2_CD", 0.12)); CORR_MAX = float(os.environ.get("X2_CM", 1.0)); PRESS = float(os.environ.get("X2_PRESS", 0.025)); INTERIOR = float(os.environ.get("X2_INT", 3.0))
_D = os.path.dirname(os.path.abspath(__file__)); _TR = np.load(os.path.join(_D, "taxel_rest.npy") if os.path.exists(os.path.join(_D, "taxel_rest.npy")) else os.path.join(_D, "..", "taxel_rest.npy"))
EDGE = np.array([np.linalg.norm(_TR[H.TPART != H.TPART[i]] - _TR[i], axis=1).min() for i in range(H.NT)])


def mid_hand(sim, part):
    lb = E.limb_of_part(part)
    if lb and lb[1] == "arm": return E.other_side(lb[0]) if hasattr(E, "other_side") else ("left" if lb[0] == "right" else "right")
    if lb and lb[1] == "leg": return lb[0]
    return None


def choose(sim, part, hand=None, near_taxel=None, ncand=10):
    """deterministic (hand, taxel) for touching `part`, from the current state: most reachable facing taxel."""
    P, N = H.taxel_world(sim.d); ix = np.where(H.TPART == part)[0]
    hands = [hand] if hand else ([mid_hand(sim, part)] if mid_hand(sim, part) else ["left", "right"])
    best = None
    for h in hands:
        tip = sim.d.site(f"tip_{h}").xpos; v = tip - P[ix]; dist = np.linalg.norm(v, axis=1)
        face = (N[ix] * v).sum(1) / dist  # >0: taxel faces the hand
        sc = dist - 0.15 * face - INTERIOR * EDGE[ix]
        if near_taxel is not None: sc = sc + NEARW * np.linalg.norm(P[ix] - P[near_taxel], axis=1)
        for ti in ix[np.argsort(sc)[:ncand]]:
            b, lp, ln = E.taxel_body_local(ti); js = RJ(h, part)
            q = E.handpose(sim.d.qpos[H.JQ].copy(), h, E.POINT)
            _, err = sim.ik(q, [(f"tip_{h}", (b, lp, ln * 0.0), None)], js, iters=40)
            c = err + 0.02 * sc[np.where(ix == ti)[0][0]]
            if best is None or c < best[0]: best = (c, h, int(ti), err)
            if err < 0.004: break
    return best[1], best[2], best[3]


class Reach:
    """closed-loop reach-and-press. goal: hand, taxel. act(sim, t) -> q_des (pre grav-comp)"""
    def __init__(s, hand, taxel, part, t_go=0):
        s.h, s.ti, s.part, s.t_go = hand, taxel, part, t_go; s.qprev = None

    def act(s, sim, t):
        q = sim.d.qpos[H.JQ].copy(); base = q if s.qprev is None else s.qprev
        if t < s.t_go:
            s.qprev = q if s.qprev is None else s.qprev; return s.qprev.copy()
        b, lp, ln = E.taxel_body_local(s.ti); js = RJ(s.h, s.part)
        k = sim.k; k.qpos[:] = sim.d.qpos; k.qpos[H.JQ] = base; mujoco.mj_kinematics(sim.m, k)
        P, N = H.taxel_world(k); tip = k.site(f"tip_{s.h}").xpos; d = np.linalg.norm(tip - P[s.ti])
        off = 0.06 if d > 0.09 else (0.0 if d > 0.03 else -PRESS)
        if RSMOOTH: off = float(np.interp(d, [0.02, 0.12], [-PRESS, 0.06]))
        # tracking compensation: actual tip vs commanded tip, relative to the target body (state-only, so DAgger-safe)
        Pa, _ = H.taxel_world(sim.d); ea = (sim.d.site(f"tip_{s.h}").xpos - Pa[s.ti]) - (tip - P[s.ti])
        Rb = k.xmat[b].reshape(3, 3); corr = -Rb.T @ ea if (d < CORR_D) else 0 * ln; cn = np.linalg.norm(corr); corr = corr * min(1, CORR_MAX / max(cn, 1e-9))
        q0 = E.handpose(base.copy(), s.h, E.POINT)
        qg, err = sim.ik(q0, [(f"tip_{s.h}", (b, lp, ln * off + corr), None)], js, iters=40)
        if off > 0 and err > 0.01: qg, err = sim.ik(q0, [(f"tip_{s.h}", (b, lp, -ln * 0.02 + corr), None)], js, iters=40)
        dq = qg - base; m = np.abs(dq[js]).max() if len(js) else 0
        if m > VMAX: dq = dq * (VMAX / m)
        qn = base + dq; qn = E.handpose(qn, s.h, E.POINT)
        # other arm: relax to rest smoothly
        oh = "left" if s.h == "right" else "right"
        if not any(j in js for j in E.ARM[oh]): qn[E.ARM[oh]] = base[E.ARM[oh]] + np.clip(E.Q0[E.ARM[oh]] - base[E.ARM[oh]], -0.1, 0.1)
        s.qprev = qn; return qn.copy()


class Grasp:
    """closed-loop grasp of the vertical handle: approach above, descend, close; phase from current state."""
    def __init__(s, hand, t_go=0):
        s.h, s.t_go = hand, t_go; s.qprev = None; s.sg = 1 if hand == "left" else -1

    def act(s, sim, t):
        q = sim.d.qpos[H.JQ].copy(); base = q if s.qprev is None else s.qprev
        if t < s.t_go: s.qprev = base; return base.copy()
        h, sg = s.h, s.sg; bp = sim.d.mocap_pos[E.PED_M].copy(); js = E.ARM[h] + E.ABD
        g = sim.d.site(f"grip_{h}").xpos; rel = g - bp; dxy = np.linalg.norm(rel[:2]); dist = np.linalg.norm(rel)
        o = [(f"grip_{h}", np.array([0, -sg, 0.]), np.array([0, -sg, 0.]), 0.3), (f"grip_{h}", np.array([0, 0, -1.]), np.array([.87, 0, -.5]), 0.3)]
        closing = dist < GR_CLOSE
        if GSMOOTH:
            wd = float(np.clip((0.09 - dxy) / 0.05, 0, 1)) * float(rel[2] > -0.05); wc = float(np.clip((GC_HI - dist) / (GC_HI - GC_LO), 0, 1))
            tgt = bp + (1 - wd) * np.array([-0.01, sg * 0.03, 0.10]); fing = {kk: E.OPEN[kk] + wc * (E.CLOSE[kk] - E.OPEN[kk]) for kk in E.OPEN}; closing = wc > 0.5
        elif closing: tgt = bp; fing = E.CLOSE
        elif dxy < 0.05 and rel[2] > -0.03: tgt = bp; fing = E.OPEN
        else: tgt = bp + [-0.01, sg * 0.03, 0.10]; fing = E.OPEN
        # tracking compensation in world frame
        k = sim.k; k.qpos[:] = sim.d.qpos; k.qpos[H.JQ] = base; mujoco.mj_kinematics(sim.m, k)
        ea = g - k.site(f"grip_{h}").xpos; ea = ea * min(1, 0.04 / max(np.linalg.norm(ea), 1e-9))
        q0 = E.handpose(base.copy(), h, E.OPEN if not closing else {kk: base[E.JID[f"{kk}_{h}"]] for kk in E.OPEN})
        qg, err = sim.ik(q0, [(f"grip_{h}", None, tgt - (ea if dist < 0.12 else 0))] + o, js, iters=60)
        dq = qg - base; m = np.abs(dq[js]).max()
        if m > VMAX: dq = dq * (VMAX / m)
        qn = base + dq
        for kk, v in fing.items():
            j = E.JID[f"{kk}_{h}"]; qn[j] = base[j] + np.clip(v - base[j], -0.5, 0.5)
        oh = "left" if h == "right" else "right"; qn[E.ARM[oh]] = base[E.ARM[oh]] + np.clip(E.Q0[E.ARM[oh]] - base[E.ARM[oh]], -0.1, 0.1)
        s.qprev = qn; return qn.copy()


GC_HI = float(os.environ.get("X2_GCHI", 0.05)); GC_LO = float(os.environ.get("X2_GCLO", 0.025)); GR_CLOSE = float(os.environ.get("X2_GRC", 0.025)); GSMOOTH = os.environ.get("X2_GSMOOTH", "0") == "1"; RSMOOTH = os.environ.get("X2_RSMOOTH", "0") == "1"


# ---- canonical (state-independent) targets: hand by a fixed rule, taxel chosen once from the rest pose ----
CANON = os.environ.get("X2_CANON", "0") in ("1", "2"); CANON2 = os.environ.get("X2_CANON", "0") == "2"; _CT = {}; _CP = {}
_CS = None
def _canon_scores():
    global _CS
    if _CS is None:
        import json; f = os.path.join(_D, "canon_scores.json"); _CS = {int(p): {int(t): v for t, v in d.items()} for p, d in json.load(open(f)).items()}
    return _CS
def rule_hand(part):
    lb = E.limb_of_part(part)
    if lb and lb[1] == "arm": return "left" if lb[0] == "right" else "right"
    if lb: return lb[0]
    return "right"
def canon(part, near_taxel=None):
    """(hand, taxel) from the rest pose, cached."""
    if CANON2:
        sc = _canon_scores()[part]; h = rule_hand(part)
        if near_taxel is None: return h, max(sc, key=sc.get)
        good = [t for t, v in sc.items() if v >= 0.75] or [max(sc, key=sc.get)]
        return h, min(good, key=lambda t: np.linalg.norm(_TR[t] - _TR[near_taxel]))
    key = (part, near_taxel)
    if key not in (_CP if near_taxel is not None else _CT):
        sim = E.Sim("mount", None); h = rule_hand(part); r = choose(sim, part, hand=h, near_taxel=near_taxel)
        (_CP if near_taxel is not None else _CT)[key] = (r[0], r[1])
    return (_CP if near_taxel is not None else _CT)[key]
