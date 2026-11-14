"""v4 whole-body sim on the rigged MakeHuman neutral mesh (built by build_body4.py -> v4/body4.npz).
SKIN: the whole body surface is ONE connected sheet (2000 samples, geodesic distances on the mesh) read at 64 electrodes
scattered over it. A contact (center sample, pressure p, width w) adds p*(0.4+1.5w)*mean_patch 1/(1+d_geo^2/SIG^2) to every
electrode; place, pressure and width must be inferred. Plus 4 'vibration' channels from a held stick's tip contacts.
PROPRIO: sin/cos + velocity of 24 joints (sensed angles may carry a per-episode bias), 4 tendon tensions.
VISION: brush tip (xyz, visible, seen touching), ball (xyz, visible), stick tip (xyz, visible).
ACTIONS: one motor token per step: 22 joints x (+/-), 4 tendon pulls (fingers follow mechanically), rest."""
import os, numpy as np
from scipy.spatial import cKDTree

_B = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4", "body4.npz"))
V, F, W, SKIN, GEO = _B["V"], _B["F"], _B["W"], _B["SKIN"], _B["GEO"]
ELEC, PART, ARC, LIMB, LMI, LMN, HT = _B["ELEC"], _B["PART"], _B["ARC"], _B["LIMB"], _B["LMI"], list(_B["LMN"]), _B["HT"]
HEADS, TAILS, PARENT, BONES = _B["HEADS"], _B["TAILS"], _B["PARENT"], list(_B["BONES"])
JB, JAX, JLO, JHI, JNAME, PARTS = _B["JB"], _B["JAX"], _B["JLO"], _B["JHI"], list(_B["JNAME"]), list(_B["PARTS"])
NB, NJ, NS, NEL = len(BONES), 24, len(SKIN), len(ELEC)
P0 = V[SKIN].astype(np.float64); N0 = _B["VN"][SKIN].astype(np.float64)
WS = W[SKIN].astype(np.float64); HARD_S = _B["HARD"][SKIN]
SIG = 1.4
GE = 1 / (1 + (GEO[:, ELEC].astype(np.float64) / SIG) ** 2)  # [NS, 64]
FING = [JNAME.index("L fingers"), JNAME.index("R fingers")]
CMD = [j for j in range(NJ) if j not in FING]  # 22 commanded joints
NA = 2 * len(CMD) + 4 + 1  # 49
A_REST = NA - 1
DQ = 0.25
PROP_DIM, SKIN_DIM, VIS_DIM = 2 * NJ + NJ + 4, NEL + 4, 13
SENSOR_DIM = PROP_DIM + SKIN_DIM + VIS_DIM  # 76 + 68 + 13 = 157
# training joint box (novel-posture tests go beyond it)
TLO = JLO + 0.1 * (JHI - JLO); THI = JHI - 0.1 * (JHI - JLO)
for j in FING: TLO[j], THI[j] = JLO[j], JHI[j]
ARM_BONES = [[BONES.index(f"{s}_{b}") for b in ("uarm", "farm", "hand", "fing")] for s in "LR"]
TOUCH_EXCL = [np.isin(HARD_S, ARM_BONES[a][1:]) for a in range(2)]  # a hand can't touch its own forearm/hand
CHEST = BONES.index("chest")


_KJ = np.stack([np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]) for a in JAX]); _KJ2 = _KJ @ _KJ
_BJ = [np.where(JB == b)[0] for b in range(NB)]
_ANC = []
for _b in range(NB):
    _c, _x = [], _b
    while _x >= 0: _c.append(_x); _x = PARENT[_x]
    _ANC.append(set(_c))
_PW = {}


def rotm(ax, th):
    """Rodrigues, batched: ax [3], th [K] -> [K,3,3]."""
    th = np.asarray(th, float); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def fk(q, bones=None):
    """q [K,24] -> R [K,NB,3,3], T [K,NB,3] (posed joint position of each bone head). bones: only these (+ancestors)."""
    q = np.atleast_2d(q); K = len(q); R = np.zeros((K, NB, 3, 3)); T = np.zeros((K, NB, 3))
    need = range(NB) if bones is None else sorted(set().union(*[_ANC[b] for b in bones]))
    for b in need:
        js = _BJ[b]; L = None
        for j in js:
            s_, c_ = np.sin(q[:, j])[:, None, None], np.cos(q[:, j])[:, None, None]
            Rj = np.eye(3)[None] + s_ * _KJ[j][None] + (1 - c_) * _KJ2[j][None]
            L = Rj if L is None else L @ Rj
        p = PARENT[b]
        if p < 0: R[:, b] = np.eye(3) if L is None else L; T[:, b] = HEADS[b]
        else:
            R[:, b] = R[:, p] if L is None else R[:, p] @ L; T[:, b] = T[:, p] + np.einsum("kij,j->ki", R[:, p], HEADS[b] - HEADS[p])
    return R, T


def pose_pts(R, T, P, Wt):
    """LBS of rest points P [n,3] with weights Wt [n,NB] for one pose (R [NB,3,3], T [NB,3])."""
    out = np.zeros_like(P)
    for b in np.where(Wt.sum(0) > 0)[0]:
        m = Wt[:, b] > 0
        out[m] += Wt[m, b, None] * ((P[m] - HEADS[b]) @ R[b].T + T[b])
    return out


def pose_pts_batch(R, T, P, Wt):
    """Batched over poses: R [K,NB,3,3], T [K,NB,3], P [n,3] -> [K,n,3]."""
    out = np.zeros((len(R), len(P), 3))
    for b in np.where(Wt.sum(0) > 0)[0]:
        m = Wt[:, b] > 0
        out[:, m] += Wt[None, m, b, None] * (np.einsum("nj,kij->kni", P[m] - HEADS[b], R[:, b]) + T[:, b, None])
    return out


def skin_pose(q):
    R, T = fk(q); R, T = R[0], T[0]
    P = pose_pts(R, T, P0, WS)
    Nn = np.einsum("nij,nj->ni", R[HARD_S], N0)
    return P, Nn, R, T


def point_world(q, idx):
    """world position of skin samples idx for poses q [K,24] -> [K,len(idx),3]"""
    key = tuple(idx) if len(idx) < 64 else None
    if key is None or key not in _PW:
        bones = np.where(WS[idx].sum(0) > 0)[0].tolist()
        if key is not None: _PW[key] = bones
    else: bones = _PW[key]
    R, T = fk(q, bones); return pose_pts_batch(R, T, P0[idx], WS[idx])


def stick_seg(R, T, hand):
    """stick held in hand: from palm along the hand direction. R,T for one pose."""
    b = ARM_BONES[hand][2]; d = R[b] @ (TAILS[b] - HEADS[b]); d /= np.linalg.norm(d)
    base = T[b] + R[b] @ (TAILS[b] - HEADS[b]) * 0.6
    return base, d


def channels(contacts):
    v = np.zeros(SKIN_DIM)
    for c in contacts:
        if c["src"] == "vib": continue
        patch = GEO[c["s"]] < max(c["w"] / 2, 0.05)
        v[:NEL] += c["p"] * (0.4 + 1.5 * c["w"]) * GE[patch].mean(0)
    return v


class Env:
    def __init__(s, q0, rng, lo=None, hi=None):
        s.q = np.array(q0, float); s.dq = np.zeros(NJ); s.t = 0; s.rng = rng; s.ten = np.zeros((2, 2))
        s.lo = TLO if lo is None else lo; s.hi = THI if hi is None else hi
        s.bias = np.zeros(NJ)  # proprioceptive bias (sensed = true + bias)
        s.ball = None  # dict(c, r, visible, p)
        s.wall = None; s.push = 0
        s.brush = None  # dict(path=[...xyz per t], on=[bool per t], seen=[xyz per t] or None, visible)
        s.other = None  # unseen external toucher: dict(s=[sample idx per t], on=[...], p, w)
        s.stick = None  # dict(hand, L)
        s._cache = None

    # ---------- geometry ----------
    def posed(s):
        if s._cache is None or not np.array_equal(s._cache[1], s.q):
            P, Nn, R, T = skin_pose(s.q); s._cache = (0, s.q.copy(), P, Nn, R, T, cKDTree(P))
        return s._cache[2:]

    def self_contacts(s, P, Nn, tree):
        C = []; depth_max = -9
        for a in range(2):
            for k, idx in enumerate(HT[a]):
                t = P[idx]; cand = tree.query_ball_point(t, 0.7)
                cand = [i for i in cand if not TOUCH_EXCL[a][i]]
                if not cand: continue
                cand = np.array(cand); d = np.linalg.norm(P[cand] - t, axis=1); i = cand[d.argmin()]
                sd = float((t - P[i]) @ Nn[i])
                if sd < 0.15 and d.min() < 0.5:
                    p = float(np.clip((0.15 - sd) / 0.35, 0.05, 1.0)) * (0.6 + 0.5 * s.ten[a, 0] if k == 0 else 1.0)
                    p = min(p, 1.0); w = 0.25 + 0.3 * p
                    C += [dict(s=int(i), p=p, w=w, src="self", by=a), dict(s=int(idx), p=p, w=w, src="selfhand", by=a)]
                    depth_max = max(depth_max, -sd)
        return C, depth_max

    def contacts(s, full=False):
        P, Nn, R, T, tree = s.posed(); C, dep = s.self_contacts(P, Nn, tree); s._depth = dep
        if s.ball is not None:
            b = s.ball; d = np.linalg.norm(P - b["c"], axis=1); i = int(d.argmin())
            if d[i] < b["r"]:
                p = float(np.clip((b["r"] - d[i]) / 0.3, 0.1, 1.0)) * b.get("p", 1.0); C.append(dict(s=i, p=p, w=0.3 + b["r"] * 0.6, src="ball"))
        if s.wall is not None:
            ax, pos = s.wall; x = P[:, ax]; i = int(x.argmax())
            if x[i] > pos - 0.08: C.append(dict(s=i, p=min(1., .3 + .3 * s.push), w=0.3, src="wall"))
        if s.other is not None and s.t < len(s.other["on"]) and s.other["on"][s.t]:
            o = s.other; C.append(dict(s=int(o["s"][s.t]), p=o["p"], w=o["w"], src="other"))
        if s.brush is not None and s.brush.get("felt") and s.brush["on"][min(s.t, len(s.brush["on"]) - 1)]:
            br = s.brush; C.append(dict(s=int(br["s"][min(s.t, len(br["s"]) - 1)]), p=br["p"], w=0.35, src="brush"))
        if s.stick is not None:
            base, d = stick_seg(R, T, s.stick["hand"]); tip = base + d * s.stick["L"]; s._stip = tip
            hit = None
            if s.ball is not None and np.linalg.norm(tip - s.ball["c"]) < s.ball["r"] + 0.1: hit = 1.0
            if s.wall is not None and tip[s.wall[0]] > s.wall[1] - 0.05: hit = min(1., .3 + .3 * s.push)
            if hit:
                palm = int(HT[s.stick["hand"]][1]); C.append(dict(s=palm, p=0.5 * hit, w=0.3, src="stickhand"))
                C.append(dict(s=-1, p=hit, w=0, src="vib", pos=s.stick["L"], tip=tip))
        return C

    # ---------- sensing ----------
    def sense(s):
        C = s.contacts(); ch = channels(C)
        for c in C:
            if c["src"] == "vib": ch[NEL:] += c["p"] * np.cos(np.arange(1, 5) * np.pi * c["pos"] / 8.0)
        ch[:NEL] += s.rng.normal(0, .01, NEL)
        qs = s.q + s.bias
        prop = np.concatenate([np.sin(qs), np.cos(qs), s.dq / DQ, s.ten.ravel()])
        vis = np.zeros(VIS_DIM)
        if s.brush is not None and s.brush.get("visible"):
            br = s.brush; k = min(s.t, len(br["seen_on"]) - 1)
            if br.get("path") is not None: seen = br["path"][k]
            else:
                P, Nn = s.posed()[:2]; i = int(br["s"][k]); seen = P[i] + 0.15 * Nn[i] + br.get("offset", 0)
            vis[0:3] = seen / 5; vis[3] = 1; vis[4] = float(br["seen_on"][k])
        if s.ball is not None and s.ball.get("visible"): vis[5:8] = s.ball["c"] / 5; vis[8] = 1
        if s.stick is not None:
            P, Nn, R, T, tree = s.posed(); base, d = stick_seg(R, T, s.stick["hand"]); vis[9:12] = (base + d * s.stick["L"]) / 5; vis[12] = 1
        s._last_contacts = C
        return np.concatenate([prop, ch, vis]).astype(np.float32)

    # ---------- acting ----------
    def next_q(s, a, q=None, ten=None):
        q = (s.q if q is None else q).copy(); ten = (s.ten if ten is None else ten).copy()
        if a < 2 * len(CMD):
            j = CMD[a // 2]; q[j] = np.clip(q[j] + (DQ if a % 2 == 0 else -DQ), s.lo[j], s.hi[j])
        elif a < 2 * len(CMD) + 4:
            k = a - 2 * len(CMD); hand, flex = k // 2, k % 2 == 0; f, e = (0, 1) if flex else (1, 0)
            ten[hand, f] = min(1, ten[hand, f] + .35); ten[hand, e] = max(0, ten[hand, e] - .2)
        for h in range(2):
            j = FING[h]; q[j] += .6 * (np.clip(1.5 * (ten[h, 0] - ten[h, 1]), JLO[j], JHI[j]) - q[j])
        return q, ten

    def step(s, a):
        q0 = s.q.copy(); q, ten = s.next_q(a); s.ten = ten
        s.q = q; s._cache = None; blocked = False
        # the wall blocks: pushing into it turns into pressure
        if s.wall is not None:
            P = s.posed()[0]; tipx = s._stip[s.wall[0]] if (s.stick is not None and s.contacts() is not None) else -9
            if P[:, s.wall[0]].max() > s.wall[1] or tipx > s.wall[1]:
                s.push += 1; blocked = True
            elif s.push and P[:, s.wall[0]].max() < s.wall[1] - .1: s.push = 0
        if not blocked:  # hands can't sink into the body more than 2.5 cm
            s.contacts()
            if s._depth > 0.25: blocked = True
        if blocked:
            for fr in (.66, .33, 0.):
                s.q = q0 + fr * (q - q0); s._cache = None
                P = s.posed()[0]; s.contacts()
                ok = s._depth <= 0.25 and (s.wall is None or P[:, s.wall[0]].max() <= s.wall[1])
                if ok: break
        s.dq = s.q - q0; s.t += 1


def rand_q(rng, lo=TLO, hi=THI):
    q = rng.uniform(lo, hi)
    # bias toward natural postures: shrink half the joints toward rest
    m = rng.random(NJ) < .4; q[m] *= rng.uniform(0, .5, m.sum())
    return np.clip(q, lo, hi)


def tip_pos(q, hand):  # fingertip-pad positions for poses q [K,24]
    return point_world(q, [HT[hand][0]])[:, 0]
