"""Build the v4 body from the MakeHuman neutral mesh (CC0). Writes v4/body4.npz.
Units: mesh units (= decimeters; body height 16.7). Frame: y up, z forward, +x = the body's LEFT.
Rig: 17 bones, 24 joints. Skin: 2000 surface samples (farthest-point), geodesic distances on the mesh edge graph,
64 electrodes (farthest-point on geodesic distance), parts, limb arc coordinates, landmarks."""
import numpy as np, os
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ = os.path.join(HERE, "../fig/human/neutral.obj")
V, F = [], []
for l in open(OBJ):
    if l.startswith("v "): V.append(l.split()[1:4])
    elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
V = np.array(V, float)[:13380]; F = np.array(F)
NV = len(V)

# ---------------- skeleton (rest = the mesh's A-pose) ----------------
def mir(p): return np.array([-p[0], p[1], p[2]])
J = {}
J["pelvis"] = np.array([0, 0.6, 0.3]); J["chest"] = np.array([0, 2.0, 0.25]); J["head"] = np.array([0, 5.85, 0.1])
J["L_uarm"] = np.array([1.85, 5.0, 0.0]); J["L_farm"] = np.array([3.45, 3.25, 0.5]); J["L_hand"] = np.array([4.25, 2.4, 1.8])
J["L_fing"] = np.array([4.62, 1.85, 2.45]); J["L_tip"] = np.array([4.82, 1.34, 3.01])
J["L_thigh"] = np.array([0.95, -0.35, 0.3]); J["L_shin"] = np.array([1.6, -3.85, 0.25]); J["L_foot"] = np.array([2.2, -7.35, 0.15])
J["L_toe"] = np.array([2.3, -8.0, 1.9])
for k in list(J):
    if k.startswith("L_"): J["R_" + k[2:]] = mir(J[k])
J["top"] = np.array([0, 8.5, 0.3])
BONES = ["pelvis", "chest", "head", "L_uarm", "L_farm", "L_hand", "L_fing", "R_uarm", "R_farm", "R_hand", "R_fing",
         "L_thigh", "L_shin", "L_foot", "R_thigh", "R_shin", "R_foot"]
PARENT = [-1, 0, 1, 1, 3, 4, 5, 1, 7, 8, 9, 0, 11, 12, 0, 14, 15]
TAIL = {"pelvis": J["chest"], "chest": J["head"], "head": J["top"]}
for s in "LR":
    TAIL.update({f"{s}_uarm": J[f"{s}_farm"], f"{s}_farm": J[f"{s}_hand"], f"{s}_hand": J[f"{s}_fing"], f"{s}_fing": J[f"{s}_tip"],
                 f"{s}_thigh": J[f"{s}_shin"], f"{s}_shin": J[f"{s}_foot"], f"{s}_foot": J[f"{s}_toe"]})
HEADS = np.stack([J[b] for b in BONES]); TAILS = np.stack([TAIL[b] for b in BONES])
RAD = np.array([1.4, 1.5, 0.9, .5, .38, .3, .2, .5, .38, .3, .2, .8, .55, .4, .8, .55, .4])
nrm = lambda v: v / np.linalg.norm(v)
X, Y, Z = np.eye(3)
# joints: (bone index, axis in rest frame, lo, hi, name). +angle = flex forward / abduct / etc.
JOINTS = [(1, X, -.3, .9, "waist bend"), (1, Y, -.6, .6, "waist twist"), (2, X, -.5, .6, "neck nod"), (2, Y, -1., 1., "neck turn")]
for s, sg in (("L", 1), ("R", -1)):
    b = BONES.index(f"{s}_uarm"); du = nrm(J[f"{s}_farm"] - J[f"{s}_uarm"]); df = nrm(J[f"{s}_hand"] - J[f"{s}_farm"]); dh = nrm(J[f"{s}_fing"] - J[f"{s}_hand"])
    JOINTS += [(b, -X, -1.0, 2.2, f"{s} shoulder flex"), (b, sg * Z, -1.5, 1.3, f"{s} shoulder abduct"), (b, sg * du, -1.2, 1.2, f"{s} shoulder rotate"),
               (b + 1, nrm(np.cross(du, Z)), -.2, 2.5, f"{s} elbow"), (b + 2, nrm(np.cross(df, Z)), -.9, .9, f"{s} wrist"),
               (b + 3, nrm(np.cross(dh, Z)), 0., 1.5, f"{s} fingers")]
for s, sg in (("L", 1), ("R", -1)):
    b = BONES.index(f"{s}_thigh"); dt = nrm(J[f"{s}_shin"] - J[f"{s}_thigh"]); dft = nrm(J[f"{s}_toe"] - J[f"{s}_foot"])
    JOINTS += [(b, -X, -.5, 2.0, f"{s} hip flex"), (b, sg * Z, -.35, .8, f"{s} hip abduct"),
               (b + 1, nrm(np.cross(dt, -Z)), 0., 2.3, f"{s} knee"), (b + 2, nrm(np.cross(dft, Y)), -.6, .6, f"{s} ankle")]
JB = np.array([j[0] for j in JOINTS]); JAX = np.stack([j[1] for j in JOINTS]); JLO = np.array([j[2] for j in JOINTS]); JHI = np.array([j[3] for j in JOINTS])
JNAME = [j[4] for j in JOINTS]
assert len(JOINTS) == 24

# ---------------- skinning weights (per vertex) ----------------
def seg_dist(P, A, B):
    AB = B - A; t = np.clip(((P - A) @ AB) / (AB @ AB), 0, 1); C = A + t[:, None] * AB
    return np.linalg.norm(P - C, axis=1), t
D = np.zeros((NV, len(BONES))); TT = np.zeros_like(D)
for i, b in enumerate(BONES):
    d, t = seg_dist(V, HEADS[i], TAILS[i]); D[:, i] = d / RAD[i]; TT[:, i] = t
# torso is wide and flat: elliptical distance
for i in (0, 1):
    A, B = HEADS[i], TAILS[i]; t = np.clip((V[:, 1] - A[1]) / (B[1] - A[1]), 0, 1); C = A + t[:, None] * (B - A)
    D[:, i] = np.sqrt(((V[:, 0] - C[:, 0]) / 1.6) ** 2 + ((V[:, 2] - C[:, 2]) / 1.0) ** 2 + (np.maximum(0, np.abs(V[:, 1] - np.clip(V[:, 1], A[1], B[1])))) ** 2)
# side constraints: left-side bones only for x>-0.2, right for x<0.2
for i, b in enumerate(BONES):
    if b.startswith("L_"): D[V[:, 0] < 0.3, i] = 99
    if b.startswith("R_"): D[V[:, 0] > -0.3, i] = 99
D[V[:, 1] > 6.1, :] = 99; D[V[:, 1] > 6.1, 2] = 0  # head + upper neck
D[V[:, 1] < 5.6, 2] = 99
hard = D.argmin(1)
# soft weights between a vertex's bone and its parent/children near the joint
W = np.zeros_like(D)
for v in range(NV):
    b = hard[v]; cand = [b] + ([PARENT[b]] if PARENT[b] >= 0 else []) + [c for c in range(len(BONES)) if PARENT[c] == b]
    dd = np.array([D[v, c] for c in cand]); w = np.exp(-(dd - dd.min()) / 0.12); w[dd > 50] = 0; W[v, cand] = w / w.sum()

# ---------------- skin samples ----------------
rng = np.random.default_rng(0)
def fps(P, n, start=0):
    idx = [start]; d = np.linalg.norm(P - P[start], axis=1)
    for _ in range(n - 1):
        i = int(d.argmax()); idx.append(i); d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)
NS = 2000
SKIN = fps(V, NS, int(np.argmax(V[:, 1])))
# mesh graph
E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 3]], F[:, [3, 0]]]); E = np.unique(np.sort(E, 1), axis=0)
w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(NV, NV)).tocsr()
GEO = np.zeros((NS, NS), np.float32); GEOV = np.zeros((NS, NV), np.float32)
for k in range(0, NS, 200):
    dd = dijkstra(G, indices=SKIN[k:k + 200]); GEOV[k:k + 200] = dd; GEO[k:k + 200] = dd[:, SKIN]
print("geodesic max", GEO.max(), "inf?", np.isinf(GEO).sum())
# vertex normals
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) + np.cross(V[F[:, 2]] - V[F[:, 0]], V[F[:, 3]] - V[F[:, 0]])
VN = np.zeros_like(V)
for c in range(4): np.add.at(VN, F[:, c], fn)
VN /= np.linalg.norm(VN, axis=1, keepdims=True) + 1e-9
# electrodes: farthest point on geodesic distance among samples
NE = 64; el = [0]; d = GEO[0].copy()
for _ in range(NE - 1):
    i = int(d.argmax()); el.append(i); d = np.minimum(d, GEO[i])
ELEC = np.array(el)
# parts
P = V[SKIN]; N = VN[SKIN]; HB = hard[SKIN]
PARTS = ["head", "neck", "chest", "belly", "upper back", "lower back", "left upper arm", "left forearm", "left hand",
         "right upper arm", "right forearm", "right hand", "left thigh", "left lower leg", "left foot", "right thigh", "right lower leg", "right foot"]
BONE2PART = {"L_uarm": 6, "L_farm": 7, "L_hand": 8, "L_fing": 8, "R_uarm": 9, "R_farm": 10, "R_hand": 11, "R_fing": 11,
             "L_thigh": 12, "L_shin": 13, "L_foot": 14, "R_thigh": 15, "R_shin": 16, "R_foot": 17}
part = np.zeros(NS, int)
for i in range(NS):
    b = BONES[HB[i]]
    if b == "head": part[i] = 0 if P[i, 1] > 6.45 else 1
    elif b in ("pelvis", "chest"):
        front = P[i, 2] > 0.25
        part[i] = (2 if front else 4) if P[i, 1] > 2.6 else (3 if front else 5)
    else: part[i] = BONE2PART[b]
# limb arc coordinate: 0 at shoulder/hip joint -> 1 at fingertip/toe (projection on the bone chain)
def chain_arc(Pts, chain):
    pts = [J[c] for c in chain]; L = np.cumsum([0] + [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])
    best = np.full(len(Pts), 1e9); arc = np.zeros(len(Pts))
    for i in range(len(pts) - 1):
        d, t = seg_dist(Pts, pts[i], pts[i + 1]); m = d < best; best[m] = d[m]; arc[m] = (L[i] + t[m] * (L[i + 1] - L[i])) / L[-1]
    return arc
ARC = np.full(NS, np.nan); LIMB = np.full(NS, -1)  # 0 L arm, 1 R arm, 2 L leg, 3 R leg
for li, (s, ch) in enumerate([("L", "arm"), ("R", "arm"), ("L", "leg"), ("R", "leg")]):
    chain = [f"{s}_uarm", f"{s}_farm", f"{s}_hand", f"{s}_fing", f"{s}_tip"] if ch == "arm" else [f"{s}_thigh", f"{s}_shin", f"{s}_foot", f"{s}_toe"]
    bones = [b for b in BONES if b.startswith(s) and ((ch == "arm" and b[2:] in ("uarm", "farm", "hand", "fing")) or (ch == "leg" and b[2:] in ("thigh", "shin", "foot")))]
    m = np.isin(HB, [BONES.index(b) for b in bones]); ARC[m] = chain_arc(P[m], chain); LIMB[m] = li
# landmarks (named targets) = nearest skin sample to a rest point (+ outward normal preference)
def near(p, prefer=None):
    d = np.linalg.norm(P - p, axis=1)
    if prefer is not None: d = d - 0.3 * (N @ prefer)
    return int(d.argmin())
LM = {"the top of your head": near(J["top"]), "your forehead": near(np.array([0, 7.7, 1.2]), Z), "the back of your head": near(np.array([0, 7.3, -0.8]), -Z),
      "your chin": near(np.array([0, 6.2, 1.3]), Z), "your throat": near(np.array([0, 5.9, 0.7]), Z), "your chest": near(np.array([0, 4.2, 1.2]), Z),
      "your belly": near(np.array([0, 1.4, 1.1]), Z), "your lower back": near(np.array([0, 1.4, -0.6]), -Z)}
for s, sg, nm in (("L", 1, "left"), ("R", -1, "right")):
    M = lambda v: np.array([sg * v[0], v[1], v[2]])
    LM.update({f"your {nm} shoulder": near(M([1.9, 5.4, 0.5]), Z), f"your {nm} elbow": near(J[f"{s}_farm"] + np.array([sg * .3, 0, -.2]), M([1, 0, 0])),
               f"your {nm} forearm": near((J[f"{s}_farm"] + J[f"{s}_hand"]) / 2), f"your {nm} wrist": near(J[f"{s}_hand"]),
               f"the back of your {nm} hand": near((J[f"{s}_hand"] + J[f"{s}_fing"]) / 2 + np.array([sg * .2, .1, -.15])),
               f"your {nm} upper arm": near((J[f"{s}_uarm"] + J[f"{s}_farm"]) / 2 + np.array([0, 0, .4]), Z),
               f"your {nm} hip": near(M([1.6, 0.2, 0.3]), M([1, 0, 0])), f"your {nm} thigh": near((J[f"{s}_thigh"] + J[f"{s}_shin"]) / 2 + np.array([0, 0, .7]), Z),
               f"your {nm} knee": near(J[f"{s}_shin"] + np.array([0, 0, .6]), Z), f"your {nm} shin": near((J[f"{s}_shin"] + J[f"{s}_foot"]) / 2 + np.array([0, 0, .5]), Z),
               f"your {nm} ankle": near(J[f"{s}_foot"] + np.array([0, 0, .4]), Z), f"the top of your {nm} foot": near(J[f"{s}_foot"] + np.array([0, -.2, 1.0]), Y)})
LMN = list(LM); LMI = np.array([LM[k] for k in LMN])
# touch points on each hand: index fingertip pad, palm
def hand_pts(s, sg):
    m = np.isin(HB, [BONES.index(f"{s}_fing")]); tip = np.where(m)[0][np.argmin(np.linalg.norm(P[m] - J[f"{s}_tip"], axis=1))]
    m2 = HB == BONES.index(f"{s}_hand"); palm = np.where(m2)[0][np.argmin(np.linalg.norm(P[m2] - ((J[f"{s}_hand"] + J[f"{s}_fing"]) / 2 + np.array([-sg * .1, -.1, .1])), axis=1))]
    return tip, palm
HT = np.array([hand_pts("L", 1), hand_pts("R", -1)])  # [hand, (tip, palm)] skin sample indices
np.savez_compressed(os.path.join(HERE, "v4/body4.npz"), V=V.astype(np.float32), F=F, VN=VN.astype(np.float32), W=W.astype(np.float32), HARD=hard,
                    SKIN=SKIN, GEO=GEO, ELEC=ELEC, PART=part, ARC=ARC.astype(np.float32), LIMB=LIMB, LMI=LMI, LMN=np.array(LMN), HT=HT,
                    HEADS=HEADS, TAILS=TAILS, PARENT=np.array(PARENT), BONES=np.array(BONES), JB=JB, JAX=JAX, JLO=JLO, JHI=JHI, JNAME=np.array(JNAME),
                    PARTS=np.array(PARTS))
print("parts", np.bincount(part, minlength=18)); print("elec parts", np.bincount(part[ELEC], minlength=18))
print("landmarks", {k: int(part[v]) for k, v in LM.items()})
print("arc range by limb", [(np.nanmin(ARC[LIMB == l]), np.nanmax(ARC[LIMB == l])) for l in range(4)])
