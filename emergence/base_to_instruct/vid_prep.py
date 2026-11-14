# Video data: OLMo-2 stages -> 54 body spots in the model's top-3 PCA dims, Procrustes-chained, plus mesh skinning weights.
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stimuli as S
OUT = sys.argv[1]; WEB = sys.argv[2]; os.makedirs(WEB, exist_ok=True)
BC = os.path.expanduser("~/fig/bodychain4")
P = np.load(f"{BC}/meshP.npy"); F = np.load(f"{BC}/meshF.npy"); LM = json.load(open(f"{BC}/landmarks.json"))
def tok_of(t):
    s = t.split("_")
    return float(s[2][:-1]) if s[1] in ("s1", "s2") else None
order = sorted([f for f in glob.glob(f"{OUT}/olmo_s1_*.npz")], key=lambda f: tok_of(os.path.basename(f)[:-4]))
order += sorted(glob.glob(f"{OUT}/olmo_s2_*.npz")) + [f"{OUT}/olmo_{k}.npz" for k in ["final", "sft", "dpo", "instruct"]]
order = [f for f in order if os.path.exists(f)]
def label(t):
    s = t.split("_")
    if s[1] == "s1": n = float(s[2][:-1]); return "pretraining", (f"{n/1000:.1f} trillion tokens" if n >= 1000 else f"{int(n)} billion tokens")
    if s[1] == "s2": return "mid-training", f"+{int(s[2][:-1])} billion tokens"
    return {"final": ("base model", "after mid-training"), "sft": ("chat tuning", "supervised (SFT)"),
            "dpo": ("chat tuning", "preference (DPO)"), "instruct": ("chat tuning", "final instruct (RL)")}[s[1]]
# body positions of the 54 spots: left landmark vertex, right = mirror vertex
def mirror(v): q = P[v] * np.array([-1, 1, 1]); return int(np.argmin(((P - q) ** 2).sum(1)))
key = {"cheek": "cheek", "ear": "ear"}
vid = []
for l in S.LOCS:
    t = l["type"]; v = LM[t]
    if l["side"] == -1 or l["key"] in ("cheekR", "earR"): v = mirror(v) if P[v][0] > 0 else v
    elif l["side"] == 1 and P[v][0] < 0: v = mirror(v)
    vid.append(v)
XM = P[vid]
Ys = []
for f in order:
    d = np.load(f); k = sorted(d.files, key=lambda s: int(s[5:]))[1]  # middle depth (~0.6)
    X = d[k].astype(np.float64); X -= X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True); X -= X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False); Y = U[:, :3] * s[:3]; Y /= np.sqrt((Y ** 2).sum(1).mean()); Ys.append(Y)
def proc(A, B):  # rotation R (reflections allowed) minimizing |A R - B|
    U, _, Vt = np.linalg.svd(A.T @ B); return U @ Vt
mu = XM.mean(0); sc = np.sqrt(((XM - mu) ** 2).sum(1).mean()); Xb = (XM - mu) / sc
Ys[-1] = Ys[-1] @ proc(Ys[-1], Xb)   # final stage oriented to best match the real body
for i in range(len(Ys) - 2, -1, -1): Ys[i] = Ys[i] @ proc(Ys[i], Ys[i + 1])
fit = [float(1 - ((Y - Xb) ** 2).sum() / (Xb ** 2).sum()) for Y in Ys]
# skinning: each vertex = weighted mix of nearby spots (Gaussian in body space, side-gated)
sd = np.array([l["side"] for l in S.LOCS]); sig = 0.9
D2 = ((P[:, None, :] - XM[None]) ** 2).sum(-1)
W = np.exp(-(D2 - D2.min(1, keepdims=True)) / (2 * sig ** 2)) * np.where(sd[None] == 0, 1.0, 1 / (1 + np.exp(-sd[None] * P[:, 0:1] / 0.25)))
k = 6; idx = np.argsort(-W, 1)[:, :k]; w = np.take_along_axis(W, idx, 1); w /= w.sum(1, keepdims=True)
reg = {"arm": 0, "hand": 1, "leg": 2, "foot": 3, "head": 4, "torso": 5}
def cat(l):
    t = l["type"]
    if t in ("thumb", "index", "middle", "ring", "little", "palm", "backhand", "wrist"): return "hand"
    if t in ("forearm", "elbow", "upperarm", "shoulder"): return "arm"
    if t in ("toes", "sole", "heel", "ankle"): return "foot"
    if t in ("shin", "knee", "thigh", "hip"): return "leg"
    return l["limb"]
cats = [cat(l) for l in S.LOCS]
tris = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]]) if F.shape[1] == 4 else F
json.dump(dict(labels=[label(os.path.basename(f)[:-4]) for f in order], Y=[Y.tolist() for Y in Ys], XM=Xb.tolist(), fit=fit,
               cats=cats, side=sd.tolist(), names=[l["key"] for l in S.LOCS]), open(f"{WEB}/stages.json", "w"))
((P - mu) / sc).astype(np.float32).tofile(f"{WEB}/pos.bin"); tris.astype(np.uint32).tofile(f"{WEB}/idx.bin")
idx.astype(np.uint16).tofile(f"{WEB}/sk_i.bin"); w.astype(np.float32).tofile(f"{WEB}/sk_w.bin")
# per-vertex region color index = region of the heaviest spot
np.array([reg[cats[i]] for i in idx[:, 0]], np.uint8).tofile(f"{WEB}/vreg.bin")
np.array([sd[i] for i in idx[:, 0]], np.int8).tofile(f"{WEB}/vside.bin")
print(len(order), "stages; fit to body", np.round(fit, 2))
