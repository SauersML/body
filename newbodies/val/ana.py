"""Analyse val.py output. Usage: ana.py OUT LAYER"""
import sys, json, re, numpy as np
sys.path.insert(0, "."); import stimuli as S
OUT, LAYER = sys.argv[1], int(sys.argv[2])
rows = json.load(open(f"{OUT}/rows.json")); LAYERS = json.load(open(f"{OUT}/layers.json"))
H = np.load(f"{OUT}/H.npy", mmap_mode="r"); li = LAYERS.index(LAYER)
keys = [l["key"] for l in S.LOCS]; TYPE = {l["key"]: l["type"] for l in S.LOCS}
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def cent(chat, locs=None):
    ls = locs or list(dict.fromkeys(r["loc"] for r in rows if r["chat"] == chat))
    M = np.stack([np.asarray(H[[k for k, r in enumerate(rows) if r["chat"] == chat and r["loc"] == l], li], np.float32).mean(0) for l in ls])
    return ls, M - M.mean(0)
REG = {"foot": ["toes", "sole", "heel", "ankle"], "hand": ["thumb", "index", "middle", "ring", "little", "palm", "backhand"],
       "head": ["forehead", "crown", "cheek", "lips", "tongue", "ear"], "trunk": ["chest", "belly", "upback", "lowback", "neckB", "neckF"]}
PRED = {"wheelL": "foot", "wheelR": "foot", "gripper": "hand", "camera": "head", "antenna": "head", "chassis": "trunk",
        "mantle": "head|trunk", "eye": "head", "tailbase": "trunk", "lowback": "trunk", "crown": "head", "handL": "hand", "footL": "foot"}
def region(k): return [g for g, ts in REG.items() if TYPE[k] in ts]
out = {}
_, HM = cent("human")
def near(P, parts, ref=HM, tag=""):
    res = {}
    for i, p in enumerate(parts):
        s = np.array([cos(P[i], h) for h in ref]); o = np.argsort(-s)
        hit = None
        if p in PRED: hit = bool(set(region(keys[o[0]])) & set(PRED[p].split("|")))
        res[p] = dict(top=[(keys[j], round(float(s[j]), 3)) for j in o[:3]], hit=hit, sims={keys[j]: round(float(s[j]), 3) for j in range(54)})
        print(f"  {tag}{p:9s}", " ".join(f"{keys[j]} {s[j]:.2f}" for j in o[:3]), "" if hit is None else ("HIT" if hit else "miss"))
    return res
print("== nearest human place (human chat, real narration), layer", LAYER)
for b in ["robot", "glorp", "glorp_nofn", "octopus", "tail"]:
    parts, P = cent(b); print(b); out[b] = near(P, parts)
print("== no-context control: touch sentence alone, both sides without any chat")
_, HN = cent("nocontext_human")
for b in ["robot", "glorp"]:
    parts, P = cent("nocontext_" + b); print("nocontext", b); out["nocontext_" + b] = near(P, parts, HN)
print("== input-embedding control (mean token embedding of the place phrase)")
import os
PE = json.load(open(f"{OUT}/phrase_emb.json")) if os.path.exists(f"{OUT}/phrase_emb.json") else None
hp = {l["key"]: "your " + l["phrase"] for l in S.LOCS}
EH = np.array([PE[hp[k]] for k in keys]) if PE else None
if PE: EH -= EH.mean(0)
for b in (["robot", "glorp", "octopus"] if PE else []):
    ph = {r["loc"]: r["phrase"] for r in rows if r["chat"] == b}; parts = list(ph)
    EP = np.array([PE[ph[p]] for p in parts]); EP -= EP.mean(0); print("emb", b); out["emb_" + b] = near(EP, parts, EH)
print("== same place across two chats (human vs human2, real narration)")
_, H2 = cent("human2")
diag = np.array([cos(HM[i], H2[i]) for i in range(54)])
rng = np.random.default_rng(0); six, best_wrong, shuf = [], [], []
for _ in range(2000):
    sub = rng.choice(54, 6, replace=False); X = H2[sub] - H2[sub].mean(0)
    for j, i in enumerate(sub):
        s = np.array([cos(X[j], h) for h in HM]); six.append(s[i])
        bad = [m for m in range(54) if TYPE[keys[m]] != TYPE[keys[i]]]; best_wrong.append(s[bad].max()); shuf.append(s[rng.choice(bad)])
six, best_wrong, shuf = map(np.array, (six, best_wrong, shuf))
q = lambda a: [round(float(x), 3) for x in np.percentile(a, [10, 25, 50, 75, 90])]
print(" 54-centred same place: pct10/25/50/75/90", q(diag))
print(" 6-subset-centred (matches new-part centring) same place:", q(six))
print(" 6-subset: best other-type place (null for 'closest'):", q(best_wrong))
print(" 6-subset: random other-type place:", q(shuf))
out["twochats"] = dict(diag=diag.tolist(), six=q(six), best_wrong=q(best_wrong), shuf=q(shuf), six_all=six[:5000].tolist(), bw_all=best_wrong[:5000].tolist())
# old imagine-chat robot vs new real-narration robot, same part
json.dump(out, open(f"{OUT}/ana_L{LAYER}.json", "w"))
