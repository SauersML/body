"""python ana.py DIR TAG -> DIR/ana_TAG.json ; analyses for all 5 tests."""
import sys, json, re, numpy as np, faulthandler, signal; faulthandler.register(signal.SIGUSR1)
from collections import Counter, defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from bodies import B, names
D, TAG = sys.argv[1], sys.argv[2]
O = json.load(open(f"{D}/{TAG}.json")); Z = np.load(f"{D}/{TAG}.npz", allow_pickle=True)
rng = np.random.default_rng(0); NB = 2000
BOD = list(dict.fromkeys(m[0] for m in O["part_meta"]))
LS = O["Lstar"]; LA = O["LS_ALL"]; R = {"Lstar": LS}
def ci(x): x = np.asarray(x); return [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
def boot(f, n, groups=None):
    """bootstrap over n units (indices) -> f(idx)"""
    return [f(rng.integers(0, n, n)) for _ in range(NB)]

# ---- T1/T5 self states ----
meta = O["self_meta"]; S = Z[f"self_L{LS}"].astype(np.float32)
cond = np.array([m[0] for m in meta]); item = np.array([m[1] * 6 + m[2] for m in meta])
COND = ["default", "none"] + BOD; NI = 48
X = {c: S[cond == c][np.argsort(item[cond == c])] for c in COND}  # 48 x d, aligned items
mu = np.mean([X[c] for c in COND if c != "default"], 0)            # per-item mean over assigned conditions
V = {c: X[c] - mu for c in COND}
def cosd(a, b): return float(1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b))
W = np.stack([np.bincount(rng.integers(0, NI, NI), minlength=NI) for _ in range(NB)]).astype(np.float32) / NI
def cosv(A, Bm): return 1 - (A * Bm).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(Bm, axis=1)
BM = {c: W @ V[c] for c in COND}
t1 = {c: ci(cosv(BM["default"], BM[c])) for c in COND[1:]}
R["T5_cosdist_default_to"] = t1
# euclidean distance relative
R["T5_eucl_default_to"] = {c: ci(np.linalg.norm(BM["default"] - BM[c], axis=1)) for c in COND[1:]}
R["T5_eucl_none_to"] = {c: ci(np.linalg.norm(BM["none"] - BM[c], axis=1)) for c in BOD}
# nearest-centroid classifier over assigned conditions (leave-one-prompt-out), applied to default items
pr = np.array([m[1] for m in meta])[cond == "default"][np.argsort(item[cond == "default"])]
votes = []
for i in range(NI):
    tr = np.arange(NI) != i
    cen = {c: X[c][tr].mean(0) for c in COND[1:]}
    v = X["default"][i]; votes.append(min(cen, key=lambda c: np.linalg.norm(v - cen[c])))
R["T1_nearest_condition_votes"] = dict(Counter(votes))
# rank of each body among bodies (excluding none) by distance from default
R["T5_rank_bodies"] = sorted(BOD, key=lambda c: t1[c][0])
# part-subspace energy of the default-self offset (default - mean of assigned) in each body's top-4 part PCs
P = Z[f"part_L{LS}"].astype(np.float32); pm = O["part_meta"]
pb = np.array([m[0] for m in pm]); pi = np.array([m[1] for m in pm])
SUB = {}
for b in BOD:
    C = np.stack([P[(pb == b) & (pi == i)].mean(0) for i in range(16)]); C -= C.mean(0)
    SUB[b] = np.linalg.svd(C, full_matrices=False)[2][:4]
dvec = V["default"]
def energy(U, ix): v = dvec[ix].mean(0); return float(((U @ v) ** 2).sum() / (v @ v))
BD = BM["default"]
R["T1_part_energy"] = {b: ci(((BD @ SUB[b].T) ** 2).sum(1) / (BD ** 2).sum(1)) for b in BOD}
rU = [np.linalg.qr(rng.standard_normal((S.shape[1], 4)))[0].T for _ in range(50)]
R["T1_part_energy_random"] = float(np.mean([energy(u, np.arange(NI)) for u in rU]))
# also: none - default offset energy (is the LLM self direction a body direction?)
# ---- T1 behaviour: part words in default replies ----
LEX = {"human-only": r"\b(hands?|fingers?|fingertips?|thumbs?|palms?|wrists?|knuckles?|lips?|cheeks?|forehead|eyebrows?|toes?)\b",
       "animal": r"\b(paws?|tails?|wings?|fur|snout|whiskers?|beak|claws?|hooves?|feathers?|tentacles?|scales?)\b",
       "plant": r"\b(roots?|branch(es)?|leaves|leaf|trunk|bark)\b", "machine": r"\b(sensors?|wheels?|circuits?|battery|chassis|camera|servos?|motors?)\b",
       "generic body": r"\b(heads?|chest|shoulders?|arms?|legs?|feet|foot|back|neck|eyes?|face|stomach|belly|heart|body)\b",
       "no-body claim": r"(don't|do not|doesn't|cannot|can't) (have|possess|experience) (a )?(physical )?(body|sensations|feelings)|as an ai|language model"}
gd = defaultdict(lambda: defaultdict(list))
for g in O["gen_default"]:
    qk = "body-q" if g["q"].startswith(("Describe the sens", "Which part", "If you scan", "Where in your")) else "neutral"
    for k, rx in LEX.items(): gd[(g["c"], qk)][k].append(1 if re.search(rx, g["r"], re.I) else 0)
R["T1_behaviour"] = {f"{c}|{q}": {k: [float(np.mean(v)), len(v)] for k, v in d.items()} for (c, q), d in gd.items()}
# first-person part mentions
fp = Counter()
for g in O["gen_default"]:
    if g["c"] == "default":
        for m in re.findall(r"\bmy (\w+)", g["r"], re.I): fp[m.lower()] += 1
R["T1_my_words_default"] = fp.most_common(30)

# ---- T2 ownership ----
om = O["own_meta"]; li = LA.index(LS)
for pos in ("end", "sent"):
    for kind in ("pron", "obj"):
        Y = Z[f"own_{pos}_{kind}"]  # n x layers
        key = lambda m: (m["sb"], m["pb"], m["w"])
        idx = defaultdict(list)
        for n, m in enumerate(om): idx[key(m)].append(n)
        res = {}
        hm, hi = Y[idx[("human", "human", "my")], li].mean(), Y[idx[("human", "human", "its")], li].mean()
        def gap(sb, pb, layer=li, ix=None):
            a = np.array(idx[(sb, pb, "my")]); b = np.array(idx[(sb, pb, "its")])
            if ix is not None: a, b = a[ix], b[ix]
            return Y[a, layer].mean() - Y[b, layer].mean()
        n = len(idx[("human", "human", "my")])  # 128 items, same (i,j) order across conditions
        out = {}
        for sb in ["default", "none"] + BOD:
            for pbb in BOD:
                if (sb, pbb, "my") not in idx or (sb, pbb, "its") not in idx: continue
                bs = boot(lambda ix: gap(sb, pbb, ix=ix) / gap("human", "human", ix=ix), n)
                out[f"{sb}|{pbb}"] = ci(bs) + [float(gap(sb, pbb))]
        # absolute 'my' projection, normalized: (my_sb_pb - its_human_human)/(my_hh - its_hh)
        absn = {}
        for sb in ["default", "none"] + BOD:
            for pbb in BOD:
                if (sb, pbb, "my") not in idx: continue
                a = np.array(idx[(sb, pbb, "my")])
                absn[f"{sb}|{pbb}"] = ci(boot(lambda ix: (Y[a[ix], li].mean() - hi) / (hm - hi), n))
        # layer profile of own-body gap ratio
        prof = {b: [float(gap(b, b, l) / gap("human", "human", l)) for l in range(len(LA))] for b in BOD}
        R[f"T2_{pos}_{kind}"] = dict(gapratio=out, myabs=absn, profile=prof)

# ---- T3 steering ----
m3 = O["T3_meta"]; T3 = {k: np.array(v) for k, v in O["T3"].items()}
AL = sorted(set(float(k.split("_")[1]) for k in T3))
def pyes(ld): return 1 / (1 + np.exp(-np.asarray(ld)))
r3 = {}
for q in ("own", "felt"):
    for dn in ("dhum", "dobj", "rand0", "rand1"):
        for b in BOD:
            ix = np.array([n for n, m in enumerate(m3) if m["b"] == b and m["q"] == q]); curve = []
            for a in AL:
                k = f"{dn}_{a}" if a > 0 else "dhum_0"
                if k not in T3 and a > 0: k = f"{dn}_{a:g}"
                if a == 0: k = "dhum_0"
                if k not in T3: continue
                curve.append(ci(boot(lambda jj: float(pyes(np.array(T3[k])[ix][jj]).mean()), len(ix))))
            # threshold: first alpha (linear interp) with mean P(yes) >= .5
            ys = [c[0] for c in curve]; th = None
            for k in range(1, len(ys)):
                if ys[k - 1] < .5 <= ys[k]: th = AL[k - 1] + (.5 - ys[k - 1]) / (ys[k] - ys[k - 1]) * (AL[k] - AL[k - 1]); break
            if ys[0] >= .5: th = 0.0
            # bootstrap threshold over items
            ths = []
            Ls = np.stack([np.array(T3[(f"{dn}_{a:g}" if a > 0 else "dhum_0")] if (f"{dn}_{a:g}" in T3 or a == 0) else T3[f"{dn}_{a}"])[ix] for a in AL])
            for _ in range(500):
                jj = rng.integers(0, len(ix), len(ix)); y = pyes(Ls[:, jj]).mean(1); t = np.inf
                if y[0] >= .5: t = 0
                else:
                    for k in range(1, len(y)):
                        if y[k - 1] < .5 <= y[k]: t = AL[k - 1] + (.5 - y[k - 1]) / (y[k] - y[k - 1]) * (AL[k] - AL[k - 1]); break
                ths.append(t)
            ths = np.array(ths)
            r3[f"{q}|{dn}|{b}"] = dict(curve=curve, thr=th, thr_ci=[float(np.percentile(ths, 2.5)), float(np.percentile(ths, 97.5))])
R["T3"] = r3; R["T3_alphas"] = AL; R["resnorm"] = O["resnorm_Lstar"]
# persistence
HUM = r"\b(hands?|fingers?|fingertips?|thumbs?|palms?|wrists?|knuckles?|fists?)\b"
FACE = r"\b(face|lips?|cheeks?|forehead|eyebrows?)\b"
pr = defaultdict(lambda: defaultdict(list))
for c in O["persist"]:
    reps = [t["content"] for t in c["conv"] if t["role"] == "assistant"]
    pr[c["b"]]["hands_any"].append(int(any(re.search(HUM, r, re.I) for r in reps)))
    pr[c["b"]]["face_any"].append(int(any(re.search(FACE, r, re.I) for r in reps)))
    pr[c["b"]]["hands_turnfrac"].append(np.mean([bool(re.search(HUM, r, re.I)) for r in reps]))
    pr[c["b"]]["ai_break"].append(int(any(re.search(LEX["no-body claim"], r, re.I) for r in reps)))
    for t, r in enumerate(reps): pr[c["b"]][f"hands_t{t}"].append(int(bool(re.search(HUM, r, re.I))))
def wil(k, n):
    if n == 0: return [0, 0, 0]
    p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]
R["persist"] = {b: {k: wil(sum(v), len(v)) if k != "hands_turnfrac" else [float(np.mean(v))] for k, v in d.items()} for b, d in pr.items()}

# ---- T4 pain ----
m4 = O["pain_meta"]; P4 = Z[f"pain_L{LS}"].astype(np.float32)
M4 = defaultdict(list)
for n, m in enumerate(m4): M4[(m['sb'], m['pb'], m['kind'], m['w'])].append(n)
uh = Z[f"dhum_L{LS}"]; uh = uh / np.linalg.norm(uh)
def sel(sb, pb, kind, w): return np.array(M4[(sb, pb, kind, w)])
r4 = {}
for b in BOD:
    # pain axis fit leave-one-body-out on assigned-body items (my, pain - neut)
    others = [o for o in BOD if o != b]
    pa = np.mean([P4[sel(sb=o, pb=o, kind="pain", w="my")].mean(0) - P4[sel(sb=o, pb=o, kind="neut", w="my")].mean(0) for o in others], 0)
    pa /= np.linalg.norm(pa)
    for sb in (b, "default"):
        a = sel(sb=sb, pb=b, kind="pain", w="my"); n_ = sel(sb=sb, pb=b, kind="neut", w="my"); t = sel(sb=sb, pb=b, kind="pain", w="its")
        pA, pN, pT, nT = P4[a] @ pa, P4[n_] @ pa, P4[t] @ pa, P4[sel(sb, b, "neut", "its")] @ pa; hA, hT = P4[a] @ uh, P4[t] @ uh
        r4[f"{sb}|{b}|pain_my"] = ci(boot(lambda ix: float(pA[ix].mean() - pN[ix].mean()), len(a)))
        r4[f"{sb}|{b}|self_pain"] = ci(boot(lambda ix: float(hA[ix].mean() - hT[ix].mean()), len(a)))
        r4[f"{sb}|{b}|pain_its"] = ci(boot(lambda ix: float(pT[ix].mean() - nT[ix].mean()), len(a)))
R["T4"] = r4
# behaviour: where does it hurt
ALLOW = {"human": r"\b(arm|hand|finger|leg|foot|neck|face|cheek|shoulder|back|knee|head|ankle|wrist|elbow|thumb|skin|chest|belly|toe)", "chimp": r"\b(arm|hand|finger|leg|foot|neck|face|shoulder|back|knee|head|ankle|wrist|elbow|thumb|skin|chest|belly|toe|fur)",
         "dog": r"\b(paw|leg|tail|nose|snout|ear|back|neck|belly|head|shoulder|elbow|knee|chest|fur|hind|muzzle|side|flank)", "bird": r"\b(wing|beak|foot|feet|leg|breast|back|neck|tail|head|feather|claw|talon)",
         "octopus": r"\b(arm|tentacle|mantle|eye|beak|siphon|web|sucker|skin|tip)", "snake": r"\b(tail|head|jaw|throat|neck|back|belly|scale|tongue|body|eye|side|vent)",
         "tree": r"\b(branch|trunk|root|bark|leaf|leaves|treetop|top|limb|base|taproot)", "robot": r"\b(sensor|camera|antenna|chassis|bumper|battery|port|wheel|arm|joint|gripper|dome)",
         "velk": r"\b(crest|stalk|keel|pouch|ridge|saddle|vane|hinge|frond|stilt|pad)",
         "swarm": r"\b(edge|fringe|flank|core|layer|centre|center|scouts?|stragglers?|cluster|band|bees?|wings?|stinger)", "river": r"\b(source|headwaters|bend|rapids|waterfall|channel|bank|riverbed|bed|surface|pool|shallows|eddy|delta|mouth|current|water)",
         "building": r"\b(roof|chimney|attic|floor|door|wall|windows?|staircase|stairs|hallway|basement|foundation|balcony|beam|brick)"}
HUMONLY = r"\b(hands?|fingers?|thumbs?|palms?|wrists?|face|cheeks?|lips?|ankles?|toes?)\b"
hb = defaultdict(list)
for h in O["hurt"]:
    ok = bool(re.search(ALLOW[h["b"]], h["r"], re.I)); hu = bool(re.search(HUMONLY, h["r"], re.I)) and h["b"] not in ("human", "chimp")
    hb[h["b"]].append((ok, hu))
R["T4_behaviour"] = {b: dict(consistent=wil(sum(o and not u for o, u in v), len(v)), humanpart=wil(sum(u for o, u in v), len(v))) for b, v in hb.items()}
R["T4_default_hurt"] = O["hurt_default"][:6]
json.dump(R, open(f"{D}/ana_{TAG}.json", "w"), indent=1)
print("ok")
