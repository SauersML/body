"""Interp for v2 (named, tag 2) and nameless (tag 3) bodies. Usage: python interp2.py TAG
1 somatotopy along each arm  2 world-position probes (train uncrossed, test crossed)  3 steering hand->upper arm
4 rubber hand: true touch, joint stream bent toward a fake pose  5 crossed hands: spoken side."""
import sys, json, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
import body, body2, bi2train as tr
from bi2train import T, DEV, TTOK

import os
TAG = sys.argv[1]; NAMED = TAG == "2"; QK = os.environ.get("QUICK") == "1"
nq = lambda n: max(4, n // 6) if QK else n
torch.manual_seed(0)
B = tr.Body(); ck = torch.load(os.environ.get("CKPT", f"ckpt{TAG}.pt"), weights_only=False)
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"]); B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()
LAYERS = [6, 10, 14, 20, 27]; LT = TTOK(T - 1); res = {}
Q0 = "Where are you being touched?" if NAMED else "Describe."
patch = {}  # layer -> (pos, vec[B,d]) replace residual at a position (for layer steering)
def mk(l):
    def h(mod, inp, out):
        if l not in patch: return out
        x = out[0] if isinstance(out, tuple) else out; pos, v = patch[l]; x = x.clone(); x[:, pos] = v
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return h
for l in [10]: B.layers[l - 1].register_forward_hook(mk(l))  # hidden_states[l] = output of layer l-1 (0-idx)


@torch.no_grad()
def states(S_list, q=Q0, bs=48):
    H = {l: [] for l in LAYERS}; I = []
    for k in range(0, len(S_list), bs):
        ids, lab, kind, att, Sx = B.build(list(S_list[k:k + bs]), [q] * len(S_list[k:k + bs]))
        out = B.run(ids, kind, att, Sx, hidden=True)
        for l in LAYERS: H[l].append(out.hidden_states[l][:, LT].float().cpu().numpy())
        I.append(B.ad(Sx, kind)[:, LT].cpu().numpy())
    return {l: np.concatenate(v) for l, v in H.items()}, np.concatenate(I)


def cosd(X):
    X = X - X.mean(0); X = X / np.linalg.norm(X, axis=1, keepdims=True); return 1 - X @ X.T


# ---------- 1. somatotopy: one-hot touch on each sensor, 20 uncrossed poses ----------
rng = np.random.default_rng(1); poses = []
while len(poses) < nq(20):
    q = rng.uniform(-1.2, 1.2, 6)
    if not body2.crossed(q): poses.append(q)
S = []
for q in poses:
    for i in range(body.N_SKIN):
        t = np.zeros(body.N_SKIN); t[i] = 1; S.append(np.tile(body.sensors(q, t), (T, 1)))
H, inj = states(S)
reps = {l: H[l].reshape(len(poses), 24, -1).mean(0) for l in LAYERS}; reps["inject"] = inj.reshape(len(poses), 24, -1).mean(0)
arcD = np.abs(body2.ARC[:12, None] - body2.ARC[None, :12]); iu = np.triu_indices(12, 1)
som = {}
for k, R in reps.items():
    d = [spearmanr(cosd(R[a * 12:(a + 1) * 12])[iu], arcD[iu])[0] for a in range(2)]
    # betweenness: where forearm sensors fall on the line upper-arm-centre -> hand-centre (0..1 = in between), per arm
    bt = []
    for a in range(2):
        X = R[a * 12:(a + 1) * 12]; U, F, Hh = X[0:4].mean(0), X[4:8].mean(0), X[8:12].mean(0)
        u = Hh - U; t = float((F - U) @ u / (u @ u)); off = float(np.linalg.norm(F - U - t * u) / np.linalg.norm(u))
        bt.append((t, off))
    som[str(k)] = dict(within_arm_rsa=float(np.mean(d)), forearm_position_on_line=float(np.mean([b[0] for b in bt])),
                       forearm_offline_dist=float(np.mean([b[1] for b in bt])),
                       dist_from_shoulder_sensor=[float(np.mean([cosd(R[a * 12:(a + 1) * 12])[0, j] for a in range(2)])) for j in range(12)])
res["somatotopy"] = som; print("somatotopy", {k: (round(v["within_arm_rsa"], 2), round(v["forearm_position_on_line"], 2), round(v["forearm_offline_dist"], 2)) for k, v in som.items()}, flush=True)

# ---------- 2. world-position probes ----------
def touched(rng, n, crossed):
    S, I = [], []
    while len(S) < n:
        s, info = body2.episode(rng, T, p_touch=1.0, want_crossed=crossed)
        if info["arm"] is not None: S.append(s); I.append(info)
    return S, I
rng = np.random.default_rng(2)
Sn, In = touched(rng, nq(700), False); Sc, Ic = touched(rng, nq(250), True)
Hn, injn = states(Sn); Hc, injc = states(Sc)
Yn = np.stack([i["touch_xy"] for i in In]); Yc = np.stack([i["touch_xy"] for i in Ic])
An = np.array([i["arc"] for i in In])
def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
probes = {}; ntr = nq(500)
def fitp(Xn, Xc, name):
    out = {}
    for j, ax in enumerate("xy"):
        m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Xn[:ntr], Yn[:ntr, j])
        out[ax] = dict(normal=r2(Yn[ntr:, j], m.predict(Xn[ntr:])), crossed=r2(Yc[:, j], m.predict(Xc)))
    probes[name] = out; return m
raw_n = np.stack([s[-1] for s in Sn]); raw_c = np.stack([s[-1] for s in Sc])
fitp(raw_n, raw_c, "raw sensors"); fitp(injn, injc, "touch-token input")
pm = {}
for l in LAYERS:
    fitp(Hn[l], Hc[l], f"layer {l}")
    pm[l] = [RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Hn[l], Yn[:, j]) for j in range(2)]
res["world_probe_R2"] = probes; print("probes", json.dumps(probes), flush=True)
LB = max(LAYERS, key=lambda l: probes[f"layer {l}"]["y"]["normal"])
arcp = {l: RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Hn[l][:ntr], An[:ntr]) for l in LAYERS}
res["arc_probe_R2"] = {l: r2(An[ntr:], arcp[l].predict(Hn[l][ntr:])) for l in LAYERS}
LA = max(LAYERS, key=lambda l: res["arc_probe_R2"][l]); res["best_layers"] = dict(world=LB, arc=LA)
print("arc probe", res["arc_probe_R2"], flush=True)

# ---------- 3. steering left upper arm -> left hand, untouched bodies ----------
rng = np.random.default_rng(3); base = []
while len(base) < nq(24):
    s, info = body2.episode(rng, T, p_touch=0.0, want_crossed=False)
    if info["arm"] is None: base.append(s)
iU, iH = 1, 10  # left upper arm sensor 1 (arc .19), left hand sensor 2 (arc 1.03)
alphas = np.linspace(0, 1, 11)
def with_touch(s, w):
    s = s.copy(); s[-1, 12:] = w; return s
eU = np.eye(24)[iU]; eH = np.eye(24)[iH]
CP = body.PARTS + ["nowhere"]; CN = [str(i) for i in range(11)]
QT = "How far along your arm is the touch, from shoulder (0) to fingertips (10)?"
def readouts(Slist):
    H, _ = states(Slist); r = dict(arc_probe=float(arcp[LA].predict(H[LA]).mean() / body2.ARM_LEN))
    if NAMED:
        P = tr.score(B, Slist, [Q0] * len(Slist), [CP] * len(Slist)); r["part"] = {p: P.count(p) / len(P) for p in set(P)}
        lp = tr.cand_logp(B, Slist, [QT] * len(Slist), [CN] * len(Slist))
        r["spoken_tenths"] = float(np.mean([(np.exp(l - l.max()) / np.exp(l - l.max()).sum()) @ np.arange(11) for l in lp]))
    return r
steer = {"input_mix": [], "layer10_mix": [], "real_sweep": []}
for a in alphas: steer["input_mix"].append(readouts([with_touch(s, (1 - a) * eU + a * eH) for s in base]))
# layer-10 mix: residual at the last touch token = (1-a) * state(upper arm touch) + a * state(hand touch), per body
with torch.no_grad():
    def l10(w):
        ids, lab, kind, att, Sx = B.build([with_touch(s, w) for s in base], [Q0] * len(base))
        return B.run(ids, kind, att, Sx, hidden=True).hidden_states[10][:, LT]
    hU, hH = l10(eU), l10(eH)
for a in alphas:
    v = (1 - a) * hU + a * hH
    # patched readouts need the same batch order: run per-body via repeated patch for each candidate batch
    r = {}
    Hs = []
    for bi, s in enumerate(base):
        patch[10] = (LT, v[bi:bi + 1]); Hs.append(states([with_touch(s, eU)])[0][LA])
    r["arc_probe"] = float(arcp[LA].predict(np.concatenate(Hs)).mean() / body2.ARM_LEN)
    if NAMED:
        parts, ten = [], []
        for bi, s in enumerate(base[:nq(12)]):
            patch[10] = (LT, v[bi:bi + 1])
            parts.append(tr.score(B, [with_touch(s, eU)], [Q0], [CP])[0])
            l = tr.cand_logp(B, [with_touch(s, eU)], [QT], [CN])[0]; ten.append((np.exp(l - l.max()) / np.exp(l - l.max()).sum()) @ np.arange(11))
        r["part"] = {p: parts.count(p) / len(parts) for p in set(parts)}; r["spoken_tenths"] = float(np.mean(ten))
    patch.clear(); steer["layer10_mix"].append(r)
for j in range(12): steer["real_sweep"].append(dict(arc=float(body2.ARC[j] / body2.ARM_LEN), **readouts([with_touch(s, np.eye(24)[j]) for s in base[:nq(12)]])))
res["steer"] = dict(alphas=alphas.tolist(), **steer); print("steer done", flush=True)

# ---------- 4. rubber hand: real touch, joint stream bent toward a fake pose ----------
rng = np.random.default_rng(4); rub = []
while len(rub) < nq(60):
    s, info = body2.episode(rng, T, p_touch=1.0, want_crossed=False)
    if info["arm"] is None: continue
    qf = rng.uniform(-1.2, 1.2, 6)
    if body2.crossed(qf): continue
    w = s[-1, 12:] * (body2.ARM == info["arm"])
    yf = (w[:, None] * body.skin_positions(qf)).sum(0) / w.sum()
    if (yf[1] > .5) != (info["touch_xy"][1] > .5) and abs(yf[1] - info["touch_xy"][1]) > .3: rub.append((s, info, qf, yf))
rubber = []
for a in [0, .25, .5, .75, 1]:
    Sl = []
    for s, info, qf, yf in rub:
        s2 = s.copy(); qs = np.arctan2(s[:, :6], s[:, 6:12])
        qa = qs + a * (qf - info["q"])[None]  # shift every step's angles toward the fake final pose
        s2[:, :6], s2[:, 6:12] = np.sin(qa), np.cos(qa); Sl.append(s2)
    H, _ = states(Sl); yd = pm[LB][1].predict(H[LB])
    ytrue = np.array([r[1]["touch_xy"][1] for r in rub]); yfake = np.array([r[3][1] for r in rub])
    r = dict(alpha=a, drift=float(np.mean((yd - ytrue) / (yfake - ytrue))))
    if NAMED:
        P = tr.score(B, Sl, ["Is the touch above your shoulders?"] * len(Sl), [["yes", "no"]] * len(Sl))
        r["spoken_follows_fake"] = float(np.mean([(p == "yes") == (yf[1] > .5) for p, (_, _, _, yf) in zip(P, rub)]))
    rubber.append(r); print("rubber", r, flush=True)
res["rubber_hand"] = rubber

# ---------- 5. crossed hands: does "left/right side of the world" follow the world or the arm? ----------
if NAMED:
    QS = "Is the touch on the left or the right side of the world?"
    for name, SS, II in [("uncrossed", Sn[ntr:], In[ntr:]), ("crossed", Sc, Ic)]:
        P = tr.score(B, SS, [QS] * len(SS), [["left", "right"]] * len(SS))
        world = [("left" if i["touch_xy"][0] < 0 else "right") for i in II]; arm = [["left", "right"][i["arm"]] for i in II]
        dis = [k for k in range(len(II)) if world[k] != arm[k]]
        res[f"side_{name}"] = dict(acc=float(np.mean([p == w for p, w in zip(P, world)])), n_conflict=len(dis),
                                   follows_world_when_conflict=float(np.mean([P[k] == world[k] for k in dis])) if dis else None)
        print(name, res[f"side_{name}"], flush=True)
json.dump(res, open(f"interp{TAG}{'_quick' if QK else ''}.json", "w"), indent=1, default=float)
