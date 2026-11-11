import os
import sys, os, json, numpy as np
sys.path.insert(0, os.path.expandvars("$BODY_REPO/map/somato")); import stimuli as S
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse, FancyBboxPatch
plt.rcParams.update({"font.family": "Avenir Next", "text.color": "#111111"})
SO = os.path.expandvars("$BODY_REPO/map/somato/out2_8b")
# ---------------- data ----------------
H = np.load(f"{SO}/H_rs.npy", mmap_mode="r"); rows = json.load(open(f"{SO}/rows.json"))
nloc = len(S.LOCS); idx = np.zeros((nloc, 8), int)
for k, r in enumerate(rows):
    if r["cond"] == "self": idx[r["li"], r["tpl"]] = k
tk = list(dict.fromkeys(l["type"] for l in S.LOCS)); Tidx = {t: [i for i, l in enumerate(S.LOCS) if l["type"] == t] for t in tk}
G = S.model_rdms()["body path distance"]; rep = [Tidx[t][0] for t in tk]; Gt = G[np.ix_(rep, rep)]; it = np.triu_indices(len(tk), 1)
W = np.load(f"{SO}/wordemb.npy").astype(np.float32); Wt = np.stack([W[Tidx[t]].mean(0) for t in tk]); Wt -= Wt.mean(0)
Dw = ((Wt[:, None] - Wt[None]) ** 2).sum(-1)
def rdm(L):
    a = np.asarray(H[idx, L], np.float32); a = a - a.mean(0, keepdims=True)
    at = np.stack([a[Tidx[t]].mean(0) for t in tk]); A, B = at[:, 0::2].mean(1), at[:, 1::2].mean(1)
    return np.einsum("ijd,ijd->ij", A[:, None] - A[None], B[:, None] - B[None])
def resid(u, z): u, z = rankdata(u), rankdata(z); return u - np.polyval(np.polyfit(z, u, 1), z)
def ps(x, y, z): return spearmanr(resid(x, z), resid(y, z))[0]
NL = H.shape[1]; Ds = [rdm(L) for L in range(NL)]
bylayer = [ps(Ds[L][it], Gt[it], Dw[it]) if L else np.nan for L in range(NL)]
rho_word = spearmanr(Dw[it], Gt[it])[0]
D = Ds[25]; x = Gt[it]; y = resid(D[it], Dw[it]); y = (y - y.min()) / (y.max() - y.min())
rho_main = ps(D[it], Gt[it], Dw[it])
def mds(D, k=2):
    D = np.maximum((D + D.T) / 2, 0); n = len(D); J = np.eye(n) - 1 / n; B = -.5 * J @ D @ J; w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    return v[:, o[:k]] * np.sqrt(np.maximum(w[o[:k]], 0))
ARM = tk[:12]; LEG = tk[12:20]
dep = {t: d for t, _, d in S.ARM} | {t: d for t, _, d in S.LEG}
maps = {}
for nm, sel in [("arm", ARM), ("leg", LEG)]:
    ii = [tk.index(t) for t in sel]; Y = mds(D[np.ix_(ii, ii)])
    if spearmanr(Y[:, 0], [dep[t] for t in sel])[0] > 0: Y[:, 0] *= -1   # proximal = +dim1
    maps[nm] = dict(zip(sel, Y)); maps[nm + "_rho"] = abs(spearmanr(Y[:, 0], [dep[t] for t in sel])[0])
nums = dict(rho_partial_L25=rho_main, rho_raw_L25=spearmanr(D[it], Gt[it])[0], rho_word=rho_word, peak_layer=int(np.nanargmax(bylayer)),
            arm_order_rho=maps["arm_rho"], leg_order_rho=maps["leg_rho"], n_pairs=len(x), bylayer=[None if np.isnan(b) else round(b, 3) for b in bylayer])
json.dump(nums, open(os.path.expandvars("$BODY_REPO/figures/bodychain/numbers.json"), "w"), indent=1); print({k: v for k, v in nums.items() if k != "bylayer"})

# ---------------- canvas (px, y down) ----------------
WPX, HPX = 2400, 1800; PT = 72 / 100
fig = plt.figure(figsize=(WPX / 100, HPX / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WPX); ax.set_ylim(HPX, 0); ax.axis("off"); ax.set_aspect("equal")
K = "#111111"; SKIN = "#E8ECF2"
def T(x, y, s, px=50, **k):
    k.setdefault("ha", "left"); k.setdefault("va", "center"); ax.text(x, y, s, fontsize=px * PT, color=k.pop("color", K), **k)
def cmix(c0, c1, t):
    c0 = np.array(matplotlib.colors.to_rgb(c0)); c1 = np.array(matplotlib.colors.to_rgb(c1)); return tuple(c0 + (c1 - c0) * t)
ARMC = lambda d: cmix("#8FC1F2", "#0A2F8C", d / 6)   # shoulder light -> fingertips dark
LEGC = lambda d: cmix("#F7B48F", "#A2240E", d / 6)
WORDC = "#B8860B"; MODC = "#0A2F8C"

# header
T(40, 62, "Inside Qwen3-8B, the arm lies in order from fingers to shoulder,", 62, weight="bold")
T(40, 136, "and near on the body means near inside the model, beyond word similarity", 62, weight="bold")

# method: one real stimulus
box = FancyBboxPatch((40, 205), 1060, 92, boxstyle="round,pad=0,rounding_size=46", fc="#EEF3FB", ec="none"); ax.add_patch(box)
T(76, 251, "A feather brushes slowly across your left ", 44)
T(76 + 790, 251, "thumb.", 44, weight="bold", color=ARMC(6))
T(1130, 251, "54 places × 8 wordings → Qwen3-8B, layer 25", 44)

# ---------------- body drawing ----------------
def capsule(p0, p1, r0, r1, n=24, **k):
    p0, p1 = np.array(p0, float), np.array(p1, float); d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    a0 = np.arctan2(nrm[1], nrm[0])
    pts = [p1 + r1 * np.array([np.cos(a0 - t), np.sin(a0 - t)]) for t in np.linspace(0, np.pi, n)]
    pts += [p0 + r0 * np.array([np.cos(a0 + np.pi - t), np.sin(a0 + np.pi - t)]) for t in np.linspace(0, np.pi, n)]
    ax.add_patch(Polygon(pts, closed=True, fc=k.get("fc", SKIN), ec="none", zorder=k.get("z", 1)))
cx = 300
ax.add_patch(Ellipse((cx, 440), 150, 182, fc=SKIN, ec="none"))                       # head
capsule((cx, 500), (cx, 580), 42, 46)                                                  # neck
torso = [(cx - 150, 600), (cx + 150, 600), (cx + 168, 640), (cx + 140, 820), (cx + 128, 960), (cx + 150, 1040),
         (cx - 150, 1040), (cx - 128, 960), (cx - 140, 820), (cx - 168, 640)]
ax.add_patch(Polygon(torso, closed=True, fc=SKIN, ec="none", joinstyle="round"))
for s in (1, -1):
    sh = (cx + s * 150, 640); el = (cx + s * 205, 870); wr = (cx + s * 238, 1075)
    capsule(sh, el, 52, 40); capsule(el, wr, 40, 28)
    capsule((cx + s * 85, 1010), (cx + s * 95, 1360), 82, 54); capsule((cx + s * 95, 1360), (cx + s * 92, 1680), 54, 34)
    ax.add_patch(Ellipse((cx + s * 112, 1712), 120, 52, angle=-s * 8, fc=SKIN, ec="none"))
    hc = (cx + s * 246, 1150); ax.add_patch(Ellipse(hc, 82, 110, angle=-s * 6, fc=SKIN, ec="none"))
    tips = [(cx + s * (246 + 64), 1140)] + [(cx + s * (246 + o), 1278 + dy) for o, dy in [(30, -6), (10, 6), (-12, 0), (-32, -16)]]
    bases = [(cx + s * 270, 1125)] + [(cx + s * (246 + o), 1196) for o in (27, 9, -9, -27)]
    for b, t in zip(bases, tips): capsule(b, t, 11, 9)
    if s == 1: TIPS = tips; HC = hc; ARMJ = dict(shoulder=sh, elbow=el, wrist=wr)
# arm nodes on the person's left arm (viewer's right)
sh, el, wr = [np.array(ARMJ[k], float) for k in ("shoulder", "elbow", "wrist")]
bodypos = {"shoulder": sh + (0, -4), "upperarm": (sh + el) / 2, "elbow": el, "forearm": (el + wr) / 2, "wrist": wr,
           "palm": np.array(HC) + (-6, -6), "backhand": np.array(HC) + (2, 40)}
for t, p in zip(["thumb", "index", "middle", "ring", "little"], TIPS): bodypos[t] = np.array(p)
lh, lk, la = np.array((cx + 85, 1030.)), np.array((cx + 95, 1360.)), np.array((cx + 92, 1660.))
legpos = {"hip": lh, "thigh": (lh + lk) / 2, "knee": lk, "shin": (lk + la) / 2, "ankle": la, "heel": la + (-30, 50), "sole": la + (22, 66), "toes": la + (75, 52)}
def node(p, c, r=19, z=5, ring=False):
    ax.add_patch(Circle(p, r + 4, fc="white", ec="none", zorder=z))
    ax.add_patch(Circle(p, r, fc="white" if ring else c, ec=c, lw=5 if ring else 0, zorder=z + 0.1))
for t, p in bodypos.items(): node(p, ARMC(dep[t]), r=14 if t in ARM[:5] else 19, ring=(t == "backhand"))
for t, p in legpos.items(): node(p, LEGC(dep[t]), r=16 if t in ("heel", "sole", "toes") else 19)

# ---------------- model maps ----------------
NAMES = dict(thumb="thumb", index="index", middle="middle", ring="ring", little="little", palm="palm", backhand="knuckles", wrist="wrist",
             forearm="forearm", elbow="elbow", upperarm="upper arm", shoulder="shoulder", toes="toes", sole="sole", heel="heel", ankle="ankle",
             shin="shin", knee="knee", thigh="thigh", hip="hip")
def draw_map(m, sel, col, x0, y0, wpx, hpx, scale):
    P = {t: np.array([x0 + wpx / 2 + scale * m[t][1], y0 + hpx / 2 - scale * m[t][0]]) for t in sel}
    groups = sorted(set(dep[t] for t in sel))
    cent = {d: np.mean([P[t] for t in sel if dep[t] == d], 0) for d in groups}
    # chain line through depth-group centroids, then spokes to members
    chain = [cent[d] for d in groups]
    for a, b in zip(chain, chain[1:]): ax.plot([a[0], b[0]], [a[1], b[1]], color=col(3), lw=7, solid_capstyle="round", zorder=2, alpha=.55)
    for t in sel:
        c = cent[dep[t]]
        if np.hypot(*(P[t] - c)) > 1: ax.plot([c[0], P[t][0]], [c[1], P[t][1]], color=col(dep[t]), lw=3, zorder=2, alpha=.5)
    for t in sel: node(P[t], col(dep[t]), r=20, ring=(t == "backhand"))
    return P
y0m = 330
PA = draw_map(maps["arm"], ARM, ARMC, 660, 380, 560, 700, 58)
PL = draw_map(maps["leg"], LEG, LEGC, 660, 1080, 560, 700, 52)
for P in (PA, PL):
    for t, p in P.items():
        if t in ARM[:5]: continue
        side = 1 if p[0] >= 940 else -1
        T(p[0] + side * 34, p[1], NAMES[t], 46, ha="left" if side > 0 else "right")
fc_ = np.mean([PA[t] for t in ARM[:5]], 0)
T(fc_[0] + 0, fc_[1] + 0, "", 46)
fig.savefig(os.path.expandvars("$BODY_REPO/figures/out/body_map_chain.png"), dpi=100, facecolor="white")
