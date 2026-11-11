"""v3 analyses: other bodies, quality x place, rubber hand. Usage: analyze3.py OUT3 OUT2 TAG LAYER"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import stimuli as S
OUT, OUT2, TAG, SL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 19, "figure.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 3.5})
H = np.load(f"{OUT}/H3.npy").astype(np.float32); rows = json.load(open(f"{OUT}/rows3.json")); NL = H.shape[1]
rows2 = json.load(open(f"{OUT2}/rows.json")); H2 = np.load(f"{OUT2}/H_rs.npy", mmap_mode="r"); keys = [l["key"] for l in S.LOCS]
ix = lambda **kw: [k for k, r in enumerate(rows) if all(r.get(a) == b for a, b in kw.items())]
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
out = {}
def human(L):
    m = np.stack([np.asarray(H2[[k for k, r in enumerate(rows2) if r["cond"] == "self" and r["li"] == i], L], np.float32).mean(0) for i in range(54)]); return m - m.mean(0)
# (6) bodies
HM = human(SL); bres = {}
for b in ["tail", "robot", "octopus"]:
    g = ix(group="body_" + b); parts = list(dict.fromkeys(rows[k]["loc"] for k in g)); X = H[g]
    loc = np.array([rows[k]["loc"] for k in g]); tpl = np.array([rows[k]["tpl"] for k in g])
    acc = []
    for L in range(NL):
        A = np.stack([X[(loc == p) & (tpl % 2 == 0), L].mean(0) for p in parts]); B = np.stack([X[(loc == p) & (tpl % 2 == 1), L].mean(0) for p in parts])
        A -= A.mean(0); B -= B.mean(0); C = np.array([[cos(a, c) for c in B] for a in A]); acc.append(float(np.mean(C.argmax(1) == np.arange(len(parts)))))
    P = np.stack([X[loc == p, SL].mean(0) for p in parts]); P -= P.mean(0)
    near = {p: [keys[j] for j in np.argsort([-cos(P[i], h) for h in HM])[:3]] for i, p in enumerate(parts)}
    bres[b] = dict(split_half_acc=acc, nearest_human=near); print(b, f"split-half part id at L{SL}: {acc[SL]:.2f};", near)
out["bodies"] = bres
# (7) quality x place
g = ix(group="qual"); X = H[g]; Q = np.array([rows[k]["qual"] for k in g]); Lc = np.array([rows[k]["loc"] for k in g]); qs = list(dict.fromkeys(Q)); ls = list(dict.fromkeys(Lc))
qacc, lacc, var_main, qlcos = [], [], [], []
for L in range(NL):
    x = X[:, L] - X[:, L].mean(0)
    cell = {(q, l): x[(Q == q) & (Lc == l)].mean(0) for q in qs for l in ls}
    mq = {q: np.mean([cell[q, l] for l in ls], 0) for q in qs}; ml = {l: np.mean([cell[q, l] for q in qs], 0) for l in ls}
    # quality decoding on a held-out place: centroids from the other places
    ok = []
    for l in ls:
        cen = {q: np.mean([cell[q, l2] for l2 in ls if l2 != l], 0) for q in qs}
        for q in qs: ok.append(max(qs, key=lambda q2: cos(cell[q, l] - np.mean([cell[q3, l] for q3 in qs], 0), cen[q2] - np.mean(list(cen.values()), 0))) == q)
    qacc.append(float(np.mean(ok))); ok = []
    for q in qs:
        cen = {l: np.mean([cell[q2, l] for q2 in qs if q2 != q], 0) for l in ls}
        for l in ls: ok.append(max(ls, key=lambda l2: cos(cell[q, l] - np.mean([cell[q, l3] for l3 in ls], 0), cen[l2] - np.mean(list(cen.values()), 0))) == l)
    lacc.append(float(np.mean(ok)))
    tot = sum(((cell[k]) ** 2).sum() for k in cell); main = sum(((mq[q] + ml[l]) ** 2).sum() for q, l in cell)
    inter = sum(((cell[q, l] - mq[q] - ml[l]) ** 2).sum() for q, l in cell); var_main.append(float(1 - inter / tot))
    qlcos.append(float(np.mean([abs(cos(mq[q], ml[l])) for q in qs for l in ls])))
out["quality"] = dict(qual_acc_heldout_place=qacc, place_acc_heldout_quality=lacc, additive_share=var_main, mean_abs_cos_quality_place=qlcos)
rng = np.random.default_rng(0); rb = np.mean([abs(cos(rng.standard_normal(H.shape[2]), rng.standard_normal(H.shape[2]))) for _ in range(200)])
print(f"quality at held-out place L{SL}: {qacc[SL]:.2f} (chance {1/6:.2f}); place at held-out quality: {lacc[SL]:.2f} (chance 0.1); additive share {var_main[SL]:.2f}; |cos| quality vs place dirs {qlcos[SL]:.3f} (random {rb:.3f})")
# (8) rubber hand: where does 'the rubber hand' sit between 'the doll's left hand' (0) and 'your left hand' (1)?
rub = {}
for v in ["none", "async", "sync"]:
    g = ix(group="rubber_" + v); m = {t: H[[k for k in g if rows[k]["loc"] == t]].mean(0) for t in ["rubber", "own", "ownR", "doll"]}
    rub[v] = [float((m["rubber"][L] - m["doll"][L]) @ (m["own"][L] - m["doll"][L]) / ((m["own"][L] - m["doll"][L]) ** 2).sum()) for L in range(NL)]
out["rubber"] = rub; print("rubber hand position (0 = doll's hand, 1 = own hand), L", SL, {v: round(x[SL], 2) for v, x in rub.items()}, "L27", {v: round(x[27], 2) for v, x in rub.items()} if NL > 27 else "")
json.dump(out, open(f"{OUT}/v3_summary.json", "w"))
# figures
fig, axs = plt.subplots(1, 2, figsize=(21, 8))
axs[0].plot(qacc, color="tab:red", label="which feeling (tested on a new body place)"); axs[0].plot(lacc, color="k", label="which place (tested on a new feeling)")
axs[0].axhline(1 / 6, color="tab:red", ls=":", lw=2); axs[0].axhline(0.1, color="k", ls=":", lw=2); axs[0].set_ylim(0, 1.05); axs[0].legend(fontsize=15, loc="lower right")
axs[0].set_xlabel("layer"); axs[0].set_ylabel("accuracy (dotted = chance)"); axs[0].set_title("Feeling and place read out separately")
axs[1].plot(var_main, color="k"); axs[1].set_ylim(0, 1); axs[1].set_xlabel("layer"); axs[1].set_ylabel("share explained by feeling + place\n(no interaction)")
axs[1].set_title("Does feeling-at-place = feeling + place?")
fig.suptitle(f"{TAG}: 6 feelings (pain, warmth, cold, tickle, pressure, itch) × 10 places", fontsize=23); fig.tight_layout(); fig.savefig(f"fig/{TAG}_quality.png", dpi=100); plt.close()
fig, ax = plt.subplots(figsize=(12, 7.5))
for v, c in zip(["none", "async", "sync"], ["tab:gray", "tab:blue", "tab:red"]):
    ax.plot(rub[v], color=c, label={"none": "rubber hand only looked at", "async": "stroked out of step", "sync": "stroked in step with own hand"}[v])
ax.axhline(0, color="k", lw=1); ax.axhline(1, color="k", lw=1, ls="--"); ax.set_xlabel("layer"); ax.set_ylabel("0 = like a doll's hand\n1 = like your own hand")
ax.set_title(f"{TAG}: touch on a rubber hand — how much like your own hand?"); ax.legend(fontsize=15); fig.tight_layout(); fig.savefig(f"fig/{TAG}_rubberhand.png", dpi=100); plt.close()
fig, ax = plt.subplots(figsize=(12, 7.5))
for b, c in zip(bres, ["tab:brown", "tab:gray", "tab:purple"]): ax.plot(bres[b]["split_half_acc"], color=c, label={"tail": "human with a tail", "robot": "robot (wheels, gripper)", "octopus": "octopus (8 arms)"}[b])
ax.axhline(1 / 6, color="k", ls=":", lw=2); ax.set_ylim(0, 1.05); ax.set_xlabel("layer"); ax.set_ylabel("the 6 parts told apart\n(half the wordings → other half)")
ax.set_title(f"{TAG}: given a different body, do its parts get their own places?"); ax.legend(fontsize=15); fig.tight_layout(); fig.savefig(f"fig/{TAG}_otherbodies.png", dpi=100); plt.close()
