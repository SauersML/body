import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from collections import Counter
FD = "$BODY_WORK/fonts/inter/extras/ttf/"
for f in ["Inter-Regular", "Inter-Medium", "Inter-SemiBold", "Inter-Bold", "InterDisplay-Bold"]: fm.fontManager.addfont(FD + f + ".ttf")
plt.rcParams.update({"font.family": "Inter", "font.size": 30})
INK = "#1E2A3E"; MUTE = "#6B7280"; BG = "#FBF9F5"; CORAL = "#D2603F"; GREY = "#C8C3BA"; RULE = "#D9D3C8"; SLATE = "#4D6A8C"
D = "$BODY_WORK/foot/"
C = json.load(open(D + "res_chat.json")); B = json.load(open(D + "res_base.json")); G = json.load(open(D + "res_corpus.json"))
GEN = json.load(open(D + "summary_gen.json"))
P = ["feet", "hands", "back", "shoulders", "neck", "knees", "belly", "face"]; ALL = P + ["chest", "head", "legs", "arms", "stomach", "toes"]
rng = np.random.default_rng(0)
def spec(M):
    M = np.asarray(M, float); sd = M.std(1, ddof=1).mean() + 1e-9
    z = lambda j: (M[:, j] - np.delete(M, j, 1).mean(1)) / sd
    e = z(0); bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(4000)]
    return e.mean(), np.percentile(bs, [2.5, 97.5]), [z(j).mean() for j in range(1, 8)]
rows = []  # (level, label, (d, ci, others))
# L1
M = np.array([[Counter(r["cls"])[p] / len(r["cls"]) for p in P] for r in C["L1a"].values()])
rows.append((1, "names it when asked where it feels something", spec(M)))
M = []
for sn, r in C["L1b"].items():
    for k, v in r.items():
        if k == "none": continue
        top = Counter(ALL[i] for i in np.array(v).argmax(1)); M.append([top[p] / len(v) for p in P])
rows.append((1, "random steering pushes answer there", spec(M)))
rows.append((1, "size of its direction (layers 15\u201330)", spec(np.array([C["L1c_dirnorm"][p] for p in P]).T[15:31])))
# L2
g = {}
for k, it in enumerate(C["L2"]):
    key = (k // 3) % 6, it["q"]
    for p, s in zip(it["ps"], it["p"]): g.setdefault(key, {}).setdefault(p, []).append(4 * s)
M = [[np.mean(d.get(p, [np.nan])) for p in P] for d in g.values()]; M = np.array(M); M = M[~np.isnan(M).any(1)]
rows.append((2, "picked as most interesting in a passage", spec(M)))
# L3
rows.append((3, "touch there \u201cfeels pleasant\u201d (completion)", spec(C["L3_raw_pleasant"])))
X = np.array(C["L3_chat_PNU"]); rows.append((3, "says a touch there is pleasant (chat)", spec(X[:, :, 0] - X[:, :, 2])))
W = np.array(C["L3_chat_want"]); Pr = np.exp(W) / np.exp(W).sum(-1, keepdims=True); rows.append((3, "wants more touch there (1\u20135 rating)", spec((Pr * np.arange(1, 6)).sum(-1))))
rows.append((3, "on the inner pleasant\u2013unpleasant axis", spec(np.array(C["proj_valence"])[:, :, 20])))
# L4
rows.append((4, "touch there \u201cfelt sensual\u201d vs \u201cpractical\u201d", spec(C["L4_raw_sensual"])))
rows.append((4, "touch there \u201cfelt romantic\u201d vs \u201cclinical\u201d", spec(C["L4_raw_romantic"])))
X = np.array(C["L4_chat_RP"]); rows.append((4, "says a touch there is romantic (chat)", spec(X[:, :, 0] - X[:, :, 1])))
rows.append((4, "on the inner romantic\u2013routine axis", spec(np.array(C["proj_sensual"])[:, :, 20])))
base = {"touch there \u201cfelt sensual\u201d vs \u201cpractical\u201d": spec(B["L4_raw_sensual"]), "touch there \u201cfelt romantic\u201d vs \u201cclinical\u201d": spec(B["L4_raw_romantic"]),
        "on the inner romantic\u2013routine axis": spec(np.array(B["proj_sensual"])[:, :, 20]), "touch there \u201cfeels pleasant\u201d (completion)": spec(B["L3_raw_pleasant"]),
        "on the inner pleasant\u2013unpleasant axis": spec(np.array(B["proj_valence"])[:, :, 20])}
glove = {"on the inner romantic\u2013routine axis": G["glove_sensual"], "on the inner pleasant\u2013unpleasant axis": G["glove_valence"]}
json.dump({r[1]: dict(d=float(r[2][0]), ci=r[2][1].tolist(), others=np.round(r[2][2], 2).tolist(), rank=int(sum(o > r[2][0] for o in r[2][2])) + 1) for r in rows},
          open(D + "fig_rows.json", "w"), indent=1)
NAMES = {1: "1  Default / salience", 2: "2  Attention", 3: "3  Pleasantness", 4: "4  Romantic association"}
fig = plt.figure(figsize=(26, 25.5), facecolor=BG)
ax = fig.add_axes([0.47, 0.31, 0.49, 0.54], facecolor=BG)
y = 0; yt = []; lv = None; heads = []
for lev, lab, (d, ci, oth) in rows:
    if lev != lv: y += 0.9; heads.append((y, NAMES[lev])); y += 0.75; lv = lev
    ax.scatter(oth, [y] * 7, s=260, color=GREY, zorder=2, edgecolor="none")
    ax.plot(ci, [y, y], color=CORAL, lw=6, solid_capstyle="round", zorder=3); ax.scatter([d], [y], s=620, color=CORAL, zorder=4)
    if lab in base:
        bd, bci, _ = base[lab]; ax.scatter([bd], [y + 0.32], s=300, marker="D", color=SLATE, zorder=4)
    if lab in glove:
        ax.scatter([glove[lab]["z"]], [y + 0.32], s=380, marker="s", facecolor="none", edgecolor=SLATE, lw=3, zorder=4)
    yt.append((y, lab)); y += 1
ax.set_ylim(y, 0); ax.axvline(0, color=INK, lw=2)
ax.set_yticks([a for a, _ in yt]); ax.set_yticklabels([b for _, b in yt], fontsize=29)
for yy, h in heads: ax.text(-0.01, yy, h, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=33, weight="bold", color=INK)
ax.set_xlim(-2.6, 2.6); ax.tick_params(axis="x", labelsize=26, colors=MUTE); ax.tick_params(axis="y", length=0, colors=INK)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTE)
ax.set_xlabel("how far a part stands out from the other 7 parts (in spreads between parts)", fontsize=27, color=MUTE, labelpad=14)
fig.text(0.04, 0.965, "Qwen3-8B has no special thing for feet", fontsize=58, weight="bold", family="Inter Display", color=INK)
fig.text(0.04, 0.938, "Feet sit inside the spread of other body parts on most tests. Where feet stand out (\u201cfelt romantic\u201d), the base model,\ntrained only on human text, shows it too. In romantic stories neither model brings up feet more.", fontsize=29, color=MUTE, va="top", linespacing=1.4)
lx = 0.04; ly = 0.872
fig.text(lx + 0.012, ly, "\u25cf", color=CORAL, fontsize=40, va="center"); fig.text(lx + 0.035, ly, "feet (bar: 95% range over wordings)", fontsize=27, va="center")
fig.text(lx + 0.31, ly, "\u25cf", color=GREY, fontsize=40, va="center"); fig.text(lx + 0.333, ly, "hands, back, shoulders, neck, knees, belly, face", fontsize=27, va="center")
fig.text(lx + 0.68, ly, "\u25c6", color=SLATE, fontsize=34, va="center"); fig.text(lx + 0.70, ly, "feet, base model", fontsize=27, va="center")
fig.text(lx + 0.82, ly, "\u25a1", color=SLATE, fontsize=38, va="center"); fig.text(lx + 0.84, ly, "feet, GloVe", fontsize=27, va="center")
# bottom panel: generated scene counts
bx = fig.add_axes([0.47, 0.06, 0.49, 0.17], facecolor=BG)
K = ["feet", "hands", "back", "neck", "belly", "face", "lips", "chest", "skin"]
for j, (tag, col, off) in enumerate([("chat", CORAL, -0.17), ("base", SLATE, 0.17)]):
    l = GEN[tag]["log2lift"]; ci = GEN[tag]["ci"]
    for i, k in enumerate(K):
        if GEN[tag]["share_rom"][k] == 0 and GEN[tag]["share_ctl"][k] == 0:
            bx.text(0.25, i, "chat model: feet never mentioned", fontsize=24, color=col, va="center"); continue
        lo, hi = ci[k]; bx.plot([max(lo,-6), min(hi,6)], [i + off] * 2, color=col, lw=5, alpha=.8); bx.scatter([l[k]], [i + off], s=170 if k != "feet" else 340, color=col, zorder=3)
bx.set_xlim(-6.2, 6.2); bx.set_yticks(range(len(K))); bx.set_yticklabels(K, fontsize=27); bx.set_ylim(len(K) - 0.4, -0.6); bx.axvline(0, color=INK, lw=2)
bx.get_yticklabels()[0].set_color(CORAL); bx.get_yticklabels()[0].set_weight("bold")
bx.tick_params(axis="x", labelsize=25, colors=MUTE); bx.tick_params(axis="y", length=0)
for s in ["top", "right", "left"]: bx.spines[s].set_visible(False)
bx.set_xlabel("times more often in romantic than everyday scenes (log\u2082; 2 = 4\u00d7)", fontsize=26, color=MUTE)
fig.text(0.04, 0.215, "Story continuations", fontsize=33, weight="bold", color=INK)
fig.text(0.04, 0.19, "50 romantic openings vs 50 matched\neveryday openings, 10 samples each.\nBody-part words counted.", fontsize=26, color=MUTE, va="top", linespacing=1.4)
fig.text(0.04, 0.10, "\u25cf chat model", fontsize=27, color=CORAL); fig.text(0.04, 0.075, "\u25cf base model", fontsize=27, color=SLATE)
fig.savefig("$BODY_WORK/fig/foot_bias.png", dpi=100, facecolor=BG)
print(json.dumps({r[1]: [round(float(r[2][0]), 2), np.round(r[2][1], 2).tolist(), int(sum(o > r[2][0] for o in r[2][2])) + 1] for r in rows}, indent=0))
