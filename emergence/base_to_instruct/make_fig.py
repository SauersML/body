# compute/fig/body_emergence.png: body measures across OLMo-2 7B training, plus Qwen3-8B base -> chat.
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 34, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5, "xtick.major.size": 10, "ytick.major.size": 10})
R = {}
for f in glob.glob(os.path.expanduser("~/body/emerge/res/*.json")):
    r = json.load(open(f)); R[r["tag"]] = r
def band(r, fmt, k):
    nl = r["nl"]; L = [l for l in range(nl + 1) if .5 <= l / nl <= .8]; return float(np.nanmean([r["fmt"][fmt][k][l] for l in L]))
def steer(r, fmt, fr="first"): return r["fmt"][fmt]["steer"][f"{fr}_d0.65_a0.5"]["spec"]
pre = sorted([t for t in R if t.startswith("olmo_s1_")], key=lambda t: int(t[8:13]))
tok = [int(t[8:13]) for t in pre]
post = [t for t in ["olmo_s2_021B", "olmo_s2_038B", "olmo_final", "olmo_sft", "olmo_dpo", "olmo_instruct"] if t in R]
xp = np.log10(tok); x0 = xp.max(); XPOST = {t: x0 + .55 * (i + 1) for i, t in enumerate(post)}
PL = {"olmo_s2_021B": "mid", "olmo_s2_038B": "mid", "olmo_final": "base", "olmo_sft": "SFT", "olmo_dpo": "DPO", "olmo_instruct": "RL"}
BLUE, ORANGE, GRAY, GREEN = "#1f5fbf", "#e8771e", "#9a9a9a", "#2a9d5b"
rows = [
  ("The body map forms at 5–10 billion tokens; chat tuning leaves it unchanged",
   "match to distance\nalong the skin",
   [("touch sentences: \"A feather brushes your left thumb\"", BLUE, lambda r, f: band(r, f, "rsa_touch"), "dlg", False),
    ("part names only: \"Word: left thumb\"", GRAY, lambda r, f: band(r, f, "rsa_list"), "dlg", False)]),
  ("A place vector steers \"It was in my ___\" from 10B tokens on, and \"her ___\" equally",
   "steered place gain\n(nats)",
   [("\"…It was in my ___\"", BLUE, lambda r, f: steer(r, f, "first"), "dlg", False),
    ("\"…It was in her ___\"", GRAY, lambda r, f: steer(r, f, "third"), "dlg", False)]),
  ("\"Mine\" follows whose body is touched only in a tuned model's own chat format",
   "\"she touches YOUR X\"\nscored as yours",
   [("plain-text transcript (every model)", GRAY, lambda r, f: band(r, f, "own_locus"), "dlg", False),
    ("the model's own chat format (tuned models)", BLUE, lambda r, f: band(r, f, "own_locus"), "chat_embod", True)]),
  ("Described wings get a root-to-tip map from 1B tokens, the same when called \"glorp\"",
   "match to distance\non the wing",
   [("named \"wing\"", BLUE, lambda r, f: band(r, f, "wing_grid"), "dlg", False), ("same text, named \"glorp\"", GRAY, lambda r, f: band(r, f, "glorp_grid"), "dlg", False)]),
]
FMT = "dlg"
fig, axs = plt.subplots(len(rows), 2, figsize=(30, 8.2 * len(rows)), gridspec_kw={"width_ratios": [5, 1.05], "wspace": .08, "hspace": .95})
for (head, ylab, series), (ax, aq) in zip(rows, axs):
    for lab, col, fn, FMT, only in series:
        if only:
            tp = [t for t in post if FMT in R[t]["fmt"]]
            ax.plot([XPOST[t] for t in tp], [fn(R[t], FMT) for t in tp], "-o", color=col, lw=5, ms=16, label=lab)
            aq.plot([1], [fn(R["qwen_chat"], FMT)], "o", color=col, ms=18); continue
        y = [fn(R[t], FMT) for t in pre]; ax.plot(xp, y, "-o", color=col, lw=5, ms=13, label=lab or None)
        yp = [fn(R[t], FMT) for t in post]; ax.plot([x0] + [XPOST[t] for t in post], [y[-1]] + yp, "-o", color=col, lw=5, ms=13)
        qb, qc = fn(R["qwen_base"], FMT), fn(R["qwen_chat"], FMT)
        aq.plot([0, 1], [qb, qc], "-o", color=col, lw=5, ms=16)
    lo = min(min(a.get_ylim()[0] for a in (ax, aq)), 0); hi = max(a.get_ylim()[1] for a in (ax, aq))
    for a in (ax, aq): a.set_ylim(lo, hi)
    ax.axvspan(x0 + .25, XPOST[post[-1]] + .3, color="#f1f1f1", zorder=0)
    ax.set_xticks(list(np.log10([1, 10, 100, 1000])) + [XPOST[t] for t in post])
    ax.set_xticklabels(["1B", "10B", "100B", "1T"] + [PL[t] for t in post])
    ax.set_xlim(-.1, XPOST[post[-1]] + .3); ax.set_ylabel(ylab, fontsize=34, labelpad=14)
    ax.text(0, 1.2, head, transform=ax.transAxes, fontsize=44, fontweight="bold", va="bottom")
    if any(s[0] for s in series): ax.legend(frameon=False, fontsize=30, loc="best")
    aq.set_xticks([0, 1]); aq.set_xticklabels(["base", "chat"]); aq.set_xlim(-.4, 1.4); aq.set_yticklabels([])
axs[0, 1].set_title("Qwen3-8B", fontsize=38, pad=20)
axs[0, 0].text(1.8, axs[0, 0].get_ylim()[1], "pretraining tokens", fontsize=30, ha="center", va="bottom")
axs[0, 0].text(x0 + .55 * (len(post) + 1) / 2, axs[0, 0].get_ylim()[1], "after pretraining", fontsize=30, ha="center", va="bottom")
fig.text(.0, 1.0, "OLMo-2 7B at 31 public training stages, and Qwen3-8B base vs chat.\nEvery model reads the same embodiment chat as plain text; layers at 50–80% depth.", fontsize=38, va="bottom")
fig.savefig(os.path.expanduser("~/fig/body_emergence.png"), dpi=100, bbox_inches="tight", facecolor="white")
print("saved")
