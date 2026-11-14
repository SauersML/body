import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("stats.json")); plt.rcParams.update({"font.size": 34, "axes.spines.top": False, "axes.spines.right": False})
BODY, CTL = "#c0392b", "#7f8c8d"
fig, ax4 = plt.subplots(4, 1, figsize=(26, 46), dpi=100); plt.subplots_adjust(hspace=0.55, top=0.95, bottom=0.04, left=0.13, right=0.97); ax = np.array([[ax4[0], ax4[1]], [ax4[2], ax4[3]]])
def bars(a, labels, pairs, ylab, names=("body tasks", "control tasks")):
    x = np.arange(len(labels)); w = 0.38
    for j, (col, nm) in enumerate([(BODY, names[0]), (CTL, names[1])]):
        v = np.array([p[j] for p in pairs]); a.bar(x + (j - .5) * w, v[:, 0], w, color=col, label=nm, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=8, error_kw=dict(lw=3))
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=28); a.axhline(0, color="k", lw=2); a.set_ylabel(ylab); a.legend(frameon=False, fontsize=30, loc="center right")
# A ablation
A = S["ablation"]; keys = list(A); lab = ["body map\n150 dims", "non-body\nsame size", "random\n150 dims", "body parts\n11 dims", "objects\n11 dims", "random\n11 dims"]
bars(ax[0, 0], lab, [(A[k]["body"], A[k]["ctl"]) for k in keys], "accuracy change")
ax[0, 0].set_title("Removing the body map hurts body tasks no more than control tasks", loc="left", fontweight="bold", fontsize=36, pad=30)
# B hand/foot push
H = S["hf_push"]; ks = ["hand/foot body direction x3", "hand/foot body direction x4", "random direction x3", "random direction B x3"]
bars(ax[0, 1], ["hand/foot\ndirection", "hand/foot\nstronger", "random", "random B"], [(H[k]["body"], H[k]["word"]) for k in ks], "answer shift toward\npushed side (log-odds)", names=("where does it hurt / what moves (64 questions)", "spelling question, same two words (16)"))
ax[0, 1].set_title("Pushing hand↔foot moves body answers, not spelling answers", loc="left", fontweight="bold", fontsize=36, pad=30)
ax[0, 1].text(0.36, 0.97, '"To tiptoe across a room, what do you mainly move: hand or foot?"\n  foot  →  hand      ("Which word starts with \'f\'?" stays foot)', transform=ax[0, 1].transAxes, fontsize=27, color=BODY, va="top")
# C left/right push
L = S["lr_push"]; x = np.arange(3); w = 0.27; a = ax[1, 0]
for j, (t, col, nm) in enumerate([("perspective", BODY, "person facing you"), ("perspective_ctl", "#2980b9", "car facing you"), ("lr_word", CTL, "spelling question")]):
    v = np.array([L[k][t] for k in L]); a.bar(x + (j - 1) * w, v[:, 0], w, color=col, label=nm, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=8, error_kw=dict(lw=3))
a.set_xticks(x); a.set_xticklabels(["left/right\nfrom bodies", "left/right\nfrom objects", "random"], fontsize=28); a.axhline(0, color="k", lw=2); a.set_ylabel("answer shift toward\npushed side (log-odds)"); a.legend(frameon=False, fontsize=28)
a.set_title("Left↔right directions move answers about people and cars alike", loc="left", fontweight="bold", fontsize=36, pad=30)
# D stem edits
T = S["stem"]; ks = list(T)
bars(ax[1, 1], ["swap\nhand↔foot", "swap\nleft↔right", "remove\nbody parts", "swap along\nrandom axis"], [(T[k]["flip_body"], T[k]["flip_ctl"]) for k in ks], "share of answers\nthat change")
ax[1, 1].set_title("Editing body directions while the question is read changes almost nothing", loc="left", fontweight="bold", fontsize=36, pad=30); ax[1, 1].set_ylim(0, 0.3)
fig.text(0.13, 0.985, "Qwen3-8B. One-word answers, scored by log-odds of right vs wrong answer", fontsize=30, va="top")
plt.savefig("../../fig/body_causal.png"); print("ok")
