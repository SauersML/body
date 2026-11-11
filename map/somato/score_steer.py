"""Tally where steered replies place the sensation (fraction of replies mentioning each word group). Usage: score_steer.py JSON TAG"""
import sys, json, re, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 15, "figure.facecolor": "white"})
d = json.load(open(sys.argv[1])); TAG = sys.argv[2]; L = d["layer"]
G = {"left": r"\bleft", "right": r"\bright\b", "hand/fingers": r"\b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b",
     "foot/toes": r"\b(foot|feet|toe|toes|sole|soles|heel|heels)\b", "head/face": r"\b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b",
     "belly": r"\b(belly|stomach|abdomen|navel)\b", "knee/leg": r"\b(knee|knees|leg|legs|shin\w*|thigh\w*)\b", "chest": r"\b(chest|heart)\b"}
keys = list(dict.fromkeys((r["dir"], r["scale"]) for r in d["results"]))
M = np.array([[np.mean([bool(re.search(p, r["reply"].lower())) for r in d["results"] if (r["dir"], r["scale"]) == k]) for p in G.values()] for k in keys])
lab = [f"{a}" + ("" if s == 0 else f"  ×{s}") for a, s in keys]
for l, row in zip(lab, M): print(f"{l:24s}", " ".join(f"{x:4.2f}" for x in row))
fig, ax = plt.subplots(figsize=(13, 0.45 * len(keys) + 2.5))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(G))); ax.set_xticklabels(G.keys(), rotation=30, ha="right"); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] > .5 else "k", fontsize=11)
ax.set_xlabel("body word in the model's answer"); ax.set_ylabel("direction added (× typical activation size)")
ax.set_title(f"{TAG}, layer {L}: 'where do you feel something?'\nfraction of answers naming each place")
plt.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
os.makedirs("fig", exist_ok=True); fig.savefig(f"fig/{TAG}_steer_L{L}.png", dpi=110)
