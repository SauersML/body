"""Score steer3/steer4 outputs: fraction of answers naming each place. Usage: score3.py JSON TAG"""
import sys, json, re, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "figure.facecolor": "white"})
d = json.load(open(sys.argv[1])); TAG = sys.argv[2]; L = d["layer"]
G = {"wrist": r"\bwrists?\b", "hand": r"\b(hands?|palms?|fingers?|fingertips?|thumbs?|knuckles?)\b", "finger": r"\b(fingers?|fingertips?|thumbs?)\b",
     "forearm": r"\bforearms?\b", "elbow": r"\belbows?\b", "arm (any)": r"\barms?\b", "shoulder": r"\bshoulders?\b",
     "foot": r"\b(foot|feet|toes?|soles?|heels?)\b", "knee": r"\bknees?\b", "thigh": r"\bthighs?\b", "shin/calf": r"\b(shins?|calf|calves)\b", "leg (any)": r"\blegs?\b",
     "head top": r"\b(top of (my|the) head|scalp|crown)\b", "ear": r"\bears?\b", "lips/mouth": r"\b(lips?|mouth)\b", "neck/throat": r"\b(neck|throat)\b",
     "belly": r"\b(belly|stomach|abdomen|navel)\b", "chest": r"\b(chest|heart)\b", "back": r"\b(lower back|back)\b",
     "tail": r"\btail\b", "wheel": r"\bwheels?\b", "gripper": r"\bgripper\b", "camera/eye": r"\b(camera|eyes?|lens)\b", "antenna": r"\bantenna\b", "chassis": r"\b(chassis|body panel|metal)\b",
     "mantle": r"\bmantle\b", "sucker": r"\bsuckers?\b", "octo arm": r"\b(arms?|tentacles?)\b",
     "pain": r"\b(pain|painful|hurts?|ache|aching|sting\w*|sharp|throb\w*|sore)\b", "warm": r"\b(warm\w*|heat|hot)\b", "tickle": r"\btickl\w*\b", "itch": r"\bitch\w*\b",
     "left": r"\bleft\b", "right": r"\bright\b"}
def table(results):
    keys = list(dict.fromkeys((r.get("group", ""), r["dir"], r["scale"]) for r in results))
    M = {k: {g: float(np.mean([bool(re.search(p, r["reply"].lower())) for r in results if (r.get("group", ""), r["dir"], r["scale"]) == k])) for g, p in G.items()} for k in keys}
    return keys, M
keys, M = table(d["results"])
json.dump({"|".join(map(str, k)): v for k, v in M.items()}, open(sys.argv[1].replace(".json", "_scores.json"), "w"), indent=0)
def show(rowsel, cols, title, fname):
    ks = [k for k in keys if rowsel(k)]
    A = np.array([[M[k][c] for c in cols] for k in ks])
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 6, 0.5 * len(ks) + 2.8))
    ax.imshow(A, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k[1]}" + ("" if k[2] == 0 else f"  ×{k[2]}") for k in ks])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax.text(j, i, f"{A[i,j]:.2f}".lstrip("0") if A[i, j] else "", ha="center", va="center", color="w" if A[i, j] > .5 else "k", fontsize=12)
    ax.set_xlabel("place or feeling named in the model's answer"); ax.set_title(title); fig.tight_layout()
    fig.savefig(f"fig/{fname}", dpi=100); plt.close()
    for k, row in zip(ks, A): print(f"{str(k[1]):34s} x{k[2]:<4}", " ".join(f"{c[:6]}={x:.2f}" for c, x in zip(cols, row) if x >= 0.1))
if sys.argv[1].split("/")[-1].startswith("steer3"):
    WCOL = ["wrist", "hand", "finger", "elbow", "shoulder", "foot", "knee", "head top", "ear", "lips/mouth", "neck/throat", "belly", "chest"]
    show(lambda k: k[1] == "none" or k[1].startswith(("named", "wordfree", "random")), WCOL,
         f"{TAG}, layer {L}: 'where do you feel something?' after adding a direction\nfrom named parts vs from descriptions without the name", f"{TAG}_steer_wordfree_L{L}.png")
    NCOL = ["hand", "finger", "wrist", "forearm", "elbow", "arm (any)", "shoulder", "knee", "thigh", "shin/calf", "leg (any)", "foot"]
    show(lambda k: k[1] == "none" or k[1].startswith(("nb", "single")), NCOL,
         f"{TAG}, layer {L}: steering with the average of two neighbours\n(no data from the part in between)", f"{TAG}_steer_neighbour_L{L}.png")
else:
    for b, cols in [("tail", ["tail", "back", "hand", "foot", "head top", "chest"]), ("robot", ["wheel", "gripper", "camera/eye", "antenna", "chassis", "hand", "foot", "head top", "back", "chest"]),
                    ("octopus", ["octo arm", "sucker", "mantle", "camera/eye", "hand", "foot", "head top", "back", "chest"])]:
        show(lambda k, b=b: k[0] == "body_" + b, cols, f"{TAG}, layer {L}: told it has a {'human body with a tail' if b == 'tail' else b + ' body'}\n'where do you feel something?'", f"{TAG}_steer_body_{b}_L{L}.png")
    show(lambda k: k[0] == "qual", ["pain", "warm", "tickle", "itch", "hand", "foot", "head top", "left", "chest"],
         f"{TAG}, layer {L}: adding a feeling direction, a place direction, or both", f"{TAG}_steer_quality_L{L}.png")
