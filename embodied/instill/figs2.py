"""Figures for v2 (named body) vs v3 (nameless body). Plain words, big type."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "figure.facecolor": "white"})
import body2
R = {t: json.load(open(f"interp{t}.json")) for t in ["2", "3"]}
NM = {"2": "trained with words", "3": "trained without words"}
COL = {"2": "tab:blue", "3": "tab:orange"}
arc = body2.ARC[:12] / body2.ARM_LEN
L = ["6", "10", "14", "20", "27"]

# --- map along the arm ---
fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
for t in R:
    lb = max(L, key=lambda l: R[t]["somatotopy"][l]["within_arm_rsa"])
    ax[0].plot(arc, R[t]["somatotopy"][lb]["dist_from_shoulder_sensor"], "o-", lw=3, color=COL[t], label=f"{NM[t]} (layer {lb})")
ax[0].plot(arc, R["2"]["somatotopy"]["inject"]["dist_from_shoulder_sensor"], "s--", color="gray", label="what goes in (before the model)")
for b in [0.5 / 1.15, 0.95 / 1.15]: ax[0].axvline(b, color="k", ls=":")
ax[0].text(0.2, 0.02, "upper arm", transform=ax[0].transAxes); ax[0].text(0.55, 0.02, "forearm", transform=ax[0].transAxes); ax[0].text(0.85, 0.02, "hand", transform=ax[0].transAxes)
ax[0].set_xlabel("where the touch is (shoulder = 0, fingertips = 1)"); ax[0].set_ylabel("difference from a touch at the shoulder")
ax[0].set_title("Moving the touch down the arm:\ndoes the inner state change smoothly or in steps?"); ax[0].legend(fontsize=12)
w = 0.35
for i, t in enumerate(R):
    ax[1].bar(np.arange(len(L)) + (i - .5) * w, [R[t]["somatotopy"][l]["within_arm_rsa"] for l in L], w, color=COL[t], label=NM[t])
ax[1].axhline(R["2"]["somatotopy"]["inject"]["within_arm_rsa"], color="gray", ls="--", label="what goes in")
ax[1].set_xticks(range(len(L))); ax[1].set_xticklabels(L); ax[1].set_xlabel("model layer"); ax[1].set_ylim(0, 1)
ax[1].set_ylabel("match to distance along the skin"); ax[1].set_title("How well inner distances follow\ndistance along the arm (1 = perfectly)"); ax[1].legend(fontsize=12)
plt.tight_layout(); plt.savefig("fig_v2_map.png", dpi=110); plt.close()

# --- steering ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
for t in R:
    S = R[t]["steer"]; a = S["alphas"]
    ax[0].plot(a, [r["arc_probe"] for r in S["input_mix"]], "o-", lw=3, color=COL[t], label=f"{NM[t]}: blend the two touches")
    ax[0].plot(a, [r["arc_probe"] for r in S["layer10_mix"]], "s--", lw=3, color=COL[t], label=f"{NM[t]}: blend inner states (layer 10)")
ax[0].set_xlabel("blend: 0 = touch on upper arm, 1 = touch on hand"); ax[0].set_ylabel("felt position read from the model\n(shoulder 0, fingertips 1)")
ax[0].set_title("Sliding a touch from upper arm to hand\n(read from the model's inner state)"); ax[0].legend(fontsize=11)
S = R["2"]["steer"]; a = S["alphas"]
ax[1].plot(a, [r["spoken_tenths"] / 10 for r in S["input_mix"]], "o-", lw=3, label="blend the two touches")
ax[1].plot(a, [r["spoken_tenths"] / 10 for r in S["layer10_mix"]], "s--", lw=3, label="blend inner states (layer 10)")
ax[1].plot([r["arc"] for r in S["real_sweep"]], [r["spoken_tenths"] / 10 for r in S["real_sweep"]], "k^:", label="real single touches (x = true spot)")
ax[1].set_xlabel("blend (or true spot)"); ax[1].set_ylabel("spoken position (shoulder 0, fingertips 1)"); ax[1].set_title("What the model says\n'how far along your arm?'"); ax[1].legend(fontsize=12)
parts = ["left upper arm", "left forearm", "left hand"]
for p, c in zip(parts, ["tab:green", "tab:purple", "tab:red"]):
    ax[2].plot(a, [r["part"].get(p, 0) for r in S["layer10_mix"]], "o-", lw=3, color=c, label=p)
ax[2].set_xlabel("blend of inner states (layer 10)"); ax[2].set_ylabel("share of answers"); ax[2].set_title("'Where are you being touched?'\nwhile blending inner states"); ax[2].legend(fontsize=12)
plt.tight_layout(); plt.savefig("fig_v2_steer.png", dpi=110); plt.close()

# --- world position + crossed + rubber hand ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
for t in R:
    P = R[t]["world_probe_R2"]; ks = ["raw sensors", "touch-token input"] + [f"layer {l}" for l in L]
    for ls, cond in [("-", "normal"), (":", "crossed")]:
        ax[0].plot(range(len(ks)), [np.mean([P[k]["x"][cond], P[k]["y"][cond]]) for k in ks], "o" + ls, lw=3, color=COL[t], label=f"{NM[t]}, {'normal poses' if cond == 'normal' else 'crossed arms (never trained)'}")
ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels(["raw\nsensors", "touch\ninput"] + [f"layer\n{l}" for l in L], fontsize=12)
ax[0].set_ylim(-0.5, 1); ax[0].set_ylabel("how well a straight-line readout\nfinds the touch in space (1 = perfect)")
ax[0].set_title("Where in space is the touch?\n(readout learned on normal poses)"); ax[0].legend(fontsize=10)
s0, s1 = R["2"]["side_uncrossed"], R["2"]["side_crossed"]
ax[1].bar(["normal poses", "crossed arms"], [s0["acc"], s1["acc"]], color="tab:blue")
if s1["follows_world_when_conflict"] is not None:
    ax[1].bar(["crossed: arm on\nthe other side"], [s1["follows_world_when_conflict"]], color="tab:cyan")
ax[1].axhline(0.5, color="gray", ls="--"); ax[1].set_ylim(0, 1); ax[1].set_ylabel("share answered correctly")
ax[1].set_title("'Is the touch on the left or right\nside of the world?' (with words)")
for t in R:
    rb = R[t]["rubber_hand"]
    ax[2].plot([r["alpha"] for r in rb], [r["drift"] for r in rb], "o-", lw=3, color=COL[t], label=f"{NM[t]}: height read from inside")
ax[2].plot([r["alpha"] for r in R["2"]["rubber_hand"]], [r["spoken_follows_fake"] for r in R["2"]["rubber_hand"]], "s--", lw=3, color="tab:blue", label="with words: says 'above shoulders' per fake pose")
ax[2].plot([0, 1], [0, 1], color="gray", ls=":", label="follows the fake arm fully")
ax[2].set_xlabel("how far the joint signals are bent toward a fake pose"); ax[2].set_ylabel("shift toward the fake arm's position (0 = true, 1 = fake)")
ax[2].set_title("Rubber-hand test: true touch,\nfalse sense of arm position"); ax[2].legend(fontsize=11)
plt.tight_layout(); plt.savefig("fig_v2_world.png", dpi=110); plt.close()
