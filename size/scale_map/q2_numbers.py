"""Q2 numbers: untrained-for-words controller (body-unify CTRL arm, Qwen3-8B), all 8 places pooled."""
import sys, json, numpy as np
sys.argv = ["stats.py", "$BODY_WORK/body3d/unify/r1"]
src = open("$BODY_WORK/body3d/unify/stats.py").read().split("for a in ARMS:\n    s = {}")[0]
exec(src)
A = "ctrl"; ALL = TRN + HLD; out = {"rsa": J[A]["rsa"]["rsa"], "rsa_p": J[A]["rsa"]["rsa_p"]}
for c in ("sensor", "word"):
    out[f"hand_gain_{c}"] = boot(b1_gain(A, ALL, c))
    out[f"hand_touch_{c}"] = wil(np.concatenate([J[A]["b1"][f"{t}|{c}"]["touch"] for t in ALL]))
out["hand_touch_random"] = wil(np.concatenate([J[A]["b1"][f"{t}|random"]["touch"] for t in ALL]))
for c in ("sensor", "word", "random"):
    for al in ("1.5", "3.0"):
        h, dl = b2(A, ALL, c, al); out[f"name_{c}{al}"] = wil(h)
base = np.array(J[A]["b2"]["base"]); out["name_base"] = wil([base[i].argmax() == NONHAND.index(PARTS.index(t)) for t in ALL for i in range(len(base))])
out["chance"] = 1 / len(NONHAND)
json.dump(out, open("q2_numbers.json", "w"), indent=1); print(json.dumps(out))
