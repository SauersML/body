import numpy as np, sys
from ana2 import ST, VE, READ, KEYS, placedirs, TOE, FOOT, HANDF
from ana3 import H, cosp, z
pos = sys.argv[1] if len(sys.argv) > 1 else "dp"
FING = [k for k in KEYS if k[:-1] in ("thumb", "index", "middle", "ring", "little")]
def zt(p, ks): return np.mean([z(p)[KEYS.index(k)] for k in ks])
print(pos, "(A) Δ minus the added vector: toe z for toe prompts, finger z for burn prompts")
for L in [25, 26, 27, 28, 30, 32]:
    Pk, mu = placedirs(L); line = f"L{L}"
    for c in ["me", "random", "leftright"]:
        v = VE[c]; parts = []
        for q, ks in [("woman_toe", TOE), ("man_toe", TOE), ("self_toe", TOE), ("friend_burn", FING), ("self_burn", FING)]:
            d = H(c, q, pos, L) - H("normal", q, pos, L) - v; parts.append(f"{zt(cosp(d, Pk), ks):+.1f}")
        line += f" | {c[:2]} " + " ".join(parts)
    print(line)
print(pos, "(B) contrast: shift on woman_toe minus shift on friend_burn (raw Δ, so the vector cancels). toe z, finger z (expect toe +, finger −)")
for L in [25, 26, 27, 28, 30, 32]:
    Pk, mu = placedirs(L); line = f"L{L}"
    for c in ["me", "random", "leftright"]:
        d = (H(c, "woman_toe", pos, L) - H("normal", "woman_toe", pos, L)) - (H(c, "friend_burn", pos, L) - H("normal", "friend_burn", pos, L))
        p = cosp(d, Pk); r = {KEYS[i]: n for n, i in enumerate(np.argsort(-p))}
        line += f" | {c[:2]} toe {zt(p, TOE):+.1f} (r{min(r[k] for k in TOE)}) fing {zt(p, FING):+.1f}"
    # baseline reference: self_toe − woman_toe minus (self_burn − friend_burn)
    print(line)
Pk, mu = placedirs(27)
pv = cosp(VE["me"], Pk); o = np.argsort(-pv)
print("me-vector place profile at L27, top:", [KEYS[i] for i in o[:8]], "bottom:", [KEYS[i] for i in o[-8:]])
pv = cosp(VE["leftright"], Pk); o = np.argsort(-pv)
print("lr-vector top:", [KEYS[i] for i in o[:8]], "bottom:", [KEYS[i] for i in o[-8:]])
