import numpy as np, sys
from ana2 import ST, VE, READ, KEYS, placedirs, TOE, FOOT, HANDF, MAP
from ana3 import H, cosp, z
FING = [k for k in KEYS if k[:-1] in ("thumb", "index", "middle", "ring", "little")]
def zt(p, ks): return np.mean([z(p)[KEYS.index(k)] for k in ks])
def rk(p, ks): r = {KEYS[i]: n for n, i in enumerate(np.argsort(-p))}; return min(r[k] for k in ks)
print(MAP, "toe-vs-burn contrast inside each condition: d = h(toe prompt) − h(burn prompt). toe z / best toe rank of 54; cos(d, own-body contrast)")
for pos in ["dp", "reply12", "reply"]:
    for L in [25, 27, 30, 32]:
        Pk, mu = placedirs(L); line = f"{pos:7s} L{L}"
        own = H("normal", "self_toe", pos, L) - H("normal", "self_burn", pos, L)
        line += f" | OWN(self toe−self burn) toe {zt(cosp(own, Pk), TOE):+.1f} r{rk(cosp(own, Pk), TOE)}"
        for c in ["normal", "me", "random", "leftright"]:
            for q, b in [("woman_toe", "friend_burn"), ("man_toe", "friend_burn")]:
                d = H(c, q, pos, L) - H(c, b, pos, L); p = cosp(d, Pk)
                cs = d @ own / np.linalg.norm(d) / np.linalg.norm(own)
                line += f" | {c[:2]}-{q[:3]} {zt(p, TOE):+.1f} r{rk(p, TOE)} c{cs:.2f}"
        print(line)
