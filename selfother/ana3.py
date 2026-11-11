import numpy as np, json, sys
from ana2 import M, ST, VE, READ, KEYS, placedirs, QS, CN, TOE, FOOT, HANDF
def H(c, q, pos, L):
    h = ST[f"{c}|{q}|{pos}"].astype(np.float32); return h[READ.index(L)] if pos == "dp" else h[:, READ.index(L)].mean(0)
def cosp(x, Pk): return Pk @ x / np.linalg.norm(x)
def z(p): return (p - p.mean()) / p.std()
def tg(q): return TOE if "toe" in q else HANDF
pos = sys.argv[1] if len(sys.argv) > 1 else "dp"
print("Δ = cond − normal; z-score of target places within the 54-place profile; rank of best target; 'v' = profile of the added vector itself")
for L in [25, 26, 27, 28, 30, 32]:
    Pk, mu = placedirs(L); line = f"L{L} "
    for c in ["me", "random", "leftright"]:
        v = VE[c]; pv = z(cosp(v, Pk))
        line += f"| {c[:2]} v:{np.mean([pv[KEYS.index(k)] for k in TOE]):+.1f} "
        for q in QS:
            d = H(c, q, pos, L) - H("normal", q, pos, L); p = z(cosp(d, Pk)); r = {KEYS[i]: n for n, i in enumerate(np.argsort(-p))}
            line += f"{q[:6]}:{np.mean([p[KEYS.index(k)] for k in tg(q)]):+.1f}/r{min(r[k] for k in tg(q))} "
    print(line)
print("\nreverse: own-toe difference (self_toe − woman_toe, both normal) vs +me shift on woman_toe")
for L in [25, 26, 27, 28, 30, 32]:
    Pk, mu = placedirs(L)
    ds = H("normal", "self_toe", pos, L) - H("normal", "woman_toe", pos, L); ps = cosp(ds, Pk)
    out = f"L{L} own-toe diff toe z {np.mean([z(ps)[KEYS.index(k)] for k in TOE]):+.1f} |"
    for c in ["me", "random", "leftright"]:
        dm = H(c, "woman_toe", pos, L) - H("normal", "woman_toe", pos, L)
        cs = ds @ dm / np.linalg.norm(ds) / np.linalg.norm(dm); pr = np.corrcoef(ps, cosp(dm, Pk))[0, 1]
        dist_n = np.linalg.norm(H("normal", "woman_toe", pos, L) - H("normal", "self_toe", pos, L)); dist_c = np.linalg.norm(H(c, "woman_toe", pos, L) - H("normal", "self_toe", pos, L))
        out += f" {c[:2]}: cos(Δ,own){cs:+.2f} profile r {pr:+.2f} dist→own {dist_c/dist_n:.2f}x |"
    print(out)
