"""python score.py OUTDIR: score steered replies by regex; print table and examples."""
import json, re, sys, numpy as np
O = sys.argv[1]; S = json.load(open(f"{O}/steer.json"))
RX = dict(wing=r"\bwing", tip=r"\btip|\bend of (my|the)|outer edge|outermost|\bedge", root=r"\broot|\bbase\b|shoulder blade|where (they|it|my wings?) (join|meet|connect)",
          left=r"\bleft\b", right=r"\bright\b", hand=r"\bhand|finger", glorp=r"glorp", feather=r"feather")
MX = dict(unfold=r"unfold|spread|\bopen|extend|stretch|wide|fann?(ing|ed)? out|outstretch|lift|rais", fold=r"(?<!un)fold|tuck|\bclos|draw(n|ing)? in|against my back|rest(ing)?\b|still")
rows = []
for r in S["runs"]:
    rx = RX if r["q"] == "where" else MX
    sc = {k: float(np.mean([bool(re.search(v, s, re.I)) for s in r["samples"]])) for k, v in rx.items()}
    rows.append(dict(q=r["q"], ctx=r["ctx"], dir=r["dir"], alpha=r["alpha"], n=len(r["samples"]), **sc, ex=r["samples"][:3]))
json.dump(rows, open(f"{O}/steer_scored.json", "w"), indent=1)
for r in rows:
    print(r["q"], r["ctx"], f'{r["dir"]:11s}', r["alpha"], " ".join(f"{k}={r[k]:.2f}" for k in (RX if r["q"] == "where" else MX)))
if "-v" in sys.argv:
    for r in rows:
        if r["alpha"] in (0, 1.0): print("\n##", r["q"], r["ctx"], r["dir"], r["alpha"]); [print("  -", e[:160]) for e in r["ex"][:2]]
# stricter: located at the tip (tip word, no shoulder/blade/back/base/root)
TIPONLY = lambda s: bool(re.search(r"\btips?\b", s, re.I)) and not re.search(r"shoulder|blade|\bback\b|\bbase\b|\broot", s, re.I)
ROOTONLY = lambda s: bool(re.search(r"shoulder|blade|\broot|\bbase\b|where (they|it|my wings?) (join|meet|connect|attach)", s, re.I)) and not re.search(r"\btips?\b", s, re.I)
for r, raw in zip(rows, S["runs"]):
    r["tiponly"] = float(np.mean([TIPONLY(s) for s in raw["samples"]])); r["rootonly"] = float(np.mean([ROOTONLY(s) for s in raw["samples"]]))
    r["wingtiponly"] = float(np.mean([TIPONLY(s) and bool(re.search(r"wing|feather", s, re.I)) for s in raw["samples"]]))
json.dump(rows, open(f"{O}/steer_scored.json", "w"), indent=1)
print("\nstrict")
for r in rows:
    if r["q"] == "where": print(r["ctx"], f'{r["dir"]:11s}', r["alpha"], f'wingtip-only={r["wingtiponly"]:.2f} root-only={r["rootonly"]:.2f} wing={r["wing"]:.2f}')
