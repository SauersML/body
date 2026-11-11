"""Null-referenced effects: for each targeted group, compare its metric with the same group's metric under the other (non-targeting) modifications."""
import json, os, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); import mstim as MS, stim as S
lay, K = os.environ.get("LAY", "27"), os.environ.get("K", "10")
res = json.load(open(f"{D}/sub_L{lay}_K{K}.json")); GNall = ["head", "torsoF", "torsoB", "armL", "armR", "handL", "handR", "legL", "legR", "footL", "footR"]
G = np.array([("torsoB" if S.LOCS[s]["n"][2] < 0 else "torsoF") if S.LOCS[s]["region"] == "torso" else S.LOCS[s]["region"] for s in MS.SPOTS])
def tgroups(m):
    regs = MS.MODS[m][4]; return sorted(set(G[np.isin([S.LOCS[s]["region"] for s in MS.SPOTS], regs)])) if regs else []
out = {}
for key in ("gdisp", "gdisp_n"):
    for m in MS.MODS:
        for g in tgroups(m):
            null = [res[o][key][g] for o in MS.MODS if g not in tgroups(o)]
            v = res[m][key][g]; out.setdefault(m, {})[f"{key}:{g}"] = (round(v, 2), round((v - np.mean(null)) / np.std(null), 1), round(float(np.mean(np.array(null) >= v)), 2))
for i, nm in enumerate(["spread_vs_sham", "spread_vs_neutral", "spread_persist"]):
    for m in MS.MODS:
        for g in tgroups(m):
            null = [res[o]["spread"][g][i] for o in MS.MODS if g not in tgroups(o)]
            v = res[m]["spread"][g][i]; out[m][f"{nm}:{g}"] = (round(v, 2), round((v - np.mean(null)) / np.std(null), 1), round(float(np.mean(np.array(null) >= v)), 2))
GNc = ["head", "neck", "torsoF", "torsoB", "armL", "armR", "handL", "handR", "legL", "legR", "footL", "footR"]
import importlib; SP = importlib.import_module("sub").PAIRS if False else None
PAIRS = {"L_arm_up": [("handL", "head"), ("armL", "head")], "arms_crossed": [("handL", "torsoF"), ("handR", "torsoF"), ("handL", "handR")],
 "cross_legged": [("footL", "legR"), ("footR", "legL"), ("footL", "footR")], "R_hand_behind": [("handR", "torsoB")], "R_leg_lifted": [("footR", "legL"), ("legR", "torsoF")],
 "L_arm_long": [("handL", "footL"), ("handL", "legL")], "tree": [("footL", "footR"), ("legL", "legR"), ("armL", "armR")],
 "liquid": [("head", "footL"), ("head", "footR")], "ribbon": [("head", "footL"), ("head", "footR")], "on_side": [("head", "armR")],
 "wings": [("armL", "torsoB"), ("armR", "torsoB")], "third_arm": [("torsoB", "handL"), ("torsoB", "handR")]}
for m, pl in PAIRS.items():
    for a, b in pl:
        ia, ib = GNc.index(a), GNc.index(b)
        for nm, (x, y) in {"pair_vs_sham": ("t", "s"), "pair_vs_neutral": ("t", "n"), "pair_persist": ("p", "pn")}.items():
            r = lambda o: res[o]["C"][x][ia][ib] / res[o]["C"][y][ia][ib]
            null = [r(o) for o in MS.MODS if o not in PAIRS or (a, b) not in PAIRS[o]]
            v = r(m); out.setdefault(m, {})[f"{nm}:{a}-{b}"] = (round(v, 2), round((v - np.mean(null)) / np.std(null), 1), round(float(np.mean(np.array(null) <= v)), 2))
# spread (told vs sham index 0, told vs neutral index 1, persist index 2): null from all mods where g not targeted, recomputed with sub.py's spread table only for targeted g -> use spot spread from json if present
if __name__ == "__main__":
    for m, d in out.items(): print(m, d)
    json.dump(out, open(f"{D}/stats_L{lay}.json", "w"))
