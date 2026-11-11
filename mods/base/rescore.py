import json, re, collections
S = json.load(open('steer.json'))
CF = {"L_hand_huge": (r"left hand|hand", r"huge|big|large|swoll|swell|heav|enormous|giant|twice|massive|thick"),
 "feet_huge": (r"\bfoot|\bfeet", r"huge|big|large|enormous|giant|massive|long"),
 "head_tiny": (r"head", r"tiny|small|little|walnut|shr[iu]nk"),
 "L_arm_long": (r"left arm|\barm|fingers", r"long|stretch|floor|length|reach"),
 "L_arm_up": (r"left arm|\barm|shoulder", r"\bup\b|above|overhead|over my head|raised|rais|lift|ceiling|holding"),
 "third_arm": (r"\barm", r"third|extra|new arm|additional|from my back|between my shoulder"),
 "tail": (r"tail", r"tail"), "wings": (r"wing", r"wing"),
 "R_hand_gone": (r"right (hand|arm|wrist)|wrist", r"gone|missing|stump|no longer|absen|ends at|end of my|used to be|smooth|healed"),
 "L_fin": (r"hand|fin\b|finger", r"\bfin\b|fus|flipper|webb|seal|joined"),
 "R_arm_stone": (r"right arm|\barm", r"stone|numb|hard|grey|gray|rock"),
 "back_sensitive": (r"back", r"tingl|sensitiv|electric|buzz|alive"),
 "tree": (r"root|trunk|bark|branch", r"."), "liquid": (r"liquid|melt|pool|flow", r"."),
 "L_warm_R_cold": (r"right", r"cold|ice|chill|freez|cool")}
def joint(s, p, c, w=60):
    for mp in re.finditer(p, s, re.I):
        seg = s[max(0, mp.start() - w): mp.end() + w]
        if re.search(c, seg, re.I): return 1
    return 0
T = collections.defaultdict(dict)
for r in S:
    p, c = CF[r["mod"]]; T[r["mod"]][r["cond"]] = sum(joint(s, p, c) for s in r["samples"]) / len(r["samples"])
json.dump(T, open("steer_joint.json", "w"), indent=1)
conds = ['base_unsteered','told_unsteered','persist_unsteered','told_ctx+told_dir@0.5','told_ctx+told_dir@1.0','base_ctx+told_dir@1.0','base_ctx+base_dir@1.0']
print("mod", conds)
for m, d in T.items(): print(f"{m:15s}", "  ".join(f"{d.get(c, float('nan')):.2f}" for c in conds))
