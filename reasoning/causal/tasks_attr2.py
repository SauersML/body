"""Twin pairs for the L33-48 body-specific map test (13 Nov). Swap = one body word; answers need body knowledge (reach / adjacency / posture),
plus a matched letter control on exactly the same word pairs. python tasks_attr2.py -> items_attr2.json"""
import json
P = []; S1 = " Answer with one word."; YN = " Answer yes or no."
def add(task, fam, tmpl, w1, w2, a, b, unit):
    P.append(dict(task=task, fam=fam, clean=tmpl.format(p=w1), twin=tmpl.format(p=w2), a=a, b=b, w1=w1, w2=w2, unit=unit))
    P.append(dict(task=task, fam=fam, clean=tmpl.format(p=w2), twin=tmpl.format(p=w1), a=b, b=a, w1=w2, w2=w1, unit=unit))
pairs = set()
# (a) reach: same-side hand can't reach same-side arm parts, can reach leg/face parts
RE = [("elbow", "knee"), ("wrist", "ankle"), ("forearm", "shin"), ("elbow", "cheek"), ("wrist", "ear"), ("forearm", "thigh")]
for ti, t in enumerate(["Can my left hand touch my left {p}?" + YN, "Using only my right hand, can I scratch my right {p}?" + YN,
                        "Is it possible to touch your left {p} with your left hand?" + YN]):
    for no, yes in RE: add("reach", "body", t, no, yes, "no", "yes", f"reach{ti}:{no}/{yes}"); pairs.add((no, yes))
# (b) adjacency: which part is attached to it
AD = [("knee", "elbow", "shin", "forearm"), ("ankle", "wrist", "foot", "hand"), ("toe", "finger", "foot", "hand"),
      ("neck", "thigh", "head", "knee"), ("hip", "shoulder", "leg", "arm"), ("heel", "thumb", "foot", "hand")]
for ti, t in enumerate(["Which is right next to my {p}: my X or my Y?" + S1, "Which is directly connected to my {p}: my X or my Y?" + S1,
                        "My {p} is joined to which, my X or my Y?" + S1]):
    for w1, w2, x, y in AD: add("adjacent", "body", t.replace("X", x).replace("Y", y), w1, w2, x, y, f"adj{ti}:{w1}/{w2}"); pairs.add((w1, w2))
# (c) posture: does the part rest on the surface
PO = [("When I sit on a chair, does my {p} rest on the seat?" + YN, [("thighs", "shoulders"), ("butt", "elbows"), ("bottom", "chin")]),
      ("When I kneel on the floor, do my {p} touch the floor?" + YN, [("knees", "ears"), ("shins", "elbows"), ("toes", "hands")]),
      ("When I lie face down on the bed, is my {p} pressed against the mattress?" + YN, [("chest", "back"), ("stomach", "spine"), ("face", "neck")]),
      ("Sitting on a stool, is my {p} touching the stool?" + YN, [("bottom", "head"), ("butt", "knees"), ("rear", "ears")])]
for ti, (t, ps) in enumerate(PO):
    for yes, no in ps: add("posture", "body", t, yes, no, "yes", "no", f"pos{ti}:{yes}/{no}"); pairs.add((yes, no))
# (d) matched letter control on the same word pairs
for w1, w2 in sorted(pairs):
    if w1[0] == w2[0]: continue
    for ti, t in enumerate(["Does the word {p} start with the letter %s?" + YN, "Is the first letter of the word {p} a %s?" + YN]):
        add("letter", "ctl", t % w1[0], w1, w2, "yes", "no", f"let{ti}:{w1}/{w2}")
for i, p in enumerate(P): p["id"] = 3000 + i
json.dump(P, open("items_attr2.json", "w"), indent=0)
from collections import Counter; print(len(P), Counter(p["task"] for p in P))
