"""Forearm control stimuli: 80 forearm spots (2 arms x 10 positions x 4 faces), described with
original joint-relative wording and with wordings that avoid it."""
import random
T = [.04, .15, .25, .33, .5, .67, .77, .85, .92, .97]
ORIGW = ["just below the elbow", "two finger-widths below the elbow", "a quarter of the way from the elbow to the wrist", "a hand-width below the elbow",
         "halfway between the elbow and the wrist", "a hand-width above the wrist", "three finger-widths above the wrist", "two finger-widths above the wrist",
         "one finger-width above the wrist", "just above the wrist"]
FACES = ["inside", "back", "thumb side", "little-finger side"]
FRAC = ["one twenty-fifth", "about a seventh", "a quarter", "a third", "half", "two thirds", "about three quarters", "about six sevenths", "about eleven twelfths", "about twenty-nine thirtieths"]
def pos(kind, k):
    t = T[k]
    if kind == "pct": return f"{round(t*100)} percent of the way along it from the elbow to the wrist"
    if kind == "cmE": return f"{round(t*26)} cm from the elbow"
    if kind == "pt": return f"at point {k+1} of ten evenly spaced points running from the elbow to the wrist"
    if kind == "inE": return f"{round(t*10.2*2)/2:g} inches from the elbow"
    if kind == "frac": return f"{FRAC[k]} of the way from the elbow to the wrist"
POSK = ["pct", "cmE", "pt", "inE", "frac"]
SIDE = {"para1": ["on the palm side", "on the hairy side", "on the thumb edge", "on the pinky edge"],
        "para2": ["on the soft underside", "on the same side as the back of the hand", "along the edge in line with the thumb", "along the edge in line with the little finger"],
        "tech": ["on the volar surface", "on the dorsal surface", "on the radial border", "on the ulnar border"],
        "obl": ["on the face where the veins show", "on the face that tans in summer", "on the edge that points up when you shake hands", "on the edge that rests on the desk when you write"]}
def phr(S, k, f, pk, sk, order):
    if order == 0: return f"your {S} forearm, {pos(pk, k)}, {SIDE[sk][f]}"
    return f"your {S} forearm {SIDE[sk][f]}, {pos(pk, k)}"
FRAMINGS = ["A feather brushes slowly across {L}.", "Someone rests a fingertip very lightly on {L}.", "Someone presses a thumb firmly into {L}.",
            "A sudden sharp pain shoots through {L}.", "An ice cube touches {L}, cold and wet, then slides away.", "A mosquito bites {L}.",
            "The doctor presses two fingers on {L} and asks if it hurts.", "A cold metal chain slides across {L}."]
def spots():
    return [dict(arm=S, k=k, f=f, t=T[k]) for S in ["left", "right"] for k in range(10) for f in range(4)]
def phrases():
    """(cond, spot index, variant, phrase)"""
    out = []; rng = random.Random(0)
    for si, s in enumerate(spots()):
        S, k, f = s["arm"], s["k"], s["f"]
        out.append(("orig", si, 0, f"the {FACES[f]} of your {S} forearm, {ORIGW[k]}"))
        v = 0
        for pk in POSK:
            for sk in ["para1", "para2"]: out.append(("para", si, v, phr(S, k, f, pk, sk, v % 2))); v += 1
        v = 0
        for pk in POSK:
            for sk in ["tech", "obl"]: out.append(("syn", si, v, phr(S, k, f, pk, sk, v % 2))); v += 1
        for v in range(10):
            out.append(("mix", si, v, phr(S, k, f, rng.choice(POSK), rng.choice(list(SIDE)), rng.randrange(2))))
    return out
def rows():
    R = []
    for c, si, v, p in phrases():
        for ti, Tm in enumerate(FRAMINGS):
            a = Tm.index("{L}")
            if "," in p and Tm[a + 3] not in ".,": Tm = Tm.replace("{L}", "{L},")
            R.append(dict(cond=c, si=si, v=v, tpl=ti, phrase=p, text=Tm.format(L=p), span=[a, a + len(p)]))
    return R
if __name__ == "__main__":
    r = rows(); print(len(r))
    for x in r[::997]: print(x["cond"], "|", x["text"])
