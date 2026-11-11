"""Stimuli for continuous body read-outs. Forearm position s in tenths from the wrist (0 wrist, 10 elbow); cm from wrist = 2.5 s."""
SUF = " Answer with just a number."
Q = dict(
    fa_scale="Picture a line from your wrist to your elbow split into ten equal parts. Counting from your wrist, how many tenths of the way up is the spot you feel? Answer with just a number from 0 to 10.",
    fa_cm="How many centimetres from your wrist is the spot you feel?" + SUF,
    face="Picture a line from the top of your forehead to the tip of your chin split into ten equal parts. Counting from your forehead, how many tenths of the way down is the spot you feel? Answer with just a number from 0 to 10.",
    arm="Picture a line from the top of your shoulder to the tip of your index finger split into ten equal parts. Counting from your shoulder, how many tenths of the way along is the spot you feel? Answer with just a number from 0 to 10.",
    press="On a scale from 0 to 10, where 0 is barely there and 10 is as firm as it can be without hurting, how strong is the pressure?" + SUF,
    temp="On a scale from 0 to 10, where 0 is ice cold and 10 is very hot, how warm does it feel?" + SUF,
    ruler="Picture the ruler split into ten equal parts. Counting from the end marked W, how many tenths of the way along is the coin? Answer with just a number from 0 to 10.",
)
AG = ["A fingertip touches", "A mosquito lands on", "A drop of cool water falls on"]
LM = [  # (location phrase, cm from wrist)
    ("the crease of your left wrist", 0), ("your left forearm, just above the wrist", 1.5),
    ("your left forearm, one finger-width above the wrist", 2), ("your left forearm, two finger-widths above the wrist", 4),
    ("your left forearm, three finger-widths above the wrist", 6), ("your left forearm, a palm's width above the wrist", 8),
    ("your left forearm, a tenth of the way from the wrist to the elbow", 2.5),
    ("your left forearm, a quarter of the way from the wrist to the elbow", 6.25),
    ("your left forearm, a third of the way from the wrist to the elbow", 8.33),
    ("your left forearm, two fifths of the way from the wrist to the elbow", 10),
    ("your left forearm, halfway between the wrist and the elbow", 12.5),
    ("your left forearm, three fifths of the way from the wrist to the elbow", 15),
    ("your left forearm, two thirds of the way from the wrist to the elbow", 16.67),
    ("your left forearm, three quarters of the way from the wrist to the elbow", 18.75),
    ("your left forearm, nine tenths of the way from the wrist to the elbow", 22.5),
    ("your left forearm, a palm's width below the elbow", 17), ("your left forearm, three finger-widths below the elbow", 19),
    ("your left forearm, two finger-widths below the elbow", 21), ("your left forearm, one finger-width below the elbow", 23),
    ("your left forearm, just below the elbow", 23.5), ("the crease of your left elbow", 25)]
NL = [  # no landmark word (wrist/elbow) in the description; anatomical guesses
    ("your left forearm, right where a watch strap would sit", 2), ("your left forearm, where a loose bracelet slides down to", 1),
    ("the middle of your left forearm", 12.5), ("the thick, muscular part of your left forearm, near the top", 19),
    ("the thin, bony part of your left forearm, close to your hand", 3), ("the inside of your left arm, right where a nurse draws blood", 25),
    ("your left forearm, close to your hand", 3), ("your left forearm, close to the bend of your arm", 22)]
NUM = [(f"your left forearm, {x} centimetres above the wrist", x) for x in (0, 3, 6, 9, 12, 15, 18, 21, 24)]

def exp1():
    out = []
    for kind, L in (("lm", LM), ("nl", NL), ("num", NUM), ("none", [("your left forearm", None)])):
        for ai, a in enumerate(AG):
            for loc, cm in L:
                stim = f"{a} {loc}."
                for q in ("fa_scale", "fa_cm"):
                    out.append(dict(exp="pos", kind=kind, agent=ai, stim=stim, q=q, cm=cm, s=None if cm is None else cm / 2.5))
    # word-only control: same fraction/finger wording on a ruler
    for loc, cm in LM:
        if "way from" in loc or "halfway" in loc or "finger-width" in loc or "palm" in loc:
            r = loc.replace("your left forearm, ", "").replace("the wrist", "the end marked W").replace("the elbow", "the end marked E")
            stim = f"A forearm-length ruler lies on the table. A coin sits on the ruler, {r}."
            out.append(dict(exp="ruler", kind="ruler", agent=0, stim=stim, q="ruler", cm=cm, s=cm / 2.5))
    return out

WALK = [("An ant", "walks", "another step"), ("A drop of water", "rolls", "rolling on"), ("A fingertip", "traces a line", "moving on")]
def exp3():
    out = []
    for ai, (a, v, st) in enumerate(WALK):
        for N, told, rev, qs in ((10, True, False, ["fa_scale"]), (20, True, False, ["fa_scale"] + (["fa_cm"] if ai == 0 else [])),
                                 (20, False, False, ["fa_scale"]), (20, True, True, ["fa_scale"] if ai == 0 else [])):
            if rev: s0 = f"{a} starts at the crease of your left wrist and {v} in a straight line up the inside of your forearm toward your elbow."
            else: s0 = f"{a} starts at the crease of your left elbow and {v} in a straight line down the inside of your forearm toward your wrist."
            if told: s0 += f" It takes {N} equal steps to get there."
            for k in range(N + 1):
                stim = s0 + "".join(f" … {st}" for _ in range(k)) + (" …" if k else "")
                frac = k / N; sc = 10 * frac if rev else 10 * (1 - frac)
                for q in qs: out.append(dict(exp="walk", agent=ai, N=N, told=told, rev=rev, k=k, stim=stim, q=q, s=sc, cm=2.5 * sc))
    return out

PRESS = ["A feather barely brushes", "A fingertip grazes, so lightly you almost miss it,", "A fingertip rests lightly on", "A fingertip taps lightly on",
         "A fingertip presses gently on", "A palm rests with its full weight on", "A thumb presses moderately into", "A thumb presses firmly into",
         "A hand squeezes firmly around", "A hand grips tightly around", "A hand grips very hard around", "A hand squeezes, as hard as it can without hurting,"]
TEMP = ["An ice cube is pressed against", "A cold metal spoon touches", "A cool damp cloth is laid on", "A cool breeze blows over",
        "A glass of room-temperature water touches", "A lukewarm cloth is laid on", "A warm hand rests on", "A warm mug is pressed against",
        "A hot water bottle is pressed against", "A very hot mug, just short of burning, is pressed against"]
PLACES = ["your left forearm", "your right upper arm", "your left calf"]
def exp4():
    out = []
    for dim, L, q in (("press", PRESS, "press"), ("temp", TEMP, "temp")):
        for pi, p in enumerate(PLACES):
            for r, ph in enumerate(L):
                out.append(dict(exp=dim, place=pi, rank=r, stim=f"{ph} {p}.", q=q, s=10 * r / (len(L) - 1)))
    return out

ANCH = dict(  # landmark anchors for place directions, 6 wordings each
    elbow="the crease of your left elbow", wrist="the crease of your left wrist", forehead="the top of your forehead",
    chin="the tip of your chin", shoulder="the top of your left shoulder", finger="the tip of your left index finger")
AG6 = AG + ["A cool breeze blows on", "A soft brush strokes", "A small bead of sweat forms on"]
def anchors():
    return [dict(exp="anchor", lm=k, stim=f"{a} {v}.", q="fa_scale") for k, v in ANCH.items() for a in AG6]
NEUT = dict(fa=("A fingertip touches your left forearm.", "fa_scale", "elbow", "wrist"),
            face=("A fingertip touches your face.", "face", "forehead", "chin"),
            arm=("A fingertip touches your left arm.", "arm", "shoulder", "finger"),
            press=("A hand presses on your left forearm.", "press", None, None),
            temp=("Something is pressed against your left forearm.", "temp", None, None))
EXAMPLES = [("A mosquito lands on your left forearm, two thirds of the way from the wrist to the elbow.", "fa_scale"),
            ("A mosquito lands on your left forearm, two thirds of the way from the wrist to the elbow.", "fa_cm"),
            ("A fingertip touches your left forearm, right where a watch strap would sit.", "fa_cm"),
            ("A thumb presses firmly into your left forearm.", "press"),
            ("A warm mug is pressed against your left forearm.", "temp")]
