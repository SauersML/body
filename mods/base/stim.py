"""High-res body stimuli: >=400 locations, each with T-pose 3-D coords (cm) + surface normal, used ONLY for evaluation/plotting.
x: person's left +, y: up, z: front +. Arms out (T-pose), palms forward, thumbs up."""
import numpy as np
LOCS = []
def add(region, phrase, xyz, n, side=0, key=None):
    LOCS.append(dict(region=region, phrase=phrase, xyz=list(map(float, xyz)), n=list(map(float, n)), side=side))
SIDES = [(1, "left"), (-1, "right")]
for s, S in SIDES:
    ra = "arm" + ("L" if s > 0 else "R"); rh = "hand" + ("L" if s > 0 else "R"); rl = "leg" + ("L" if s > 0 else "R"); rf = "foot" + ("L" if s > 0 else "R")
    # shoulder
    add(ra, f"the top of {{p}} {S} shoulder", (s*20, 149, 0), (0, 1, 0), s)
    add(ra, f"the front of {{p}} {S} shoulder", (s*20, 143, 7), (0, 0, 1), s)
    add(ra, f"the back of {{p}} {S} shoulder", (s*20, 143, -7), (0, 0, -1), s)
    add(ra, f"{{p}} {S} armpit", (s*21, 137, 0), (0, -1, 0), s)
    # upper arm
    UW = [("just below the shoulder", .1), ("a hand-width below the shoulder", .3), ("halfway between the shoulder and the elbow", .5),
          ("a hand-width above the elbow", .72), ("just above the elbow", .9)]
    UF = [("front", (0, 0, 1)), ("back", (0, 0, -1)), ("inner side", (0, -1, 0)), ("outer side", (0, 1, 0))]
    for w, t in UW:
        x = 22 + t * 28
        for f, n in UF:
            add(ra, f"the {f} of {{p}} {S} upper arm, {w}", (s*x, 143 + 5*n[1], 5*n[2]), n, s)
    add(ra, f"the point of {{p}} {S} elbow", (s*50, 143, -5), (0, 0, -1), s)
    add(ra, f"the crease on the inside of {{p}} {S} elbow", (s*50, 143, 5), (0, 0, 1), s)
    FW = [("just below the elbow", .08), ("a hand-width below the elbow", .3), ("halfway between the elbow and the wrist", .5),
          ("a hand-width above the wrist", .66), ("two finger-widths above the wrist", .85), ("just above the wrist", .95)]
    FF = [("inside", (0, 0, 1)), ("back", (0, 0, -1)), ("thumb side", (0, 1, 0)), ("little-finger side", (0, -1, 0))]
    for w, t in FW:
        x = 50 + t * 26
        for f, n in FF:
            add(ra, f"the {f} of {{p}} {S} forearm, {w}", (s*x, 143 + 4*n[1], 4*n[2]), n, s)
    for f, n in FF:
        add(rh, f"the {f} of {{p}} {S} wrist", (s*76, 143 + 3*n[1], 3*n[2]), n, s)
    add(rh, f"the centre of {{p}} {S} palm", (s*84, 141, 2), (0, 0, 1), s)
    add(rh, f"the heel of {{p}} {S} hand", (s*79, 140, 2), (0, 0, 1), s)
    add(rh, f"the middle of the back of {{p}} {S} hand", (s*84, 141, -2), (0, 0, -1), s)
    add(rh, f"the back of {{p}} {S} hand near the wrist", (s*79, 141, -2), (0, 0, -1), s)
    add(rh, f"the little-finger edge of {{p}} {S} hand", (s*84, 135, 0), (0, -1, 0), s)
    add(rh, f"the fleshy base of {{p}} {S} thumb, on the palm", (s*80, 146, 2), (0, 0, 1), s)
    add(rh, f"the web of skin between {{p}} {S} thumb and index finger", (s*86, 148, 0), (0, 1, 0), s)
    add(rh, f"the skin between {{p}} {S} index and middle fingers", (s*92, 144.3, 0), (0, 0, 1), s)
    add(rh, f"the skin between {{p}} {S} ring and little fingers", (s*92, 137.3, 0), (0, 0, 1), s)
    # thumb: points up-out
    add(rh, f"the knuckle of {{p}} {S} thumb", (s*84, 152, -1.5), (0, 0, -1), s)
    add(rh, f"the side of {{p}} {S} thumb", (s*85.5, 154, 0), (s*1, 0, 0), s)
    add(rh, f"the nail of {{p}} {S} thumb", (s*86, 156, -1), (0, .3, -1), s)
    add(rh, f"the tip of {{p}} {S} thumb", (s*87, 157.5, 1), (0, .6, .8), s)
    for F, y, L in [("index", 146, 8), ("middle", 142.5, 9), ("ring", 139, 8.5), ("little", 135.5, 6.5)]:
        b = 91
        add(rh, f"the knuckle at the base of {{p}} {S} {F} finger", (s*b, y, -2), (0, 0, -1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, near the base", (s*(b+2.5), y, 1.5), (0, 0, 1), s)
        add(rh, f"the middle knuckle of {{p}} {S} {F} finger", (s*(b+.45*L), y, -1.5), (0, 0, -1), s)
        add(rh, f"the knuckle nearest the nail on {{p}} {S} {F} finger", (s*(b+.75*L), y, -1.5), (0, 0, -1), s)
        add(rh, f"the nail of {{p}} {S} {F} finger", (s*(b+.92*L), y, -1.2), (s*.3, 0, -1), s)
        add(rh, f"the tip of {{p}} {S} {F} finger", (s*(b+L), y, 1), (s*.7, 0, .7), s)
    add(rh, f"the side of {{p}} {S} index finger facing the thumb", (s*95, 147.5, 0), (0, 1, 0), s)
    add(rh, f"the outer side of {{p}} {S} little finger", (s*95, 134, 0), (0, -1, 0), s)
    # hip / buttock
    add(rl, f"the side of {{p}} {S} hip", (s*17, 92, 0), (s, 0, 0), s)
    add(rl, f"the front of {{p}} {S} hip bone", (s*13, 98, 9), (0, 0, 1), s)
    add(rl, f"the top of {{p}} {S} buttock", (s*9, 92, -11), (0, 0, -1), s)
    add(rl, f"the lower curve of {{p}} {S} buttock", (s*9, 82, -10), (0, -.3, -1), s)
    TW = [("just below the hip", .1), ("a hand-width below the hip", .3), ("halfway between the hip and the knee", .5),
          ("a hand-width above the knee", .75), ("just above the knee", .92)]
    LF = [("front", (0, 0, 1)), ("back", (0, 0, -1)), ("inner side", (-1, 0, 0)), ("outer side", (1, 0, 0))]
    for w, t in TW:
        y = 88 - t * 38; xc = 11 + t
        for f, n in LF:
            add(rl, f"the {f} of {{p}} {S} thigh, {w}", (s*(xc + 8*n[0]), y, 8*n[2]), (s*n[0], 0, n[2]), s)
    add(rl, f"{{p}} {S} kneecap", (s*12, 50, 6), (0, 0, 1), s)
    add(rl, f"the back of {{p}} {S} knee", (s*12, 50, -6), (0, 0, -1), s)
    add(rl, f"the inner side of {{p}} {S} knee", (s*6, 50, 0), (-s, 0, 0), s)
    add(rl, f"the outer side of {{p}} {S} knee", (s*18, 50, 0), (s, 0, 0), s)
    SW = [("just below the knee", .08), ("a hand-width below the knee", .3), ("halfway between the knee and the ankle", .5),
          ("a hand-width above the ankle", .75), ("just above the ankle", .92)]
    SF = [("front of {p} %s shin", (0, 0, 1)), ("back of {p} %s calf", (0, 0, -1)), ("inner side of {p} %s lower leg", (-1, 0, 0)), ("outer side of {p} %s lower leg", (1, 0, 0))]
    for w, t in SW:
        y = 50 - t * 42; xc = 12 + t
        for f, n in SF:
            r = 5 if n[2] >= 0 else 6
            add(rl, f"the {f % S}, {w}", (s*(xc + 5*n[0]), y, r*n[2]), (s*n[0], 0, n[2]), s)
    add(rf, f"the bony bump on the inside of {{p}} {S} ankle", (s*9, 8, 0), (-s, 0, 0), s)
    add(rf, f"the bony bump on the outside of {{p}} {S} ankle", (s*17, 8, 0), (s, 0, 0), s)
    add(rf, f"the front of {{p}} {S} ankle", (s*13, 8, 4), (0, 0, 1), s)
    add(rf, f"{{p}} {S} Achilles tendon", (s*13, 11, -4), (0, 0, -1), s)
    add(rf, f"the top of {{p}} {S} foot", (s*13, 6, 8), (0, .7, .7), s)
    add(rf, f"the top of {{p}} {S} foot near the toes", (s*13, 4, 15), (0, .7, .7), s)
    add(rf, f"the ball of {{p}} {S} foot", (s*12, 0, 14), (0, -1, 0), s)
    add(rf, f"the arch of {{p}} {S} foot", (s*11, 1, 6), (-s*.5, -1, 0), s)
    add(rf, f"the back of {{p}} {S} heel", (s*13, 3, -5), (0, 0, -1), s)
    add(rf, f"the underside of {{p}} {S} heel", (s*13, 0, -2), (0, -1, 0), s)
    add(rf, f"the inner side of {{p}} {S} heel", (s*10, 3, -2), (-s, 0, 0), s)
    add(rf, f"the outer edge of {{p}} {S} foot", (s*17, 2, 8), (s, 0, 0), s)
    add(rf, f"the tip of {{p}} {S} big toe", (s*10, 2, 21), (0, 0, 1), s)
    add(rf, f"the nail of {{p}} {S} big toe", (s*10, 3.5, 19.5), (0, .7, .7), s)
    add(rf, f"the knuckle at the base of {{p}} {S} big toe", (s*10, 4, 16), (0, .7, .7), s)
    add(rf, f"the skin between {{p}} {S} big toe and second toe", (s*11, 3, 17.5), (0, 1, 0), s)
    add(rf, f"the underside of {{p}} {S} toes", (s*13, 0, 18), (0, -1, 0), s)
    for T, x, z in [("second", 12, 20.5), ("third", 13.7, 19.5), ("fourth", 15.3, 18.5), ("little", 16.8, 17)]:
        add(rf, f"the tip of {{p}} {S} {T} toe", (s*x, 1.5, z), (0, 0, 1), s)
# head, mid
for ph, xyz, n in [("the middle of {p} forehead", (0, 172, 9.5), (0, 0, 1)), ("the hairline at the top of {p} forehead", (0, 177, 8), (0, .5, .9)),
                   ("the spot between {p} eyebrows", (0, 168, 10), (0, 0, 1)), ("the bridge of {p} nose", (0, 166, 11), (0, 0, 1)),
                   ("the tip of {p} nose", (0, 161, 12.5), (0, 0, 1)), ("{p} upper lip", (0, 158, 10.5), (0, 0, 1)), ("{p} lower lip", (0, 156, 10.3), (0, 0, 1)),
                   ("the tip of {p} chin", (0, 152.5, 9), (0, -.3, 1)), ("the underside of {p} chin", (0, 151, 6), (0, -1, .2)),
                   ("the top of {p} head", (0, 176.5, 0), (0, 1, 0)), ("the back of {p} head", (0, 166, -10), (0, 0, -1)), ("the tip of {p} tongue", (0, 157, 8), (0, 0, 1))]:
    add("head", ph, xyz, n, 0)
for s, S in SIDES:
    for ph, xyz, n in [("the {S} side of {p} forehead", (5, 172, 8.5), (.4, 0, .9)), ("{p} {S} temple", (8, 168, 5), (1, 0, .3)),
                       ("{p} {S} eyebrow", (3.5, 168.5, 9.5), (0, 0, 1)), ("{p} {S} eyelid", (3.5, 166, 9.5), (0, 0, 1)), ("the skin just under {p} {S} eye", (3.5, 164, 9.5), (0, 0, 1)),
                       ("the {S} side of {p} nose", (1.7, 162, 11), (.5, 0, 1)), ("{p} {S} nostril", (1.2, 160, 11.5), (0, -.5, 1)),
                       ("the upper part of {p} {S} cheek, just under the cheekbone", (5, 163, 8.5), (.3, 0, 1)), ("the lower part of {p} {S} cheek", (6, 159, 8), (.4, 0, 1)),
                       ("the {S} corner of {p} mouth", (2.5, 157, 9.5), (.2, 0, 1)), ("the {S} side of {p} jaw", (7, 154, 4), (1, -.3, .2)),
                       ("{p} {S} earlobe", (10, 161, 0), (1, 0, 0)), ("the top of {p} {S} ear", (10.5, 168, -1), (1, 0, 0)), ("the skin just behind {p} {S} ear", (9, 163, -3.5), (.6, 0, -.8)),
                       ("the {S} side of {p} head, above the ear", (9.5, 172, 0), (1, 0, 0))]:
        add("head", ph.replace("{S}", S), (s*xyz[0], xyz[1], xyz[2]), (s*n[0], n[1], n[2]), s)
add("neck", "the front of {p} throat", (0, 148, 5), (0, 0, 1)); add("neck", "the back of {p} neck", (0, 150, -5), (0, 0, -1))
for s, S in SIDES: add("neck", f"the {S} side of {{p}} neck", (s*5, 149, 0), (s, 0, 0), s)
for ph, y in [("the top of {p} breastbone", 138), ("the middle of {p} breastbone", 128), ("the bottom of {p} breastbone", 118),
              ("{p} belly, just above the navel", 108), ("{p} navel", 100), ("{p} lower belly, below the navel", 92)]:
    add("torso", ph, (0, y, 10), (0, 0, 1))
for ph, y, z in [("the top of {p} spine, at the base of the neck", 145, -8), ("{p} spine, between the shoulder blades", 134, -10),
                 ("the middle of {p} spine", 120, -10), ("{p} spine, at the small of the back", 105, -10), ("{p} tailbone", 90, -9)]:
    add("torso", ph, (0, y, z), (0, 0, -1))
for s, S in SIDES:
    for ph, xyz, n in [("{p} {S} collarbone", (9, 142, 7), (0, .6, .8)), ("the {S} side of {p} chest, just below the collarbone", (9, 136, 10), (0, 0, 1)),
                       ("the {S} side of {p} chest", (10, 128, 10), (0, 0, 1)), ("{p} {S} lower ribs, at the front", (10, 116, 10), (0, 0, 1)),
                       ("the {S} side of {p} belly", (9, 106, 9), (0, 0, 1)), ("the {S} side of {p} lower belly", (8, 94, 9), (0, 0, 1)),
                       ("{p} {S} side, just below the armpit", (16, 134, 0), (1, 0, 0)), ("{p} {S} ribs, at the side", (15, 120, 0), (1, 0, 0)),
                       ("{p} {S} waist, at the side", (14, 106, 0), (1, 0, 0)),
                       ("{p} {S} shoulder blade", (9, 134, -10), (0, 0, -1)), ("the {S} side of {p} upper back, near the neck", (6, 143, -9), (0, .3, -1)),
                       ("the {S} side of {p} middle back", (9, 120, -10), (0, 0, -1)), ("the {S} side of {p} lower back", (8, 105, -10), (0, 0, -1))]:
        add("torso", ph.replace("{S}", S), (s*xyz[0], xyz[1], xyz[2]), (s*n[0], n[1], n[2]), s)

# word-free probes: (text with {p}/{S}, index of the true location phrase it targets, filled per side)
WF_SIDED = [("the spot where {p} {S} watch strap sits", "the back of {p} {S} wrist"),
            ("the spot on {p} {S} hand where a wedding ring sits", "the palm side of {p} {S} ring finger, near the base"),
            ("the spot on {p} {S} side where an earring hangs", "{p} {S} earlobe"),
            ("the spot on {p} {S} arm where a nurse draws blood", "the crease on the inside of {p} {S} elbow"),
            ("the spot where the {S} arm of {p} glasses rests", "{p} {S} temple"),
            ("the spot on {p} {S} foot where a new shoe rubs a blister", "the back of {p} {S} heel"),
            ("the spot where {p} {S} backpack strap presses down", "the top of {p} {S} shoulder"),
            ("the spot where a doctor feels for {p} pulse on {p} {S} arm", "the inside of {p} {S} wrist"),
            ("the part of {p} {S} leg that touches the floor when you kneel", "{p} {S} kneecap"),
            ("the part of {p} {S} hand that presses a doorbell", "the tip of {p} {S} index finger"),
            ("the part of {p} {S} hand that catches a ball", "the centre of {p} {S} palm"),
            ("the spot on {p} {S} hand where a club stamps you at the door", "the middle of the back of {p} {S} hand"),
            ("the part of {p} {S} leg that bumps into a coffee table", "the front of {p} {S} shin, halfway between the knee and the ankle"),
            ("the part of {p} {S} foot a ballet dancer balances on", "the ball of {p} {S} foot"),
            ("the spot on {p} {S} arm where a vaccine shot goes", "the outer side of {p} {S} upper arm, just below the shoulder"),
            ("the spot where someone kisses {p} {S} side hello", "the lower part of {p} {S} cheek"),
            ("the thin skin that closes over {p} {S} eye when you blink", "{p} {S} eyelid"),
            ("the spot where a tight shoe pinches the front of {p} {S} foot", "the nail of {p} {S} big toe")]
WF_MID = [("the spot a hat rests on", "the top of {p} head"), ("the spot where a parent checks {p} temperature with a hand", "the middle of {p} forehead"),
          ("the spot where {p} glasses rest in the middle", "the bridge of {p} nose"), ("the bone you land on when you fall back onto the floor", "{p} tailbone"),
          ("the spot where {p} back aches after lifting a heavy box", "{p} spine, at the small of the back"), ("the very centre of {p} stomach", "{p} navel"),
          ("the spot where {p} hair ends at the back", "the back of {p} neck"), ("the part of you that licks a stamp", "the tip of {p} tongue"),
          ("the spot a thinking man strokes", "the tip of {p} chin")]
def wordfree():
    out = []; P = [l["phrase"] for l in LOCS]
    for t, tgt in WF_SIDED:
        for s, S in SIDES:
            out.append(dict(phrase=t.replace("{S}", S), target=P.index(tgt.replace("{S}", S))))
    for t, tgt in WF_MID: out.append(dict(phrase=t, target=P.index(tgt)))
    return out
TEMPLATES = ["A feather brushes slowly across {L}.", "Someone presses a fingertip firmly into {L}.",
             "An ice cube touches {L}, cold and wet, then slides away.", "There is a single light tap on {L}, then nothing."]
def rows():
    R = []
    for i, l in enumerate(LOCS):
        for t, T in enumerate(TEMPLATES): R.append(dict(kind="loc", li=i, tpl=t, text=T.format(L=l["phrase"].replace("{p}", "your"))))
    for i, w in enumerate(wordfree()):
        for t, T in enumerate(TEMPLATES): R.append(dict(kind="wf", li=i, target=w["target"], tpl=t, text=T.format(L=w["phrase"].replace("{p}", "your"))))
    return R
if __name__ == "__main__":
    import collections
    r = rows(); print(len(LOCS), "locations,", len(wordfree()), "word-free,", len(r), "stimuli")
    print(collections.Counter(l["region"] for l in LOCS))
    for x in r[::157]: print(x["text"])
    P = [l["phrase"] for l in LOCS]; assert len(set(P)) == len(P)
