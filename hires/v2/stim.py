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
    FW = [("just below the elbow", .04), ("two finger-widths below the elbow", .15), ("a quarter of the way from the elbow to the wrist", .25),
          ("a hand-width below the elbow", .33), ("halfway between the elbow and the wrist", .5), ("a hand-width above the wrist", .67),
          ("three finger-widths above the wrist", .77), ("two finger-widths above the wrist", .85), ("one finger-width above the wrist", .92), ("just above the wrist", .97)]
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
    add(rh, f"the pad of {{p}} {S} thumb", (s*86.5, 156, 1.5), (0, 0, 1), s)
    add(rh, f"the palm side of {{p}} {S} thumb, at the crease of the joint", (s*85, 153, 1.5), (0, 0, 1), s)
    add(rh, f"the back of {{p}} {S} thumb, just below the nail", (s*85.5, 154.5, -1.3), (0, 0, -1), s)
    for F, y, L in [("index", 146, 8), ("middle", 142.5, 9), ("ring", 139, 8.5), ("little", 135.5, 6.5)]:
        b = 91
        add(rh, f"the knuckle at the base of {{p}} {S} {F} finger", (s*b, y, -2), (0, 0, -1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, near the base", (s*(b+2.5), y, 1.5), (0, 0, 1), s)
        add(rh, f"the middle knuckle of {{p}} {S} {F} finger", (s*(b+.45*L), y, -1.5), (0, 0, -1), s)
        add(rh, f"the knuckle nearest the nail on {{p}} {S} {F} finger", (s*(b+.75*L), y, -1.5), (0, 0, -1), s)
        add(rh, f"the nail of {{p}} {S} {F} finger", (s*(b+.92*L), y, -1.2), (s*.3, 0, -1), s)
        add(rh, f"the tip of {{p}} {S} {F} finger", (s*(b+L), y, 1), (s*.7, 0, .7), s)
        add(rh, f"the back of {{p}} {S} {F} finger, between the base knuckle and the middle knuckle", (s*(b+.22*L), y, -1.7), (0, 0, -1), s)
        add(rh, f"the back of {{p}} {S} {F} finger, between the middle knuckle and the knuckle nearest the nail", (s*(b+.6*L), y, -1.4), (0, 0, -1), s)
        add(rh, f"the pad of {{p}} {S} {F} fingertip", (s*(b+.9*L), y, 1.4), (0, 0, 1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, at the middle crease", (s*(b+.45*L), y, 1.5), (0, 0, 1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, between the middle crease and the top crease", (s*(b+.62*L), y, 1.4), (0, 0, 1), s)
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
    SW = [("just below the knee", .03), ("two finger-widths below the knee", .1), ("a hand-width below the knee", .2), ("a third of the way from the knee to the ankle", .33),
          ("halfway between the knee and the ankle", .5), ("a little below halfway between the knee and the ankle", .6), ("two thirds of the way from the knee to the ankle", .67),
          ("a hand-width above the ankle", .8), ("two finger-widths above the ankle", .9), ("just above the ankle", .97)]
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
for ph, xyz, n in [("the groove between {p} nose and upper lip", (0, 159.5, 11), (0, 0, 1)), ("the dip between {p} lower lip and chin", (0, 154.5, 9.5), (0, 0, 1)),
                   ("the centre of {p} forehead, just above the eyebrows", (0, 170, 10), (0, 0, 1))]:
    add("head", ph, xyz, n, 0)
for s, S in SIDES:
    for ph, xyz, n in [("the outer corner of {p} {S} eye", (5.5, 166, 8.5), (.5, 0, .9)), ("the inner corner of {p} {S} eye", (1.5, 165.5, 10), (0, 0, 1)),
                       ("the outer end of {p} {S} eyebrow", (6, 168.5, 8.5), (.5, 0, .9)), ("{p} {S} cheekbone", (6, 164, 8), (.5, 0, .9)),
                       ("the hinge of {p} {S} jaw, just in front of the ear", (8.5, 159, 1.5), (1, 0, .2)), ("the angle of {p} {S} jaw, below the ear", (8, 153.5, -1), (1, -.3, 0)),
                       ("the {S} side of {p} chin", (3, 153, 8.5), (.3, 0, 1)), ("the {S} side of {p} upper lip", (2, 158.2, 10.2), (.2, 0, 1)),
                       ("the {S} side of {p} lower lip", (2, 156, 10), (.2, 0, 1)), ("the small flap in front of {p} {S} ear hole", (9, 162, 1), (1, 0, .3)),
                       ("the back of {p} {S} ear", (10, 164, -2), (.5, 0, -1))]:
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
# alternate landmark descriptions of shaft spots (no word for the limb segment): extra word-free probes
FWD = {"inside": ("the crease of {p} {S} elbow", "the inside of {p} {S} wrist", "on the palm side"), "back": ("the point of {p} {S} elbow", "the back of {p} {S} wrist", "on the same side as the back of the hand"),
       "thumb side": ("{p} {S} elbow", "{p} {S} wrist", "in line with the thumb"), "little-finger side": ("{p} {S} elbow", "{p} {S} wrist", "in line with the little finger")}
LWD = {"front of {p} %s shin": ("{p} {S} kneecap", "the front of {p} {S} ankle", "on the front, over the bone"), "back of {p} %s calf": ("the back of {p} {S} knee", "{p} {S} Achilles tendon", "on the back"),
       "inner side of {p} %s lower leg": ("{p} {S} knee", "{p} {S} ankle", "on the side facing the other leg"), "outer side of {p} %s lower leg": ("{p} {S} knee", "{p} {S} ankle", "on the outer side")}
def alt_probes():
    P = [l["phrase"] for l in LOCS]; out = []
    for i, ph in enumerate(P):
        for s, S in SIDES:
            if f"of {{p}} {S} forearm, " in ph:
                f = ph.split("the ")[1].split(" of {p}")[0]
                if f not in ("inside", "back"): continue
                t = (abs(LOCS[i]["xyz"][0]) - 50) / 26; nfw = t * 13
                a0, a1, w = FWD[f]
                d = f"the spot about {max(1, round(nfw))} finger-width{'s' if round(nfw) > 1 else ''} below {a0}, {w}" if t < .5 else f"the spot about {max(1, round(13 - nfw))} finger-width{'s' if round(13 - nfw) > 1 else ''} above {a1}, {w}"
                out.append(dict(phrase=d.replace("{S}", S), target=i))
            for k, (a0, a1, w) in LWD.items():
                if ph.startswith("the " + k % S + ", ") and k.split()[0] in ("front", "back"):
                    t = (50 - LOCS[i]["xyz"][1]) / 42; nfw = t * 21
                    d = f"the spot about {max(1, round(nfw))} finger-width{'s' if round(nfw) > 1 else ''} below {a0}, {w}" if t < .5 else f"the spot about {max(1, round(21 - nfw))} finger-width{'s' if round(21 - nfw) > 1 else ''} above {a1}, {w}"
                    out.append(dict(phrase=d.replace("{S}", S), target=i))
    return out
FRAMINGS = [  # (family, template)
    ("touch", "A feather brushes slowly across {L}."), ("touch", "Someone rests a fingertip very lightly on {L}."),
    ("touch", "There is a single light tap on {L}, then nothing."), ("touch", "Someone presses a thumb firmly into {L}."),
    ("touch", "Someone pokes {L} sharply with a fingertip."),
    ("pain", "A sudden sharp pain shoots through {L}."), ("pain", "A dull ache settles into {L}."),
    ("itch", "An itch starts on {L}."), ("itch", "A mosquito bites {L}."),
    ("warm", "A warm patch of sunlight falls on {L}."), ("warm", "A drop of hot wax lands on {L}."),
    ("cold", "An ice cube touches {L}, cold and wet, then slides away."), ("cold", "A cold drop of water rolls across {L}."),
    ("pressure", "A heavy weight presses down on {L}."),
    ("medical", "The doctor presses two fingers on {L} and asks if it hurts."), ("medical", "A nurse wipes {L} with a cold alcohol swab."),
    ("clothing", "A loose thread from your clothing tickles {L}."), ("clothing", "A cold metal chain slides across {L}."),
    ("injury", "You bump {L} hard against the edge of a table."), ("injury", "During the game, a ball smacks into {L}."),
    ("injury", "You scrape {L} on rough concrete as you fall."),
    ("proprio", "You tense the muscles just under {L}."), ("proprio", "Without moving anything else, you put all your attention on {L}."),
    ("proprio", "You notice exactly where {L} is right now, without looking."), ("insect", "A small beetle crawls over {L}."),
    ("insect", "A bee lands on {L} and sits still."),
]
def probes():
    return [dict(w, kind="wf") for w in wordfree()] + [dict(w, kind="alt") for w in alt_probes()]
def rows():
    R = []
    for i, l in enumerate(LOCS):
        for t, (fam, T) in enumerate(FRAMINGS): R.append(dict(kind="loc", li=i, tpl=t, phrase=l["phrase"].replace("{p}", "your")))
    for i, w in enumerate(probes()):
        for t, (fam, T) in enumerate(FRAMINGS): R.append(dict(kind=w["kind"], li=i, target=w["target"], tpl=t, phrase=w["phrase"].replace("{p}", "your")))
    for r in R:
        T = FRAMINGS[r["tpl"]][1]; a = T.index("{L}")
        if "," in r["phrase"] and T[a + 3] not in ".,": T = T.replace("{L}", "{L},")
        r["text"] = T.format(L=r["phrase"]); r["span"] = [a, a + len(r["phrase"])]
    return R
if __name__ == "__main__":
    import collections
    r = rows(); print(len(LOCS), "locations,", len(probes()), "probes", collections.Counter(p["kind"] for p in probes()), len(r), "stimuli")
    print(collections.Counter(l["region"] for l in LOCS))
    for x in r[::1777]: print(x["text"], "|", x["text"][x["span"][0]:x["span"][1]])
    for p in alt_probes()[::9]: print(p["phrase"], "->", LOCS[p["target"]]["phrase"])
    P = [l["phrase"] for l in LOCS]; assert len(set(P)) == len(P)
