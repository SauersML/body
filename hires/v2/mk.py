s = open("stim_v1.py").read()
import re
a = s.index("    FW = ["); b = s.index("\n", s.index("]", a)) + 1
s = s[:a] + '''    FW = [("just below the elbow", .04), ("two finger-widths below the elbow", .15), ("a quarter of the way from the elbow to the wrist", .25),
          ("a hand-width below the elbow", .33), ("halfway between the elbow and the wrist", .5), ("a hand-width above the wrist", .67),
          ("three finger-widths above the wrist", .77), ("two finger-widths above the wrist", .85), ("one finger-width above the wrist", .92), ("just above the wrist", .97)]
''' + s[b:]
a = s.index("    SW = ["); b = s.index("\n", s.index("]", a)) + 1
s = s[:a] + '''    SW = [("just below the knee", .03), ("two finger-widths below the knee", .1), ("a hand-width below the knee", .2), ("a third of the way from the knee to the ankle", .33),
          ("halfway between the knee and the ankle", .5), ("a little below halfway between the knee and the ankle", .6), ("two thirds of the way from the knee to the ankle", .67),
          ("a hand-width above the ankle", .8), ("two finger-widths above the ankle", .9), ("just above the ankle", .97)]
''' + s[b:]
a = s.index('        add(rh, f"the tip of {{p}} {S} {F} finger"'); b = s.index("\n", a) + 1
s = s[:a] + s[a:b] + '''        add(rh, f"the back of {{p}} {S} {F} finger, between the base knuckle and the middle knuckle", (s*(b+.22*L), y, -1.7), (0, 0, -1), s)
        add(rh, f"the back of {{p}} {S} {F} finger, between the middle knuckle and the knuckle nearest the nail", (s*(b+.6*L), y, -1.4), (0, 0, -1), s)
        add(rh, f"the pad of {{p}} {S} {F} fingertip", (s*(b+.9*L), y, 1.4), (0, 0, 1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, at the middle crease", (s*(b+.45*L), y, 1.5), (0, 0, 1), s)
        add(rh, f"the palm side of {{p}} {S} {F} finger, between the middle crease and the top crease", (s*(b+.62*L), y, 1.4), (0, 0, 1), s)
''' + s[b:]
a = s.index('    add(rh, f"the tip of {{p}} {S} thumb"'); b = s.index("\n", a) + 1
s = s[:a] + s[a:b] + '''    add(rh, f"the pad of {{p}} {S} thumb", (s*86.5, 156, 1.5), (0, 0, 1), s)
    add(rh, f"the palm side of {{p}} {S} thumb, at the crease of the joint", (s*85, 153, 1.5), (0, 0, 1), s)
    add(rh, f"the back of {{p}} {S} thumb, just below the nail", (s*85.5, 154.5, -1.3), (0, 0, -1), s)
''' + s[b:]
a = s.index('add("neck", "the front of {p} throat"')
s = s[:a] + '''for ph, xyz, n in [("the groove between {p} nose and upper lip", (0, 159.5, 11), (0, 0, 1)), ("the dip between {p} lower lip and chin", (0, 154.5, 9.5), (0, 0, 1)),
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
''' + s[a:]
a = s.index("TEMPLATES = ["); s = s[:a]
s += open("tail.py").read()
open("stim.py", "w").write(s)
