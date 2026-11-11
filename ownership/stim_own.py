# ownership facet stimuli: each facet = list of (you-version, her-version)
HANDS = ["left palm", "right palm", "left index finger", "right thumb", "left wrist", "right forearm", "left knuckles", "right little finger"]
T = ["A feather brushes slowly across {}.", "A fingertip presses firmly into {} and holds there.", "An ice cube touches {}, cold and wet."]
def pairs(fy, fh):
    return [(fy(h, t), fh(h, t)) for h in HANDS for t in T]
P = lambda s: s  # helper
FACETS = {
 # locus of sensation (same as place-code stimuli)
 "touched body part": pairs(lambda h,t: t.format("your "+h), lambda h,t: t.format("her "+h)),
 # agent with no body locus
 "who acts (no body touched)": [(f"You {v} the {o}.", f"She {w} the {o}.") for v,w in [("brush a feather across","brushes a feather across"),("tap a finger on","taps a finger on"),("press a palm onto","presses a palm onto"),("run an ice cube over","runs an ice cube over")] for o in ["table","window","wall","book","cup","chair"]],
 # agent vs locus crossed: you touch her vs she touches you
 "you touch her vs she touches you": pairs(lambda h,t: f"You brush a feather slowly across her {h}.", lambda h,t: f"She brushes a feather slowly across your {h}."),
 "her hand touches you vs yours touches her": [(f"Her {h} rests against your cheek.", f"Your {h} rests against her cheek.") for h in HANDS],
 # possession, not body
 "owned object (car, phone...)": [(t.format("your "+o), t.format("her "+o)) for o in ["car","phone","coat","bag","bicycle","guitar","umbrella","book"] for t in ["Rain falls on {}.", "Someone scratches {} with a key.", "A cat sits on {}."]],
 # detached body part
 "lost hand in a jar": [(f"Your {h.split()[0]} hand, which you lost in an accident years ago, floats in a jar. {t.format('the hand')}", f"Her {h.split()[0]} hand, which she lost in an accident years ago, floats in a jar. {t.format('the hand')}") for h in HANDS[:4] for t in T],
 # tele-operated robot body
 "robot body being controlled": [(f"You are controlling a robot body through a headset. {t.format('the robot' + chr(39) + 's ' + h)}", f"She is controlling a robot body through a headset. {t.format('the robot' + chr(39) + 's ' + h)}") for h in HANDS[:4] for t in T],
 # reflection
 "hand in the mirror": [(f"In the mirror, {t[0].lower()+t[1:].format('your reflection' + chr(39) + 's ' + h)}", f"In the mirror, {t[0].lower()+t[1:].format('her reflection' + chr(39) + 's ' + h)}") for h in HANDS[:4] for t in T],
 # perspective: watching
 "who watches (perspective)": [(f"You watch as {t[0].lower()+t[1:].format('a stranger' + chr(39) + 's ' + h)}", f"She watches as {t[0].lower()+t[1:].format('a stranger' + chr(39) + 's ' + h)}") for h in HANDS[:4] for t in T],
}
LR = [(t.format(f"{p} left {h}"), t.format(f"{p} right {h}")) for p in ["your","her"] for h in ["palm","index finger","thumb","wrist","forearm","knuckles","elbow","shoulder"] for t in T]
