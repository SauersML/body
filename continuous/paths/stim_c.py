"""Continuous-path stimuli: one motion sentence, then K identical step phrases. No landmark words between start and end."""
K = 16
AGENTS = {  # name: (subject, verb phrase present, step phrase)
 "water": ("A single drop of cool water", "rolls slowly", "still rolling"),
 "ant": ("A small ant", "walks slowly", "still walking"),
 "feather": ("A soft feather", "is drawn slowly", "still moving"),
 "finger": ("Someone's fingertip", "traces a slow line", "still tracing"),
}
# path: (start phrase, route, end phrase, reverse route)
PATHS = {}
for S in ["left", "right"]:
    PATHS[f"arm_palm_{S}"] = (f"the tip of your {S} middle finger", f"along the palm side of your {S} hand and arm", f"your {S} armpit", f"along the inner side of your {S} arm and the palm side of your hand")
    PATHS[f"arm_back_{S}"] = (f"the tip of your {S} middle finger", f"along the back of your {S} hand and arm", f"the top of your {S} shoulder", f"along the back of your {S} arm and hand")
    PATHS[f"leg_{S}"] = (f"the tip of your {S} big toe", f"up the front of your {S} foot and leg", f"your {S} hip", f"down the front of your {S} leg and foot")
PATHS["head"] = ("the top of your head", "down the middle of your forehead and face", "the tip of your chin", "up the middle of your face and forehead")
PATHS["spine"] = ("the back of your neck", "down along your spine", "your tailbone", "up along your spine")
PATHS["chest"] = ("your left shoulder", "across your chest", "your right shoulder", "across your chest")
NONBODY = {
 "table": ("one end of the long kitchen table", "along the tabletop", "the other end of the table", "along the tabletop"),
 "ruler": ("the zero mark of a wooden ruler", "along the ruler", "the far end of the ruler", "along the ruler"),
 "sill": ("the left corner of the windowsill", "along the windowsill", "the right corner of the windowsill", "along the windowsill"),
}
def sentence(agent, a, route, b):
    subj, vp, step = AGENTS[agent]
    if agent == "finger": s = f"{subj} {vp} from {a} {route} to {b}."
    else: s = f"{subj} starts at {a} and {vp.split()[0]} {vp.split(1)[1] if False else ' '.join(vp.split()[1:])} {route} toward {b}."
    return " ".join(s.split()), step
def items():
    out = []
    for kind, P in [("body", PATHS), ("nonbody", NONBODY)]:
        for pn, (a, route, b, rroute) in P.items():
            for ag in AGENTS:
                for d in ["fwd", "rev"]:
                    s, step = sentence(ag, a, route, b) if d == "fwd" else sentence(ag, b, rroute, a)
                    out.append(dict(kind=kind, path=pn, agent=ag, dir=d, sent=s, step=step))
    # speed variant (water, arms): quickly
    for pn in ["arm_palm_left", "arm_back_left", "arm_palm_right", "arm_back_right"]:
        a, route, b, _ = PATHS[pn]; s = f"A single drop of cool water starts at {a} and rolls quickly {route} toward {b}."
        out.append(dict(kind="body", path=pn, agent="water_fast", dir="fwd", sent=s, step="still rolling"))
    # static: no motion verb, steps 'still there'
    for pn in PATHS:
        a = PATHS[pn][0]; out.append(dict(kind="static", path=pn, agent="water", dir="none", sent=f"A single drop of cool water sits on {a}.", step="still there"))
    for it in out:
        it["text"] = it["sent"] + "".join(f" … {it['step']}" for _ in range(K))
    return out
if __name__ == "__main__":
    I = items(); print(len(I))
    for i in I[::13]: print(i["text"][:220])

# hand and face paths (fine-grained). Same design: only start and end named in the sentence, then identical steps.
HF = {
 "palm_to_index_left": ("the heel of your left hand", "across your left palm and up your left index finger", "the tip of your left index finger", "down your left index finger and across your palm"),
 "palm_across_left": ("the thumb side of your left palm", "across your left palm", "the little-finger edge of your left palm", "across your left palm"),
 "backhand_left": ("the back of your left wrist", "along the back of your left hand", "the knuckle at the base of your left middle finger", "along the back of your left hand"),
 "face_up": ("the tip of your chin", "up over your lips and nose", "the middle of your forehead", "down over your nose and lips"),
 "cheek_left": ("the front of your left ear", "across your left cheek", "the left side of your nose", "across your left cheek"),
}
def items_hf():
    out = []
    for kind, P in [("body", HF), ("nonbody", NONBODY)]:
        for pn, (a, route, b, rroute) in P.items():
            for ag in AGENTS:
                for d in ["fwd", "rev"]:
                    s, step = sentence(ag, a, route, b) if d == "fwd" else sentence(ag, b, rroute, a)
                    out.append(dict(kind=kind, path=pn, agent=ag, dir=d, sent=s, step=step))
    for it in out: it["text"] = it["sent"] + "".join(f" … {it['step']}" for _ in range(K))
    return out
