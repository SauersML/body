import re
OWN = {"self": dict(S="You", P="Your", be="are", have="have", v=0), "her": dict(S="She", P="Her", be="is", have="has", v=1),
       "his": dict(S="He", P="His", be="is", have="has", v=1), "woman": dict(S="The woman", P="The woman's", be="is", have="has", v=1),
       "friend": dict(S="A friend", P="A friend's", be="is", have="has", v=1)}
T = {"touch": ["A feather brushes across {p} left hand.", "Someone presses a thumb into {p} shoulder.", "An ice cube touches {p} neck.", "A mosquito bites {p} ankle.",
  "{P} back is being massaged.", "Someone taps {p} knee twice.", "A cold drop of rain lands on {p} forehead.", "A cat rubs against {p} leg.",
  "The doctor presses on {p} belly.", "Someone squeezes {p} right hand.", "A sharp pin pricks {p} finger.", "Warm sunlight falls on {p} face."],
 "feeling": ["{S} [feel] a wave of sadness.", "{S} {be} suddenly very happy.", "{S} [feel] anxious about tomorrow.", "{S} {be} angry at the delay.",
  "{S} [feel] lonely tonight.", "{S} {be} proud of the work.", "{S} [feel] embarrassed after the mistake.", "{S} {be} bored and restless.",
  "{S} [feel] grateful for the help.", "{S} {be} afraid of the dark.", "{S} [feel] calm and content.", "{S} {be} jealous of the attention."],
 "belief": ["{S} [believe] it will rain later.", "{S} [think] the meeting is on Friday.", "{S} {be} sure the keys are in the drawer.", "{S} [doubt] the story is true.",
  "{S} [believe] honesty matters most.", "{S} [think] the bridge is safe.", "{S} [remember] the door was locked.", "{S} [expect] the train to be late.",
  "{S} {be} convinced the answer is seven.", "{S} [suspect] the milk has gone off.", "{S} [know] the way home.", "{S} [think] the movie was too long."],
 "action": ["{S} [open] the window.", "{S} [pick] up the cup.", "{S} [walk] to the station.", "{S} [write] a letter to the bank.", "{S} [close] the book and [stand] up.",
  "{S} [cook] dinner.", "{S} [turn] off the light.", "{S} [sign] the form.", "{S} [push] the door open.", "{S} [wave] goodbye.", "{S} [pour] a glass of water.", "{S} [tie] the shoelaces."],
 "possession": ["{P} phone is on the table.", "{P} bag is by the door.", "{P} car is parked outside.", "{P} keys are in the drawer.", "{P} coat hangs on the hook.",
  "{P} book is open on the desk.", "{P} umbrella is wet.", "{P} watch has stopped.", "{P} bike has a flat tyre.", "{P} cup is still warm.", "{P} letter arrived today.", "{P} wallet is missing."]}
def conj(v, third): return v if not third else (v + "es" if re.search(r"(sh|ch|x|o)$", v) else v + "s")
def fill(t, o):
    d = OWN[o]; s = t.replace("{S}", d["S"]).replace("{P}", d["P"]).replace("{p}", d["P"] if d["P"] in ("Your", "Her", "His") and False else d["P"].lower() if not d["P"].startswith(("The", "A ")) else d["P"][0].lower() + d["P"][1:])
    s = s.replace("{be}", d["be"]).replace("{have}", d["have"]); s = re.sub(r"\[(\w+)\]", lambda m: conj(m.group(1), d["v"]), s); return s
ROWS = [dict(o=o, c=c, i=i, text=fill(t, o)) for c in T for i, t in enumerate(T[c]) for o in OWN]
PARTS = ["hand", "foot", "knee", "shoulder", "ear", "elbow", "wrist", "cheek"]
LRT = ["A feather brushes across {q} {s} {x}.", "Someone presses a thumb into {q} {s} {x}.", "An ice cube touches {q} {s} {x}.", "A mosquito bites {q} {s} {x}.",
       "Someone taps {q} {s} {x} twice.", "{Q} {s} {x} is itchy."]
LR = [dict(s=s, text=t.replace("{q}", q).replace("{Q}", q.capitalize()).replace("{s}", s).replace("{x}", x)) for t in LRT for x in PARTS for s in ["left", "right"] for q in ["your", "her"]]
