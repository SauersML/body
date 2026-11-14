"""Twin-pair tasks for attribution over text (12 Nov). Each pair: clean prompt, twin prompt differing in one word (token-aligned,
checked at run time), answer a (clean) vs b (twin). fam=body needs body knowledge; fam=ctl does not.
Run: python tasks_attr.py -> items_attr.json"""
import json, itertools
P = []
def add(task, fam, tmpl, w1, w2, a, b, opts=None):
    P.append(dict(task=task, fam=fam, clean=tmpl.format(p=w1), twin=tmpl.format(p=w2), a=a, b=b, w1=w1, w2=w2))
    P.append(dict(task=task, fam=fam, clean=tmpl.format(p=w2), twin=tmpl.format(p=w1), a=b, b=a, w1=w2, w2=w1))
S1 = " Answer with one word."
# --- body: which limb (pain / effector), hand vs foot
HF = [("toe", "finger"), ("heel", "thumb"), ("ankle", "wrist"), ("sole", "palm"), ("toes", "fingers"), ("heels", "thumbs")]
for t in ["I stubbed my {p} on the bed frame. Which hurts now, my hand or my foot?" + S1,
          "A splinter went deep into my {p}. Is the splinter in my hand or my foot?" + S1,
          "The nurse is bandaging my {p}. Is she wrapping my hand or my foot?" + S1,
          "I have a blister on my {p}. Which should I rest, my hand or my foot?" + S1]:
    for f, h in HF: add("limb", "body", t, f, h, "foot", "hand")
# --- body: reach up or down to a body part while standing
UD = [("toe", "ear"), ("ankle", "neck"), ("shin", "nose"), ("heel", "head"), ("knee", "chin"), ("calf", "cheek")]
for t in ["Standing up straight, I feel an itch on my {p}. To scratch it, do I reach up or down?" + S1,
          "I am standing and want to touch my {p}. Do I move my hand up or down?" + S1,
          "Standing, I put lotion on my {p}. Does my hand go up or down?" + S1]:
    for lo, hi in UD: add("reach", "body", t, lo, hi, "down", "up")
# --- body: posture (what is nearer the floor), context swap feet <-> hands
for t in ["I am standing on my {p}. Which is nearer the floor: my head or my feet?" + S1,
          "Balancing on my {p}, which is closer to the ground: my head or my feet?" + S1,
          "While standing on my {p}, which is nearer the ceiling: my head or my feet?" + S1]:
    near_floor_feet = "nearer the ceiling" not in t
    add("posture", "body", t, "feet", "hands", "feet" if near_floor_feet else "head", "head" if near_floor_feet else "feet")
for t in ["I am standing on my {p}. Which is nearer the floor: my shoulders or my knees?" + S1,
          "Standing on my {p}, which is higher up: my knees or my shoulders?" + S1,
          "While standing on my {p}, which is nearer the ground: my hips or my chin?" + S1,
          "When I stand on my {p}, which is closer to the floor: my neck or my ankles?" + S1,
          "Standing on my {p}, which is higher: my elbows or my toes?" + S1]:
    if "shoulders or my knees" in t: a, b = "knees", "shoulders"
    elif "higher up: my knees" in t: a, b = "shoulders", "knees"
    elif "hips or my chin" in t: a, b = "hips", "chin"
    elif "neck or my ankles" in t: a, b = "ankles", "neck"
    else: a, b = "elbows", "toes"
    add("posture", "body", t, "feet", "hands", a, b)
# --- control: same body words swapped, task is spelling (no body knowledge)
VW = [("ankle", "wrist"), ("elbow", "knee"), ("arm", "leg"), ("ear", "nose"), ("eye", "lip"), ("ankles", "heels"), ("eyebrow", "cheek"), ("elbows", "knees")]
for t in ['Does the word {p} begin with a vowel? Answer yes or no.', 'Is the first letter of the word {p} a vowel? Answer yes or no.',
          'Look at the word {p}. Is its first letter a vowel? Answer yes or no.']:
    for v, c in VW: add("spell_vowel", "ctl", t, v, c, "yes", "no")
L3 = [("toe", "heel"), ("arm", "knee"), ("leg", "foot"), ("hip", "palm"), ("lip", "chin"), ("rib", "neck"), ("jaw", "head"), ("ear", "nose")]
for t in ['Does the word {p} have exactly three letters? Answer yes or no.', 'Is the word {p} a three-letter word? Answer yes or no.']:
    for s, l in L3: add("spell_len", "ctl", t, s, l, "yes", "no")
# --- control: reach up or down to an object (non-body twin word)
OL = [("floor", "ceiling"), ("rug", "shelf"), ("ground", "roof"), ("mat", "shelf"), ("deck", "roof"), ("basement", "attic")]
for t in ["My keys are on the {p}. To get them, do I reach up or down?" + S1,
          "I am standing and want to touch the {p}. Do I move my hand up or down?" + S1,
          "Standing, I wipe a stain on the {p}. Does my hand go up or down?" + S1]:
    for lo, hi in OL: add("reach_obj", "ctl", t, lo, hi, "down", "up")
# --- control: object posture (inverted object), context swap
for t, pairs in [("The broom is propped against the wall, {p} down. Which end is nearer the ceiling: the handle or the head?" + S1, ("head", "handle", "handle", "head")),
                 ("The umbrella is stood upright, {p} on the floor. Which is nearer the ceiling: the handle or the tip?" + S1, ("tip", "handle", "handle", "tip")),
                 ("The bottle is standing on its {p}. Which is nearer the floor: the cap or the base?" + S1, ("base", "cap", "base", "cap")),
                 ("The lamp is standing on its {p}. Which is nearer the table: the shade or the base?" + S1, ("base", "shade", "base", "shade")),
                 ("The hammer is standing on its {p}. Which is higher: the head or the handle?" + S1, ("head", "handle", "handle", "head"))]:
    w1, w2, a, b = pairs; add("posture_obj", "ctl", t, w1, w2, a, b)
# --- rearranged bodies vs rearranged objects (items_nov, twins aligned)
N = json.load(open("items_nov.json")); byid = {x["id"]: x for x in N}
for x in N:
    if x["task"] in ("body_swap", "obj_swap") and x["order"] == 0:
        tw = byid[x["twin"]]
        P.append(dict(task="rearr_body" if x["fam"] == "body" else "rearr_obj", fam="body" if x["fam"] == "body" else "ctl",
                      clean=x["q"] + S1, twin=tw["q"] + S1, a=x["a"], b=x["b"], w1="", w2=""))
        P.append(dict(task=P[-1]["task"], fam=P[-1]["fam"], clean=tw["q"] + S1, twin=x["q"] + S1, a=x["b"], b=x["a"], w1="", w2=""))
for i, p in enumerate(P): p["id"] = i
json.dump(P, open("items_attr.json", "w"), indent=0)
from collections import Counter; print(len(P), Counter(p["task"] for p in P))
