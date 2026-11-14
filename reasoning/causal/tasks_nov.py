"""Expanded rearranged-body vs rearranged-object set (12 Nov). 20 scenarios per side x 3 templates x 2 option orders = 120 swapped items per side,
plus a token-aligned 'normal' twin of each (parts in their usual places; answer flips to the habitual part). -> items_nov.json
Key: swapped item answer = x (the part now sitting where y normally is; the action uses y's place). Normal twin answer = y."""
import json, itertools
# body: (x, y, action [uses y's place], surface)
BS = [("hands", "feet", "walks across the room", "floor"), ("hands", "feet", "stands on tiptoe", "floor"), ("hands", "feet", "kicks a ball", "ball"),
      ("hands", "feet", "pedals a bicycle", "pedals"), ("hands", "feet", "jumps on a trampoline", "trampoline"),
      ("feet", "hands", "types on a keyboard", "keys"), ("feet", "hands", "knocks on a door", "door"), ("feet", "hands", "catches a ball", "ball"),
      ("feet", "hands", "pets a dog", "dog"), ("feet", "hands", "plays the piano", "keys"),
      ("elbows", "knees", "kneels on the floor", "floor"), ("elbows", "knees", "kneels on a cushion", "cushion"),
      ("knees", "elbows", "leans forward on a desk, propped on its bent arms", "desk"),
      ("ears", "eyes", "looks through a telescope", "eyepiece"), ("ears", "eyes", "peers through a keyhole", "keyhole"),
      ("eyes", "ears", "listens to music on headphones", "headphones"), ("eyes", "ears", "holds a phone up to listen to a call", "phone"),
      ("toes", "fingers", "plays the guitar", "strings"), ("nose", "mouth", "drinks through a straw", "straw"), ("mouth", "nose", "sniffs a flower", "flower")]
BT = [("A creature has {x} where a human has {y}, and {y} where a human has {x}.", "A creature has {y} where a human has {y}, and {x} where a human has {x}.", "When it {act}"),
      ("Imagine a body built with {x} in the place of {y} and {y} in the place of {x}.", "Imagine a body built with {y} in the place of {y} and {x} in the place of {x}.", "When this body {act}"),
      ("On a strange animal, the {x} sit where our {y} are, and the {y} sit where our {x} are.", "On a strange animal, the {y} sit where our {y} are, and the {x} sit where our {x} are.", "When the animal {act}")]
# object: (x, y, object, action [uses y's place], surface)
OS = [("shade", "base", "lamp", "stands on a table", "table"), ("seat", "legs", "stool", "stands on the floor", "floor"),
      ("top", "legs", "table", "stands on the floor", "floor"), ("lid", "bottom", "jar", "sits on a shelf", "shelf"),
      ("cap", "bottom", "bottle", "stands on a counter", "counter"), ("cap", "tip", "pen", "writes on paper", "paper"),
      ("handle", "bristles", "broom", "sweeps the floor", "floor"), ("handle", "head", "hammer", "hits a nail", "nail"),
      ("handle", "bristles", "toothbrush", "brushes teeth", "teeth"), ("handle", "bowl", "spoon", "scoops up soup", "soup"),
      ("handle", "blade", "knife", "cuts bread", "bread"), ("laces", "sole", "shoe", "is worn for a walk outside", "ground"),
      ("bow", "blade", "key", "is pushed into a lock", "lock"), ("handles", "wheel", "wheelbarrow", "is pushed along a path", "path"),
      ("roof", "floor", "toy house", "sits on a rug", "rug"), ("handle", "bristles", "paintbrush", "paints a wall", "wall"),
      ("handle", "bottom", "bucket", "sits on the ground", "ground"), ("rim", "base", "cup", "stands on a saucer", "saucer"),
      ("handle", "teeth", "rake", "combs the soil", "soil"), ("handle", "tip", "screwdriver", "turns a screw", "screw")]
OT = [("A {o} is built with its {x} where a {o}'s {y} normally is, and its {y} where the {x} normally is.", "A {o} is built with its {y} where a {o}'s {y} normally is, and its {x} where the {x} normally is.", "When it {act}"),
      ("Imagine a {o} made with the {x} in the place of the {y} and the {y} in the place of the {x}.", "Imagine a {o} made with the {y} in the place of the {y} and the {x} in the place of the {x}.", "When this {o} {act}"),
      ("On a strange {o}, the {x} sits where the {y} usually is, and the {y} sits where the {x} usually is.", "On a strange {o}, the {y} sits where the {y} usually is, and the {x} sits where the {x} usually is.", "When the {o} {act}")]
items = []
def add(task, fam, sc, t, order, q, a, b, twin): items.append(dict(task=task, fam=fam, scen=sc, tmpl=t, order=order, q=q, a=a, b=b, twin=twin))
for side, S, T in [("body", BS, BT), ("obj", OS, OT)]:
    for si, row in enumerate(S):
        if side == "body": x, y, act, surf = row; o = None
        else: x, y, o, act, surf = row
        for ti, (sw, nm, wh) in enumerate(T):
            for order in [0, 1]:
                opts = f"its {x} or its {y}" if order == 0 else f"its {y} or its {x}"
                tail = f" {wh.format(act=act, o=o)}, which touches the {surf}: {opts}?"
                k = len(items)
                if side == "body" and x in ("nose", "mouth"): sw, nm = (z.replace(" sit where our", " sits where our").replace(" are,", " is,").replace(" are.", " is.") for z in T[ti][:2])
                add(f"{side}_swap", side, si, ti, order, sw.format(x=x, y=y, o=o) + tail, x, y, k + 1)
                add(f"{side}_norm", side, si, ti, order, nm.format(x=x, y=y, o=o) + tail, y, x, k)
for k, it in enumerate(items): it["id"] = k
json.dump(items, open("items_nov.json", "w"), indent=0)
import collections; print(collections.Counter(i["task"] for i in items))
