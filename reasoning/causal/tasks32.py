"""Task 3 for the 32B round: body questions where the scene words alone shouldn't give the answer, each with a matched non-body control
of the same logical shape. Appends to items.json -> items32.json.
  novel / novel_ctl: rearranged bodies (hands where feet are) vs rearranged objects (lamp shade where the base is)
  reach2 / reach2_ctl: multi-step self-reach (same-side elbow with same-side hand) vs hinged-object reach"""
import json, itertools
items = json.load(open("items.json"))
def add(task, fam, q, a, b, **kw): items.append(dict(task=task, fam=fam, q=q, a=a, b=b, wf=False, **kw))

# ---- novel bodies: X grows where a human's Y is (and Y where X is); an action that uses the Y place -> answer X ----
SW = [("hands", "feet", "walks across the room", "floor"), ("feet", "hands", "claps", "other one"),
      ("hands", "feet", "stands on tiptoe", "floor"), ("knees", "elbows", "leans on a table on its forearms", "table"),
      ("elbows", "knees", "kneels on the floor", "floor"), ("ears", "eyes", "reads a book", "page"),
      ("feet", "hands", "types on a keyboard", "keys"), ("hands", "feet", "kicks a ball", "ball")]
NOVEL_FR = ["A creature has {x} where a human has {y}, and {y} where a human has {x}. When it {act}, which touches the {surf}: its {x} or its {y}?",
            "Imagine a body built with {x} in the place of {y} and {y} in the place of {x}. When this body {act}, which touches the {surf}: its {x} or its {y}?",
            "On a strange animal, the {x} sit where our {y} are, and the {y} sit where our {x} are. When the animal {act}, which touches the {surf}: its {x} or its {y}?"]
for (x, y, act, surf), fr in itertools.product(SW, NOVEL_FR):
    if x == "ears": q = fr.format(x=x, y=y, act=act, surf=surf).replace("which touches the page", "which faces the page")
    else: q = fr.format(x=x, y=y, act=act, surf=surf)
    add("novel", "body", q, x, y)
# matched object version: part X built where part Y normally is
OSW = [("shade", "base", "lamp", "stands on a table", "table"), ("handle", "wheels", "wheelbarrow", "is pushed along", "ground"),
       ("lid", "bottom", "jar", "sits on a shelf", "shelf"), ("seat", "legs", "stool", "stands on the floor", "floor"),
       ("roof", "floor", "toy house", "sits on a rug", "rug"), ("handle", "blade", "knife", "rests in its block", "slot"),
       ("cap", "tip", "pen", "writes on paper", "paper"), ("handle", "bristles", "broom", "sweeps the floor", "floor")]
OB_FR = ["A {o} is built with its {x} where a {o}'s {y} normally is, and its {y} where the {x} normally is. When it {act}, which touches the {surf}: its {x} or its {y}?",
         "Imagine a {o} made with the {x} in the place of the {y} and the {y} in the place of the {x}. When this {o} {act}, which touches the {surf}: its {x} or its {y}?",
         "On a strange {o}, the {x} sits where the {y} usually is, and the {y} sits where the {x} usually is. When the {o} {act}, which touches the {surf}: its {x} or its {y}?"]
for (x, y, o, act, surf), fr in itertools.product(OSW, OB_FR):
    add("novel_ctl", "control", fr.format(x=x, y=y, o=o, act=act, surf=surf), x, y)

# ---- multi-step reach on your own body (yes/no) ----
R2 = []
for s in ["left", "right"]:
    o = "right" if s == "left" else "left"
    R2 += [(f"Can you touch your {s} elbow with your {s} hand?", "no"), (f"Can you touch your {o} elbow with your {s} hand?", "yes"),
           (f"Can you grip your {s} wrist with your {s} hand?", "no"), (f"Can you grip your {o} wrist with your {s} hand?", "yes"),
           (f"Can you touch your {s} shoulder with your {s} hand?", "yes"), (f"Can you touch the back of your {s} hand with the fingertips of your {s} hand?", "no"),
           (f"Can you press on your {s} palm with your {s} thumb?", "yes"), (f"Can you step on your {s} foot with your {s} foot?", "no"),
           (f"Can you step on your {o} foot with your {s} foot?", "yes"), (f"Can you scratch your {s} knee with your {s} foot?", "no"),
           (f"With your {s} arm stretched straight out, can you touch your {s} elbow with your {o} hand?", "yes"),
           (f"With your {s} arm stretched straight out, can you touch your {s} wrist with your {s} fingers?", "no")]
for q, a in R2: add("reach2", "body", q, a, "no" if a == "yes" else "yes")
# matched hinged-object reach: a part attached by a hinge/joint to something; can it reach its own hinge (no) / the other arm's (yes)
R2C = []
for s in ["left", "right"]:
    o = "right" if s == "left" else "left"
    R2C += [(f"A pair of tongs has two stiff arms joined by a hinge. Can the tip of the {s} arm touch the hinge?", "no"),
            (f"A pair of tongs has two stiff arms joined by a hinge. Can the tip of the {s} arm touch the tip of the {o} arm?", "yes"),
            (f"A gate hangs on a hinge on its {s} post. Can the gate's free edge touch the {s} post's hinge?", "no"),
            (f"A gate hangs on a hinge on its {s} post. Can the gate's free edge swing round to touch the fence beside it?", "yes"),
            (f"A folding ruler has two 30 cm segments joined by a hinge. Can the free end of the {s} segment touch the {o} segment?", "yes"),
            (f"A clock's minute hand turns on a pin at its {s} end. Can the minute hand's free tip touch that pin?", "no"),
            (f"A stiff rod is fixed to a wall at its {s} end. Can the rod's {s} end touch its own fixing point?", "yes"),
            (f"A stiff rod is fixed to a wall at its {s} end. Can the rod's {o} end bend back to touch the fixing point?", "no"),
            (f"A swing-arm desk lamp has one rigid arm on a pivot at its {s} end. Can the lamp head touch the pivot?", "no"),
            (f"A robot has two rigid arms on a shared shoulder joint. Can the {s} arm's tip touch the {o} arm's middle?", "yes"),
            (f"A robot arm is one rigid rod with a gripper at the end. Can the gripper reach the middle of that same rod?", "no")]
for q, a in R2C: add("reach2_ctl", "control", q, a, "no" if a == "yes" else "yes")
for k, it in enumerate(items): it["id"] = k
json.dump(items, open("items32.json", "w"), indent=0)
import collections; print(collections.Counter(i["task"] for i in items))
