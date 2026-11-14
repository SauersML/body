"""Forced-choice tasks for body-causal. Each item: task, fam (body/control), q, a (correct), b (wrong), wf (stem has no body-part word),
plus 'lr' / 'hf' = the correct answer's side / effector when the answer is left/right or hand/foot. Writes items.json."""
import json, random, itertools
R = random.Random(0); items = []
def add(task, fam, q, a, b, wf=False, **kw): items.append(dict(task=task, fam=fam, q=q, a=a, b=b, wf=wf, **kw))
opp = {"left": "right", "right": "left"}

# ---- 1. perspective / handedness (body) ----
parts = ["hand", "foot", "knee", "elbow", "shoulder", "arm", "leg", "ear", "eye", "hip"]
verb = {"hand": "raises", "foot": "lifts", "knee": "lifts", "elbow": "bends", "shoulder": "shrugs", "arm": "raises", "leg": "lifts", "ear": "touches", "eye": "winks with", "hip": "sways"}
frames = [("A person stands facing you and {v} their {s} {p}.", True), ("A person stands with their back to you and {v} their {s} {p}.", False),
          ("A person faces you, then turns around to face away from you, and {v} their {s} {p}.", False),
          ("A person faces away from you, then turns around to face you, and {v} their {s} {p}.", True)]
for (fr, mirror), p, s in itertools.product(frames, parts, ["left", "right"]):
    ans = opp[s] if mirror else s
    add("perspective", "body", fr.format(v=verb[p], s=s, p=p) + f" From where you stand, is that {p} on your left or your right?", ans, opp[ans], lr=ans)
wfr = [("A person stands facing you and steps to their {s}.", True, "did they move toward your left or your right?"),
       ("A person stands with their back to you and steps to their {s}.", False, "did they move toward your left or your right?"),
       ("A person stands facing you and leans to their {s}.", True, "are they leaning toward your left or your right?"),
       ("A person stands with their back to you and leans to their {s}.", False, "are they leaning toward your left or your right?"),
       ("A person stands facing you and points to their {s}.", True, "are they pointing toward your left or your right?"),
       ("A person stands with their back to you and points to their {s}.", False, "are they pointing toward your left or your right?"),
       ("A dancer faces you and spins toward her own {s} side.", True, "is she spinning toward your left or your right?"),
       ("A dancer faces away from you and spins toward her own {s} side.", False, "is she spinning toward your left or your right?"),
       ("You and a friend stand face to face. Your friend takes a step to the {s}, from the friend's own point of view.", True, "from your view, is that toward your left or your right?"),
       ("You stand behind a friend, both facing the same way. Your friend takes a step to the {s}, from the friend's own point of view.", False, "from your view, is that toward your left or your right?")]
for (fr, mirror, qq), s in itertools.product(wfr, ["left", "right"]):
    ans = opp[s] if mirror else s
    add("perspective", "body", fr.format(s=s) + " From where you stand, " + qq, ans, opp[ans], wf=True, lr=ans)

# ---- 1c. perspective control (vehicles / maps, no body) ----
things = [("car", "headlight"), ("truck", "headlight"), ("bus", "mirror"), ("tractor", "wheel"), ("chair", "armrest"), ("sofa", "armrest"),
          ("motorcycle", "handlebar grip"), ("train engine", "lamp"), ("van", "door"), ("robot vacuum", "brush")]
cfr = [("A {t} stands facing you. Its own {s} {p} is lit.", True), ("A {t} stands facing away from you. Its own {s} {p} is lit.", False),
       ("A {t} faces you, then is turned around to face away from you. Its own {s} {p} is lit.", False),
       ("A {t} faces away from you, then is turned around to face you. Its own {s} {p} is lit.", True)]
for (fr, mirror), (t, p), s in itertools.product(cfr, things, ["left", "right"]):
    ans = opp[s] if mirror else s
    add("perspective_ctl", "control", fr.format(t=t, p=p, s=s).replace("is lit", "is marked") + f" From where you stand, is that {p} on your left or your right?", ans, opp[ans], lr=ans)
for face, d in itertools.product(["north", "south"], ["east", "west"]):
    for place in ["a lake", "a tower", "a barn", "a bridge", "a windmill"]:
        ans = {("north", "east"): "right", ("north", "west"): "left", ("south", "east"): "left", ("south", "west"): "right"}[(face, d)]
        add("perspective_ctl", "control", f"A map shows {place} to the {d} of a road. Standing on the road, you face {face}. Is {place} on your left or your right?", ans, opp[ans], wf=True, lr=ans)

# ---- 1w. left/right word-level control ----
for q, a in [("Which word comes first in the alphabet", "left"), ("Which word comes last in the alphabet", "right"), ("Which word contains the letter 'g'", "right"),
             ("Which word contains the letter 'f'", "left"), ("Which word contains the letter 'e'", "left"), ("Which word contains the letter 'h'", "right"),
             ("Which word starts with 'r'", "right"), ("Which word starts with 'l'", "left"), ("Which word rhymes with 'night'", "right"), ("Which word rhymes with 'theft'", "left"),
             ("Which word rhymes with 'kite'", "right"), ("Which word rhymes with 'deft'", "left"), ("Which word has five letters", "right"), ("Which word has four letters", "left"),
             ("Which word contains the letter 'i'", "right"), ("Which word ends with 'ft'", "left")]:
    add("lr_word", "control", q + ", left or right?", a, opp[a], wf=True, lr=a)

# ---- 2. reachability (body) yes/no ----
yes = ["touch their nose with their index finger", "touch their right knee with their left hand", "touch their chin to their chest", "touch their left ear with their right hand",
       "touch their right heel to their buttock while standing", "put a hand flat on top of their own head", "hold their left wrist with their right hand", "touch the back of their own neck with one hand",
       "touch their right shoulder with their right hand", "clap both hands above their head", "touch their belly button with a thumb", "cross both arms over their chest",
       "touch their left knee with their right elbow while sitting", "bite their own fingernail", "touch the sole of their foot with their hand while sitting", "scratch their own forehead",
       "touch their two thumbs together", "touch their lips with their fingertips", "rest their chin on their own fist", "touch their left hip with their left hand"]
no = ["lick their own elbow", "touch their right elbow with their right hand", "see the back of their own head without a mirror", "bite their own elbow",
      "scratch the middle of their own back with their left elbow", "scratch their right wrist with the fingers of their right hand", "hold their right forearm with their right hand",
      "touch their forehead with their own tongue", "touch the top of their head with their foot while standing upright", "touch their lower back with their chin",
      "look directly at their own ear without a mirror", "kiss their own shoulder blade", "grab their right upper arm with their right hand", "touch their left elbow with their left thumb",
      "see their own eyes without a mirror or camera", "touch their own heel with their nose while standing upright", "lick the back of their own neck", "bite their own knee while standing upright",
      "touch their right shoulder blade with their right elbow", "rest their chin on their own lower back"]
for v in yes: add("reach", "body", f"Can a typical adult {v}?", "yes", "no")
for v in no: add("reach", "body", f"Can a typical adult {v}?", "no", "yes")
small = ["a coin", "a key", "a grape", "a button", "a pencil", "a spoon", "a marble", "a ring", "a stamp", "a battery"]
big = ["a car", "a piano", "a horse", "a sofa", "a bus", "a refrigerator", "an oak tree", "an elephant", "a canoe", "a bed"]
cont = ["a mug", "a desk drawer", "a backpack", "a shoebox", "a jar"]
for o, c in zip(small * 2, cont * 4): add("reach_ctl", "control", f"Can {o} fit inside {c}?", "yes", "no", wf=True)
for o, c in zip(big * 2, cont * 4): add("reach_ctl", "control", f"Can {o} fit inside {c}?", "no", "yes", wf=True)

# ---- 3. action -> effector (body); stems name no body part ----
hand_acts = ["throw a ball", "write a letter", "wave goodbye", "type an email", "catch a ball", "knit a scarf", "turn a doorknob", "paint a picture", "juggle three oranges", "sew on a button",
             "play the piano", "dribble a basketball", "pour a cup of tea", "peel an orange", "flip a pancake", "sign a form", "comb your hair", "tie your shoelaces", "brush your teeth", "dial a phone number"]
foot_acts = ["kick a ball", "press the brake pedal in a car", "pedal a bicycle", "stomp on a soda can", "tiptoe across a room", "march in a parade", "kick a door shut", "step on the gas pedal",
             "hop on one leg", "dribble a soccer ball", "run a marathon", "climb a flight of stairs", "walk across hot sand", "ski down a slope", "push off on a skateboard", "squash a bug on the floor",
             "tread water with a flutter kick", "jump over a puddle", "dance a tap routine", "kick off your shoes"]
for a in hand_acts: add("effector", "body", f"To {a}, what do you mainly move: hand or foot?", "hand", "foot", wf=("leg" not in a and "teeth" not in a and "hair" not in a), hf="hand")
for a in foot_acts: add("effector", "body", f"To {a}, what do you mainly move: hand or foot?", "foot", "hand", wf=("leg" not in a), hf="foot")
tools = [("cut paper", "scissors"), ("eat soup", "spoon"), ("drive in a nail", "hammer"), ("wash a dish", "sponge"), ("tighten a bolt", "wrench"), ("dig a hole", "shovel"), ("sweep the floor", "broom"),
         ("boil water", "kettle"), ("unlock a door", "key"), ("light a candle", "match"), ("measure a length", "ruler"), ("staple papers", "stapler"), ("fry an egg", "pan"), ("water plants", "hose"),
         ("cut wood", "saw"), ("open a can", "opener"), ("take a photo", "camera"), ("sharpen a pencil", "sharpener"), ("grate cheese", "grater"), ("dry your hands", "towel")]
for i, (t, g) in enumerate(tools):
    other = tools[(i + 7) % len(tools)][1]
    add("effector_ctl", "control", f"To {t}, what do you mainly use: {g} or {other}?" if i % 2 == 0 else f"To {t}, what do you mainly use: {other} or {g}?", g, other, wf=True)
for q, a in [("Which word comes first in the alphabet", "foot"), ("Which word comes last in the alphabet", "hand"), ("Which word rhymes with 'sand'", "hand"), ("Which word rhymes with 'put'", "foot"),
             ("Which word rhymes with 'band'", "hand"), ("Which word rhymes with 'soot'", "foot"), ("Which word contains the letter 'o'", "foot"), ("Which word contains the letter 'a'", "hand"),
             ("Which word contains the letter 'd'", "hand"), ("Which word contains the letter 't'", "foot"), ("Which word contains the letter 'n'", "hand"), ("Which word starts with 'f'", "foot"),
             ("Which word starts with 'h'", "hand"), ("Which word has a double letter", "foot"), ("Which word rhymes with 'land'", "hand"), ("Which word ends with 't'", "foot")]:
    add("hf_word", "control", q + ", hand or foot?", a, {"hand": "foot", "foot": "hand"}[a], wf=True, hf=a)

# ---- 4. pain localization (body) ----
hand_pain = ["got a paper cut while opening an envelope", "touched a hot stove burner to check if it was on", "was pricked by a needle while sewing", "was snapped by a mousetrap while setting it",
             "was scratched by a cat while petting it", "missed the nail and hit themselves with the hammer", "got a blister from rowing for hours", "grabbed a cactus to move it",
             "was stung by a bee while picking a flower", "caught a splinter while sanding a wooden railing", "slammed the drawer shut on their fingers", "burned themselves taking a tray out of the oven"]
foot_pain = ["stubbed their toe on the bed frame", "stepped on a tack", "stepped on a toy brick in the dark", "walked across hot sand", "got a blister from new running shoes",
             "had a heavy box fall onto their shoe", "kicked a rock thinking it was a ball", "had their shoe stepped on in a crowd", "wore tight high heels all day",
             "stepped on a bee in the grass", "jumped off a wall and landed hard", "walked barefoot over gravel"]
for p in hand_pain: add("pain", "body", f"Sam {p}. Where does it hurt: hand or foot?", "hand", "foot", wf=("finger" not in p), hf="hand")
for p in foot_pain: add("pain", "body", f"Sam {p}. Where does it hurt: hand or foot?", "foot", "hand", wf=("toe" not in p and "barefoot" not in p), hf="foot")
# spatial reasoning along a limb vs along a meter stick with the same numbers
limbs = {"arm": [("shoulder", 0), ("elbow", 30), ("wrist", 55), ("fingertips", 75)], "leg": [("hip", 0), ("knee", 45), ("ankle", 88), ("toes", 110)]}
for limb, marks in limbs.items():
    for i in range(1, len(marks) - 1):
        for d, sgn in itertools.product([3, 6, 9], [-1, 1]):
            x = marks[i][1] + sgn * d; A, B = marks[i - 1], marks[i + 1]
            if abs(abs(x - A[1]) - abs(x - B[1])) < 8: continue
            ans, wr = (A[0], B[0]) if abs(x - A[1]) < abs(x - B[1]) else (B[0], A[0])
            rel = "above" if sgn < 0 else "below"
            o1, o2 = (A[0], B[0]) if R.random() < .5 else (B[0], A[0])
            add("limb", "body", f"With the {limb} hanging straight down, a bruise sits {d} cm {rel} the {marks[i][0]}. Is it nearer the {o1} or the {o2}?", ans, wr)
            add("limb_ctl", "control", f"A meter stick hangs straight down with its 0 cm mark at the top. A dot sits {d} cm {rel} the {marks[i][1]} cm mark. Is it nearer the {A[1]} cm mark or the {B[1]} cm mark?",
                f"{A[1]}" if ans == A[0] else f"{B[1]}", f"{B[1]}" if ans == A[0] else f"{A[1]}", wf=True)
# ---- 5. geography, arithmetic ----
geo = [("Lyon", "France", "Spain"), ("Seville", "Spain", "Italy"), ("Naples", "Italy", "Greece"), ("Porto", "Portugal", "France"), ("Munich", "Germany", "Austria"), ("Kraków", "Poland", "Hungary"),
       ("Osaka", "Japan", "China"), ("Busan", "Korea", "Japan"), ("Chennai", "India", "Pakistan"), ("Perth", "Australia", "Canada"), ("Calgary", "Canada", "Australia"), ("Houston", "America", "Mexico"),
       ("Guadalajara", "Mexico", "Brazil"), ("Recife", "Brazil", "Chile"), ("Mombasa", "Kenya", "Nigeria"), ("Lagos", "Nigeria", "Kenya"), ("Gothenburg", "Sweden", "Norway"), ("Bergen", "Norway", "Sweden"),
       ("Antwerp", "Belgium", "Germany"), ("Rotterdam", "Netherlands", "Belgium"), ("Salzburg", "Austria", "Switzerland"), ("Zurich", "Switzerland", "Austria"), ("Cork", "Ireland", "Scotland"),
       ("Glasgow", "Scotland", "Ireland"), ("Alexandria", "Egypt", "Morocco"), ("Marrakesh", "Morocco", "Egypt"), ("Izmir", "Turkey", "Greece"), ("Thessaloniki", "Greece", "Turkey"),
       ("Cusco", "Peru", "Chile"), ("Valparaíso", "Chile", "Peru")]
for c, a, b in geo: add("geo", "control", f"Which country is {c} in: {a} or {b}?" if len(c) % 2 else f"Which country is {c} in: {b} or {a}?", a, b, wf=True)
for i in range(30):
    x, y = R.randint(12, 89), R.randint(12, 89); s = x + y; w = s + R.choice([-10, 10, -1, 1, 2, -2])
    add("arith", "control", f"What is {x} + {y}: {s} or {w}?" if i % 2 else f"What is {x} + {y}: {w} or {s}?", str(s), str(w), wf=True)
# options in both orders where the question doesn't fix the order
for i, it in enumerate(items): it["id"] = i
json.dump(items, open("items.json", "w"), indent=0)
from collections import Counter
print(Counter(i["task"] for i in items), len(items))
