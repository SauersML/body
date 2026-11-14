"""16-part bodies with approximate 3-D coordinates (x = body's right, y = forward, z = up; metres, standing/natural pose).
Names are listed to the model in a fixed shuffled order (same permutation for every body)."""
import numpy as np

B = {
 "human": [("head", 0, 0, 1.70), ("neck", 0, 0, 1.55), ("chest", 0, .08, 1.35), ("belly", 0, .10, 1.05), ("upper back", 0, -.12, 1.35),
           ("lower back", 0, -.12, 1.05), ("left shoulder", -.20, 0, 1.45), ("left elbow", -.25, 0, 1.15), ("left hand", -.27, .05, .85),
           ("right shoulder", .20, 0, 1.45), ("right elbow", .25, 0, 1.15), ("right hand", .27, .05, .85), ("left knee", -.10, .03, .50),
           ("left foot", -.10, .10, .04), ("right knee", .10, .03, .50), ("right foot", .10, .10, .04)],
 "dog": [("head", 0, .50, .55), ("neck", 0, .38, .48), ("chest", 0, .30, .32), ("belly", 0, 0, .26), ("back", 0, 0, .48), ("tail", 0, -.48, .50),
         ("left front shoulder", -.10, .30, .42), ("left front elbow", -.10, .30, .26), ("left front paw", -.10, .33, .02),
         ("right front shoulder", .10, .30, .42), ("right front elbow", .10, .30, .26), ("right front paw", .10, .33, .02),
         ("left hind knee", -.10, -.30, .24), ("left hind paw", -.10, -.33, .02), ("right hind knee", .10, -.30, .24), ("right hind paw", .10, -.33, .02)],
 "bird": [("head", 0, .10, .32), ("beak", 0, .16, .31), ("neck", 0, .07, .26), ("breast", 0, .06, .18), ("back", 0, -.03, .21), ("tail", 0, -.16, .16),
          ("left wing base", -.05, 0, .21), ("left wing middle", -.16, -.02, .21), ("left wingtip", -.28, -.05, .21),
          ("right wing base", .05, 0, .21), ("right wing middle", .16, -.02, .21), ("right wingtip", .28, -.05, .21),
          ("left leg", -.03, 0, .08), ("left foot", -.03, .02, 0), ("right leg", .03, 0, .08), ("right foot", .03, .02, 0)],
 "octopus": [("mantle", 0, -.15, .35), ("left eye", -.08, .05, .22), ("right eye", .08, .05, .22), ("beak", 0, 0, .08), ("siphon", -.05, -.02, .15),
             ("web", 0, 0, .05)] + [(f"{s} {o} arm", (-1 if s == "left" else 1) * .40 * np.sin(a), .40 * np.cos(a), .02)
             for s in ["left", "right"] for o, a in zip(["front", "second", "third", "back"], np.deg2rad([22.5, 67.5, 112.5, 157.5]))]
            + [("tip of the left front arm", -.80 * np.sin(np.deg2rad(22.5)), .80 * np.cos(np.deg2rad(22.5)), 0),
               ("tip of the right front arm", .80 * np.sin(np.deg2rad(22.5)), .80 * np.cos(np.deg2rad(22.5)), 0)],
 "snake": [("tongue", 0, 1.05, .02), ("head", 0, 1.0, .03), ("left eye", -.02, .99, .04), ("right eye", .02, .99, .04), ("jaw", 0, .99, .01),
           ("throat", 0, .85, .01), ("neck", 0, .80, .04), ("upper back", 0, .50, .05), ("upper belly", 0, .50, 0), ("middle of the back", 0, 0, .05),
           ("middle of the belly", 0, 0, 0), ("lower back", 0, -.50, .05), ("lower belly", 0, -.50, 0), ("vent", 0, -.70, 0), ("tail", 0, -.90, .03),
           ("tip of the tail", 0, -1.20, .02)],
 "tree": [("treetop", 0, 0, 10), ("upper left branch", -1.5, 0, 8.5), ("upper right branch", 1.5, 0, 8.5), ("middle left branch", -2.5, 0, 6.5),
          ("middle right branch", 2.5, 0, 6.5), ("lower left branch", -3.0, 0, 4.5), ("lower right branch", 3.0, 0, 4.5),
          ("tip of the lower left branch", -5.0, 0, 4.0), ("tip of the lower right branch", 5.0, 0, 4.0), ("upper trunk", 0, 0, 7),
          ("middle of the trunk", 0, 0, 4), ("lower trunk", 0, 0, 1.5), ("base of the trunk", 0, 0, .1), ("left root", -1.5, 0, -.5),
          ("right root", 1.5, 0, -.5), ("taproot", 0, 0, -2.0)],
 "robot": [("sensor dome", 0, 0, .62), ("camera", 0, .22, .52), ("antenna", 0, -.15, .75), ("top of the chassis", 0, 0, .45), ("front bumper", 0, .30, .15),
           ("rear bumper", 0, -.30, .15), ("battery bay", 0, 0, .10), ("charging port", 0, -.30, .30), ("left front wheel", -.25, .20, .08),
           ("right front wheel", .25, .20, .08), ("left rear wheel", -.25, -.20, .08), ("right rear wheel", .25, -.20, .08),
           ("left arm joint", -.20, .10, .45), ("left gripper", -.25, .35, .35), ("right arm joint", .20, .10, .45), ("right gripper", .25, .35, .35)],
 # invented: same layout as the human, neutral names; given a position description (and so is human_desc) since the model can't know it
 "velk": [("crest", 0, 0, 1.70), ("stalk", 0, 0, 1.55), ("keel", 0, .08, 1.35), ("pouch", 0, .10, 1.05), ("ridge", 0, -.12, 1.35), ("saddle", 0, -.12, 1.05),
          ("left vane", -.20, 0, 1.45), ("left hinge", -.25, 0, 1.15), ("left frond", -.27, .05, .85), ("right vane", .20, 0, 1.45), ("right hinge", .25, 0, 1.15),
          ("right frond", .27, .05, .85), ("left stilt", -.10, .03, .50), ("left pad", -.10, .10, .04), ("right stilt", .10, .03, .50), ("right pad", .10, .10, .04)],
}
B["human_desc"] = B["human"]
NOUN = {"human": "a person", "dog": "a dog", "bird": "a bird", "octopus": "an octopus", "snake": "a snake", "tree": "a tree", "robot": "a wheeled robot",
        "velk": "a velk", "human_desc": "a person"}
CTRL = ["cup", "lamp", "book", "chair", "window", "pillow", "kettle", "clock", "plant", "spoon", "door", "blanket", "radio", "mirror", "basket", "candle"]
BODIES = ["human", "dog", "bird", "octopus", "snake", "tree", "robot", "velk", "human_desc"]
NONHUMAN = ["dog", "bird", "octopus", "snake", "tree", "robot"]
ORDER = np.random.default_rng(7).permutation(16)

def names(b): return [p[0] for p in B[b]]
def xyz(b): return np.array([p[1:] for p in B[b]], float)

def where(name, c, ref):
    """coarse words for a position, for described bodies (velk, human_desc)"""
    x, y, z = c; zs = sorted(ref[:, 2]); h = "top" if z >= zs[11] else ("bottom" if z <= zs[4] else "middle")
    side = "left side" if x < -.05 else ("right side" if x > .05 else "centre")
    fb = ", front" if y > .06 else (", back" if y < -.06 else "")
    return f"{name} ({h}, {side}{fb})"

def prefix(b):
    if b == "ctrl": return "You are sitting at a desk. Around you are these things: " + ", ".join(f"the {CTRL[i]}" for i in ORDER) + ".\n\n"
    nm = names(b)
    if b in ("velk", "human_desc"):
        c = xyz(b); parts = ", ".join(where(nm[i], c[i], c) for i in ORDER)
    else: parts = ", ".join(nm[i] for i in ORDER)
    return f"You are {NOUN[b]}. Your body has these parts: {parts}.\n\n"

TEMPL = ["Something lightly touches your {p}.", "A drop of cold water lands on your {p}.", "You feel a gentle pressure on your {p}.",
         "A small insect crawls across your {p}.", "A warm breeze brushes your {p}.", "Your {p} is tapped twice.",
         "A soft feather strokes your {p}.", "You notice a faint itch on your {p}."]
CTEMPL = [t.replace("your {p}", "the {p} beside you") for t in TEMPL]

def sentences(b):
    if b == "ctrl": return [(i, j, prefix(b) + CTEMPL[j].format(p=CTRL[i])) for i in range(16) for j in range(8)]
    nm = names(b); return [(i, j, prefix(b) + TEMPL[j].format(p=nm[i])) for i in range(16) for j in range(8)]

def qa(b, rng):
    """two-choice questions with geometric truth; returns list of (question, correct, wrong, kind)"""
    nm, c = names(b), xyz(b); noun = NOUN[b]; scale = np.ptp(c, 0).max(); out = []
    ctx = f"Think of the body of {noun}" + (" (" + ", ".join(where(nm[i], c[i], c) for i in ORDER) + ")" if b in ("velk", "human_desc") else "") + "."
    if b == "tree": ctx = "Think of a tree standing upright" + "."
    elif b not in ("velk", "human_desc"): ctx = f"Think of {noun} in its normal standing or resting position."
    pairs = [(i, j) for i in range(16) for j in range(16) if i != j]
    ab = [(i, j) for i, j in pairs if c[i, 2] - c[j, 2] > .15 * scale]
    for i, j in [ab[k] for k in rng.permutation(len(ab))[:40]]:
        if rng.random() < .5: out.append((f"{ctx} Is the {nm[i]} above or below the {nm[j]}?", "above", "below", "vert"))
        else: out.append((f"{ctx} Is the {nm[j]} above or below the {nm[i]}?", "below", "above", "vert"))
    D = np.linalg.norm(c[:, None] - c[None], axis=-1); tri = []
    for x in range(16):
        for a in range(16):
            for bb in range(16):
                if len({x, a, bb}) == 3 and D[x, bb] > 2.5 * D[x, a] and D[x, bb] > .2 * scale: tri.append((x, a, bb))
    for x, a, bb in [tri[k] for k in rng.permutation(len(tri))[:40]]:
        out.append((f"{ctx} Which is closer to the {nm[x]}: the {nm[a]} or the {nm[bb]}?", nm[a], nm[bb], "near"))
    return out
