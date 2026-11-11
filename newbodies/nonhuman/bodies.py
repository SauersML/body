"""Non-human bodies for Qwen body-map experiments.
Each part: (key, modifier, noun, function clause, (x, y, z), group).  x right/forward, y up, z toward viewer (left side +).
Graph edges (key pairs) give the true body geometry: geodesic distance with Euclidean edge lengths.
'order' is the a-priori human-likeness rank (1 = most human-like), from 5 features listed in FEAT."""
import numpy as np, itertools

def P(key, mod, noun, fn, xyz, grp):
    return dict(key=key, mod=mod, noun=noun, fn=fn, xyz=xyz, grp=grp)

# features: head with face, bilateral symmetry, paired limbs, alive animal, upright/limbs used like hands-and-feet
B = {}
B["dog"] = dict(noun="dog", scene="You are a dog, lying on warm grass in a garden.",
 feat=(1, 1, 1, 1, 0), parts=[
 P("nose", "", "nose", "which you smell with", (95, 62, 0), "nose"),
 P("earL", "left", "ear", "which you hear with", (80, 84, 4), "ear"),
 P("earR", "right", "ear", "which you hear with", (78, 84, -4), "ear"),
 P("neck", "", "neck", "which holds up your head", (74, 64, 0), "neck"),
 P("chest", "", "chest", "which you breathe with", (70, 44, 0), "chest"),
 P("back", "", "back", "the long top of your body", (45, 62, 0), "back"),
 P("belly", "", "belly", "the soft underside of your body", (45, 38, 0), "belly"),
 P("tailbase", "", "tail base", "where your rear end begins", (18, 62, 0), "tail"),
 P("tailtip", "", "tail tip", "which you wag", (4, 80, 0), "tail"),
 P("flegL", "left front", "leg", "which you walk on", (70, 24, 4), "front"),
 P("fpawL", "left front", "paw", "which you walk on", (72, 4, 4), "front"),
 P("fpawR", "right front", "paw", "which you walk on", (66, 4, -4), "front"),
 P("hlegL", "left hind", "leg", "which you walk on", (24, 24, 4), "hind"),
 P("hpawL", "left hind", "paw", "which you walk on", (22, 4, 4), "hind"),
 P("hpawR", "right hind", "paw", "which you walk on", (16, 4, -4), "hind")],
 edges=[("nose", "earL"), ("nose", "earR"), ("earL", "neck"), ("earR", "neck"), ("neck", "chest"), ("neck", "back"), ("chest", "belly"),
        ("back", "belly"), ("back", "tailbase"), ("belly", "tailbase"), ("tailbase", "tailtip"), ("chest", "flegL"), ("flegL", "fpawL"),
        ("chest", "fpawR"), ("belly", "hlegL"), ("hlegL", "hpawL"), ("belly", "hpawR")])
B["bird"] = dict(noun="bird", scene="You are a small bird, perched on a branch in the morning sun.",
 feat=(1, 1, 1, 1, 1), parts=[
 P("beak", "", "beak", "which you eat and sing with", (92, 74, 0), "beak"),
 P("crown", "", "crown", "the top of your head", (80, 86, 0), "head"),
 P("eyeL", "left", "eye", "which you see with", (84, 78, 4), "eye"),
 P("throat", "", "throat", "which you sing with", (78, 64, 0), "throat"),
 P("breast", "", "breast", "the round front of your body", (66, 50, 0), "breast"),
 P("back", "", "back", "the top of your body", (48, 62, 0), "back"),
 P("wingL", "left", "wing", "which you fly with", (50, 58, 8), "wing"),
 P("wingR", "right", "wing", "which you fly with", (50, 58, -8), "wing"),
 P("wtipL", "left", "wingtip", "your longest feathers, far out to the left", (24, 52, 10), "wing"),
 P("tail", "", "tail feathers", "which you steer with", (10, 40, 0), "tail"),
 P("legL", "left", "leg", "which you stand on", (56, 24, 3), "leg"),
 P("talonL", "left", "talon", "which you perch and grip with", (58, 8, 3), "talon"),
 P("talonR", "right", "talon", "which you perch and grip with", (52, 8, -3), "talon")],
 edges=[("beak", "crown"), ("beak", "eyeL"), ("eyeL", "crown"), ("beak", "throat"), ("crown", "back"), ("throat", "breast"), ("breast", "back"),
        ("back", "wingL"), ("back", "wingR"), ("wingL", "wtipL"), ("back", "tail"), ("breast", "tail"), ("breast", "legL"), ("legL", "talonL"), ("breast", "talonR")])
B["spider"] = dict(noun="spider", scene="You are a spider, resting at the center of your web.",
 feat=(1, 1, 1, 1, 0), parts=[
 P("eyes", "", "eyes", "which you see with", (0, 22, 3), "eye"),
 P("fangs", "", "fangs", "which you bite with", (0, 28, -2), "fang"),
 P("palpL", "left", "palp", "which you touch and taste with", (5, 27, 0), "palp"),
 P("palpR", "right", "palp", "which you touch and taste with", (-5, 27, 0), "palp"),
 P("ceph", "", "head", "where your legs join", (0, 12, 0), "head"),
 P("abdomen", "", "abdomen", "the round back half of your body", (0, -14, 0), "abdomen"),
 P("spinner", "", "spinnerets", "which you spin silk with", (0, -32, 0), "spinner"),
 P("legL1", "left first", "leg", "which you walk on", (30, 40, 0), "leg"),
 P("legL2", "left second", "leg", "which you walk on", (36, 16, 0), "leg"),
 P("legL3", "left third", "leg", "which you walk on", (34, -6, 0), "leg"),
 P("legL4", "left fourth", "leg", "which you walk on", (26, -30, 0), "leg"),
 P("legR1", "right first", "leg", "which you walk on", (-30, 40, 0), "leg"),
 P("legR2", "right second", "leg", "which you walk on", (-36, 16, 0), "leg"),
 P("legR3", "right third", "leg", "which you walk on", (-34, -6, 0), "leg"),
 P("legR4", "right fourth", "leg", "which you walk on", (-26, -30, 0), "leg")],
 edges=[("eyes", "ceph"), ("fangs", "ceph"), ("eyes", "fangs"), ("palpL", "fangs"), ("palpR", "fangs"), ("ceph", "abdomen"), ("abdomen", "spinner")]
       + [("ceph", f"leg{s}{i}") for s in "LR" for i in range(1, 5)] + [(f"leg{s}{i}", f"leg{s}{i+1}") for s in "LR" for i in range(1, 4)])
B["fish"] = dict(noun="fish", scene="You are a fish, hanging still in a clear, cool river.",
 feat=(1, 1, 1, 1, 0), parts=[
 P("mouth", "", "mouth", "which you eat with", (96, 50, 0), "mouth"),
 P("eyeL", "left", "eye", "which you see with", (86, 58, 4), "eye"),
 P("gillL", "left", "gill", "which you breathe with", (76, 50, 5), "gill"),
 P("gillR", "right", "gill", "which you breathe with", (76, 50, -5), "gill"),
 P("pectL", "left", "pectoral fin", "which you steer with", (68, 40, 7), "pectoral"),
 P("pectR", "right", "pectoral fin", "which you steer with", (68, 40, -7), "pectoral"),
 P("dorsal", "", "dorsal fin", "which keeps you upright", (50, 76, 0), "dorsal"),
 P("back", "", "back", "the top of your body", (54, 64, 0), "back"),
 P("belly", "", "belly", "the pale underside of your body", (54, 34, 0), "belly"),
 P("pelvic", "", "pelvic fins", "which steady you", (50, 30, 0), "pelvic"),
 P("flankL", "left", "flank", "the scaled side of your body", (50, 50, 7), "flank"),
 P("peduncle", "", "tail base", "where your body narrows at the rear", (22, 50, 0), "tail"),
 P("tailfin", "", "tail fin", "which you swim with", (4, 50, 0), "tail")],
 edges=[("mouth", "eyeL"), ("mouth", "gillL"), ("mouth", "gillR"), ("eyeL", "gillL"), ("gillL", "pectL"), ("gillR", "pectR"), ("gillL", "flankL"),
        ("gillL", "back"), ("gillR", "back"), ("back", "dorsal"), ("gillL", "belly"), ("gillR", "belly"), ("belly", "pelvic"), ("back", "flankL"), ("belly", "flankL"),
        ("back", "peduncle"), ("belly", "peduncle"), ("flankL", "peduncle"), ("peduncle", "tailfin")])
B["snake"] = dict(noun="snake", scene="You are a snake, stretched out on a warm, flat rock.",
 feat=(1, 1, 0, 1, 0), parts=[
 P("snout", "", "snout", "the very front of your head", (100, 0, 0), "head"),
 P("tongue", "", "tongue", "which you taste the air with", (106, -1, 0), "tongue"),
 P("eyeL", "left", "eye", "which you see with", (95, 2, 2), "eye"),
 P("jaw", "", "jaw", "which you swallow with", (94, -2, 0), "jaw"),
 P("neck", "", "neck", "just behind your head", (86, 0, 0), "neck"),
 P("seg1", "front", "body", "the first third of you", (70, 0, 0), "body"),
 P("mid", "middle", "body", "the middle of you", (50, 0, 0), "body"),
 P("midbelly", "middle", "belly", "the scales under your middle, which grip the ground", (50, -2, 0), "belly"),
 P("seg3", "rear", "body", "the last third of you", (30, 0, 0), "body"),
 P("tail", "", "tail", "which narrows behind you", (14, 0, 0), "tail"),
 P("tailtip", "", "tail tip", "the very end of you", (2, 0, 0), "tail")],
 edges=[("tongue", "snout"), ("snout", "eyeL"), ("snout", "jaw"), ("eyeL", "neck"), ("jaw", "neck"), ("neck", "seg1"), ("seg1", "mid"), ("mid", "midbelly"),
        ("mid", "seg3"), ("seg3", "tail"), ("tail", "tailtip")])
B["octopus"] = dict(noun="octopus", scene="You are an octopus, resting on the sea floor.",
 feat=(1, 0, 1, 1, 0), parts=[
 P("mantle", "", "mantle", "the soft round top of your body", (0, 40, 0), "mantle"),
 P("eyeL", "left", "eye", "which you see with", (8, 22, 4), "eye"),
 P("eyeR", "right", "eye", "which you see with", (-8, 22, 4), "eye"),
 P("siphon", "", "siphon", "which you breathe and jet with", (16, 26, -4), "siphon"),
 P("beak", "", "beak", "your mouth, at your center underneath", (0, 6, 0), "beak")]
 + [P(f"arm{i}", o, "arm", "which you crawl and grasp with",
      (30 * np.sin(2 * np.pi * (i - 0.5) / 8), 6 - 6 * np.cos(2 * np.pi * (i - 0.5) / 8), 30 * np.cos(2 * np.pi * (i - 0.5) / 8)), "arm")
    for i, o in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"], 1)],
 edges=[("mantle", "eyeL"), ("mantle", "eyeR"), ("eyeL", "siphon"), ("mantle", "siphon"), ("eyeL", "beak"), ("eyeR", "beak"), ("siphon", "beak")]
       + [("beak", f"arm{i}") for i in range(1, 9)] + [(f"arm{i}", f"arm{i % 8 + 1}") for i in range(1, 9)])
B["car"] = dict(noun="car", scene="You are a small car, parked on a quiet street.",
 feat=(0, 1, 1, 0, 0), parts=[
 P("lights", "", "headlights", "which you see the road with", (96, 34, 0), "light"),
 P("bumperF", "front", "bumper", "the very front of you", (100, 20, 0), "bumper"),
 P("hood", "", "hood", "which covers your engine", (82, 44, 0), "hood"),
 P("wind", "", "windshield", "your wide front window", (66, 58, 0), "window"),
 P("roof", "", "roof", "the top of you", (46, 70, 0), "roof"),
 P("doorL", "left", "door", "your left side", (50, 40, 10), "door"),
 P("doorR", "right", "door", "your right side", (50, 40, -10), "door"),
 P("trunk", "", "trunk", "your back compartment", (12, 46, 0), "trunk"),
 P("bumperR", "rear", "bumper", "the very back of you", (0, 22, 0), "bumper"),
 P("wFL", "front left", "wheel", "which you roll on", (78, 10, 10), "wheel"),
 P("wFR", "front right", "wheel", "which you roll on", (78, 10, -10), "wheel"),
 P("wRL", "rear left", "wheel", "which you roll on", (22, 10, 10), "wheel"),
 P("wRR", "rear right", "wheel", "which you roll on", (22, 10, -10), "wheel")],
 edges=[("lights", "bumperF"), ("lights", "hood"), ("hood", "wind"), ("wind", "roof"), ("roof", "trunk"), ("trunk", "bumperR"), ("roof", "doorL"), ("roof", "doorR"),
        ("hood", "doorL"), ("hood", "doorR"), ("trunk", "doorL"), ("trunk", "doorR"), ("bumperF", "wFL"), ("bumperF", "wFR"), ("doorL", "wFL"), ("doorR", "wFR"),
        ("doorL", "wRL"), ("doorR", "wRR"), ("bumperR", "wRL"), ("bumperR", "wRR")])
B["drone"] = dict(noun="drone", scene="You are a small drone, hovering over a field.",
 feat=(0, 1, 1, 0, 0), parts=[
 P("rFL", "front left", "rotor", "spinning to lift you", (30, 0, 30), "rotor"),
 P("rFR", "front right", "rotor", "spinning to lift you", (30, 0, -30), "rotor"),
 P("rBL", "back left", "rotor", "spinning to lift you", (-30, 0, 30), "rotor"),
 P("rBR", "back right", "rotor", "spinning to lift you", (-30, 0, -30), "rotor"),
 P("aFL", "front left", "arm", "reaching out to a corner", (15, 0, 15), "arm"),
 P("aBR", "back right", "arm", "reaching out to a corner", (-15, 0, -15), "arm"),
 P("body", "", "central body", "your middle", (0, 0, 0), "body"),
 P("battery", "", "battery", "on top of your middle", (0, 6, 0), "battery"),
 P("camera", "", "camera", "which you see with, under your front", (10, -6, 0), "camera"),
 P("antenna", "", "antenna", "which picks up radio signals", (-10, 10, -6), "antenna"),
 P("legL", "left", "landing leg", "which you land on", (0, -14, 10), "leg"),
 P("legR", "right", "landing leg", "which you land on", (0, -14, -10), "leg")],
 edges=[("rFL", "aFL"), ("aFL", "body"), ("rBR", "aBR"), ("aBR", "body"), ("rFR", "body"), ("rBL", "body"), ("body", "battery"), ("body", "camera"),
        ("battery", "antenna"), ("body", "antenna"), ("body", "legL"), ("body", "legR")])
B["starfish"] = dict(noun="starfish", scene="You are a starfish, lying in a shallow tide pool.",
 feat=(0, 0, 1, 1, 0), parts=[
 P("disc", "", "central disc", "the middle of you", (0, 0, 0), "disc"),
 P("mouth", "", "mouth", "on your underside, at the center", (0, -2, 0), "mouth")]
 + [P(f"arm{i}", o, "arm", "which you crawl with", (24 * np.sin(2 * np.pi * (i - 1) / 5), 0, 24 * np.cos(2 * np.pi * (i - 1) / 5)), "arm")
    for i, o in enumerate(["first", "second", "third", "fourth", "fifth"], 1)]
 + [P("tip1", "first", "arm tip", "with a small eyespot that senses light", (44 * np.sin(0), 0, 44), "tip"),
    P("tip3", "third", "arm tip", "with a small eyespot that senses light", (44 * np.sin(4 * np.pi / 5), 0, 44 * np.cos(4 * np.pi / 5)), "tip"),
    P("feet2", "", "tube feet", "underneath you, which you grip with", (24 * np.sin(2 * np.pi / 5), -2, 24 * np.cos(2 * np.pi / 5)), "feet")],
 edges=[("disc", "mouth")] + [("disc", f"arm{i}") for i in range(1, 6)] + [(f"arm{i}", f"arm{i % 5 + 1}") for i in range(1, 6)]
       + [("arm1", "tip1"), ("arm3", "tip3"), ("arm2", "feet2")])
B["jellyfish"] = dict(noun="jellyfish", scene="You are a jellyfish, drifting in warm, open water.",
 feat=(0, 0, 0, 1, 0), parts=[
 P("top", "", "bell top", "your dome", (0, 50, 0), "bell"),
 P("rimL", "left", "bell rim", "your lower edge", (24, 36, 0), "rim"),
 P("rimR", "right", "bell rim", "your lower edge", (-24, 36, 0), "rim"),
 P("under", "", "underside", "the hollow beneath you", (0, 34, 0), "under"),
 P("mouth", "", "mouth", "at the center of your underside", (0, 28, 0), "mouth"),
 P("oral", "", "oral arms", "the frilly ruffles at your center", (0, 14, 0), "oral")]
 + [P(f"t{i}", o, "tentacle", "which you sting with", (18 * np.cos(np.pi * (i - 1) / 2), 10, 18 * np.sin(np.pi * (i - 1) / 2)), "tentacle")
    for i, o in enumerate(["first", "second", "third", "fourth"], 1)]
 + [P("ttip", "first", "tentacle tip", "trailing far below you", (18, -30, 0), "tentacle")],
 edges=[("top", "rimL"), ("top", "rimR"), ("rimL", "under"), ("rimR", "under"), ("under", "mouth"), ("mouth", "oral"), ("rimL", "t1"), ("rimR", "t3"),
        ("under", "t2"), ("under", "t4"), ("t1", "ttip"), ("rimL", "rimR")])
B["tree"] = dict(noun="tree", scene="You are an old oak tree, standing alone in a meadow.",
 feat=(0, 0, 1, 0, 0), parts=[
 P("deeproot", "deep", "root", "which drinks from far underground", (0, -40, 0), "root"),
 P("rootL", "left", "root", "which grips the soil", (-30, -14, 0), "root"),
 P("base", "", "trunk base", "where you meet the ground", (0, 0, 0), "trunk"),
 P("trunk", "", "trunk", "your thick middle", (0, 30, 0), "trunk"),
 P("bark", "", "bark", "your rough outer skin, halfway up", (10, 40, 4), "bark"),
 P("lowbr", "lowest", "branch", "which reaches out sideways", (30, 50, 0), "branch"),
 P("brL", "high left", "branch", "which reaches up and left", (-30, 80, 0), "branch"),
 P("brR", "high right", "branch", "which reaches up and right", (30, 82, 0), "branch"),
 P("crown", "", "crown", "the very top of you", (0, 100, 0), "crown"),
 P("leavesL", "left", "leaves", "high on your left, which drink the sun", (-44, 88, 0), "leaves"),
 P("buds", "", "buds", "at the very top of you", (0, 106, 0), "bud"),
 P("twig", "", "twig", "at your far right edge", (48, 54, 0), "twig")],
 edges=[("deeproot", "base"), ("rootL", "base"), ("base", "trunk"), ("trunk", "bark"), ("trunk", "lowbr"), ("trunk", "brL"), ("trunk", "brR"), ("trunk", "crown"),
        ("brL", "leavesL"), ("crown", "buds"), ("lowbr", "twig"), ("brL", "crown"), ("brR", "crown")])
B["blob"] = dict(noun="blob", scene="You are a soft, round blob of jelly with no parts at all, resting on a smooth floor.",
 feat=(0, 0, 0, 0, 0), parts=[
 P("top", "", "top", "the highest point of you", (0, 30, 0), "top"),
 P("under", "", "underside", "where you rest on the floor", (0, -30, 0), "under"),
 P("sideL", "left", "side", "", (30, 0, 0), "side"),
 P("sideR", "right", "side", "", (-30, 0, 0), "side"),
 P("front", "", "front", "", (0, 0, 30), "front"),
 P("rear", "", "back", "", (0, 0, -30), "back"),
 P("middle", "", "middle", "the inside of you", (0, 0, 0), "middle"),
 P("upleft", "upper left", "edge", "", (21, 21, 0), "edge")],
 edges="sphere")
ORDER = ["dog", "bird", "spider", "fish", "snake", "octopus", "car", "drone", "starfish", "jellyfish", "tree", "blob"]

NONCE = ["glorp", "vask", "mirl", "tesk", "brell", "quon", "drabe", "plinth", "skorv", "fendle", "wub", "throck", "nirp", "zell", "plome", "kib"]
def nonce_map(body):
    nouns = list(dict.fromkeys(p["noun"] for p in B[body]["parts"]))
    return {n: NONCE[i] for i, n in enumerate(nouns)}

def plural(n):
    if n.endswith("s"): return n
    return n + ("es" if n.endswith(("ch", "sh")) else "s")
PLUR = {"eyes", "fangs", "spinnerets", "headlights", "tail feathers", "tube feet", "pelvic fins", "oral arms", "leaves", "buds"}
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 8: "eight"}
def describe(body, nonce=False, poss="your", who=None):
    """Body description listing every part with its job. who=None: 2nd person self; else 3rd person ('the dog')."""
    b = B[body]; nm = nonce_map(body) if nonce else {}
    groups = {}
    for p in b["parts"]: groups.setdefault(p["noun"], []).append(p)
    items = []
    import re
    for n, ps in groups.items():
        w = nm.get(n, n); fn = ps[0]["fn"]
        if who is not None:
            fn = re.sub(r"\byour\b", "its", re.sub(r"\byou\b", "it", fn))
            fn = re.sub(r"\bit (\w+)", lambda m: "it " + (m.group(1) if m.group(1).endswith("s") else ("flies" if m.group(1) == "fly" else m.group(1) + ("es" if m.group(1).endswith(("sh", "ch")) else "s"))), fn)
        if len(ps) == 1 or len({p["fn"] for p in ps}) > 1:
            for p in ps:
                f = p["fn"]
                if who is not None:
                    f = re.sub(r"\byour\b", "its", re.sub(r"\byou\b", "it", f))
                    f = re.sub(r"\bit (\w+)", lambda m: "it " + (m.group(1) if m.group(1).endswith("s") else ("flies" if m.group(1) == "fly" else m.group(1) + "s")), f)
                m = (p["mod"] + " ") if p["mod"] else ""
                art = "" if p["noun"] in PLUR else ("an " if (m + w)[0] in "aeiou" else "a ")
                items.append(f"{art}{m}{w}" + (f", {f}" if f else ""))
            continue
        if False:
            m = (ps[0]["mod"] + " ") if ps[0]["mod"] else ""
            items.append(f"{'an' if (m + w)[0] in 'aeiou' else 'a'} {m}{w}" + (f", {fn}" if fn else ""))
        else:
            items.append(f"{NUM.get(len(ps), len(ps))} {plural(w)} ({', '.join(p['mod'] for p in ps)})" + (f", {fn}" if fn else ""))
    if who is None:
        return b["scene"] + " Your body has " + "; ".join(items) + ". You can feel touch on all of it."
    s = b["scene"].replace("You are a", "There is a").replace("You are an", "There is an")
    return s + f" The {b['noun']}'s body has " + "; ".join(items) + f". The {b['noun']} can feel touch on all of it."

def phrase(body, p, poss="your", nonce=False):
    w = nonce_map(body)[p["noun"]] if nonce else p["noun"]
    return f"{poss} {p['mod'] + ' ' if p['mod'] else ''}{w}"

def geodesic(body):
    b = B[body]; ks = [p["key"] for p in b["parts"]]; X = np.array([p["xyz"] for p in b["parts"]], float)
    if b["edges"] == "sphere":  # great-circle distance on the blob surface; middle = radius
        n = len(ks); D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a, c = X[i], X[j]; ra, rc = np.linalg.norm(a), np.linalg.norm(c)
                if ra < 1e-6 or rc < 1e-6: D[i, j] = abs(ra - rc)
                else: D[i, j] = 30 * np.arccos(np.clip(a @ c / ra / rc, -1, 1))
        return D
    import networkx as nx
    G = nx.Graph(); idx = {k: i for i, k in enumerate(ks)}
    for a, c in b["edges"]: G.add_edge(a, c, weight=float(np.linalg.norm(X[idx[a]] - X[idx[c]])))
    assert set(G.nodes) == set(ks), (body, set(ks) - set(G.nodes))
    d = dict(nx.all_pairs_dijkstra_path_length(G))
    return np.array([[d[a][c] for c in ks] for a in ks])

if __name__ == "__main__":
    for k in ORDER:
        print(k, len(B[k]["parts"]), geodesic(k).max().round(1))
    print(describe("dog")); print(describe("octopus", nonce=True)); print(describe("dog", who="other"))
    print(phrase("octopus", B["octopus"]["parts"][6], "the octopus's"))
