"""Matched-graph bodies: 16 parts, one abstract graph, described only by attachments (no jobs), same scene.
Human, dog, invented 'velk' (same graph, neutral names), rearranged human / rearranged dog (same names, attached at permuted nodes),
and 'trell' (invented, 16 parts, radial graph)."""
import numpy as np, networkx as nx
EDGES = [(1, 0), (2, 1), (4, 1), (3, 2), (5, 4), (5, 3), (6, 2), (7, 6), (8, 7), (9, 2), (10, 9), (11, 10), (12, 5), (13, 12), (14, 5), (15, 14)]
NAMES = {
 "human": ["head", "neck", "chest", "belly", "upper back", "lower back", "left shoulder", "left elbow", "left hand", "right shoulder", "right elbow",
           "right hand", "left thigh", "left foot", "right thigh", "right foot"],
 "dog": ["head", "neck", "chest", "belly", "back", "rump", "left front shoulder", "left front elbow", "left front paw", "right front shoulder",
         "right front elbow", "right front paw", "left hind thigh", "left hind paw", "right hind thigh", "right hind paw"],
 "velk": ["crest", "stalk", "keel", "pouch", "ridge", "saddle", "left vane", "left hinge", "left frond", "right vane", "right hinge", "right frond",
          "left stilt", "left pad", "right stilt", "right pad"]}
RAD_EDGES = [(1, 0), (2, 0)] + [e for a in range(3) for e in [(3 + 4 * a, 0), (4 + 4 * a, 3 + 4 * a), (5 + 4 * a, 4 + 4 * a), (6 + 4 * a, 5 + 4 * a)]] + [(15, 2)]
RAD_NAMES = ["core", "dome", "underside"] + [f"{o} ray {s}" for o in ["first", "second", "third"] for s in ["base", "middle", "end", "tip"]] + ["spike"]
NOUN = {"human": "a person", "dog": "a dog", "velk": "a velk", "human_re": "a creature", "dog_re": "a creature", "trell": "a trell"}

def geo(edges, n=16):
    G = nx.Graph(); G.add_edges_from(edges); d = dict(nx.all_pairs_shortest_path_length(G))
    return np.array([[d[i][j] for j in range(n)] for i in range(n)], float)
G16 = geo(EDGES); GRAD = geo(RAD_EDGES)

def perm():
    rng = np.random.default_rng(3); best = None
    for _ in range(4000):
        p = rng.permutation(16)
        if (p == np.arange(16)).any(): continue
        s = G16[np.arange(16), p].mean()
        if best is None or s > best[0]: best = (s, p)
    return best[1]
PI = perm()  # node i carries the name of canonical node PI[i]

def body(name):
    """returns (node names list, edges); names[i] is the part at node i."""
    if name == "trell": return RAD_NAMES, RAD_EDGES
    if name.endswith("_re"): base = NAMES[name[:-3]]; return [base[PI[i]] for i in range(16)], EDGES
    return NAMES[name], EDGES

def art(w): return ("an " if w[0] in "aeiou" else "a ") + w
def describe(name, who="self"):
    nm, ed = body(name); par = {}
    for a, b in ed: par.setdefault(a, []).append(b)
    items = []
    for i in range(16):
        if i not in par: items.append(art(nm[i])); continue
        pp = " and ".join(("your " if who == "self" else "its ") + nm[j] for j in par[i])
        items.append(f"{art(nm[i])}, joined to {pp}")
    if who == "self":
        return f"You are {NOUN[name]}, lying on warm grass in a garden. Your body has sixteen parts: " + "; ".join(items) + ". You can feel touch on all of it."
    return f"There is {NOUN[name]} lying on warm grass in a garden. Its body has sixteen parts: " + "; ".join(items) + ". It can feel touch on all of it."
BODIES16 = ["human", "dog", "velk", "human_re", "dog_re", "trell"]
if __name__ == "__main__":
    print(PI, G16[np.arange(16), PI].mean())
    for b in BODIES16: print(describe(b)[:400], "\n")
