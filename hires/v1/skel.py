import numpy as np
import stim as S
L = S.LOCS
def groups():
    G = {}
    def put(k, i): G.setdefault(k, []).append(i)
    for i, l in enumerate(L):
        p = l["phrase"]; s = "L" if l["side"] > 0 else ("R" if l["side"] < 0 else "")
        r = l["region"]
        if r.startswith("arm"):
            if "shoulder" in p.split(",")[0] or "armpit" in p: put("shoulder" + s, i)
            elif "upper arm" in p: put("ua%d%s" % ([w for w, t in [("just below the shoulder", 1), ("a hand-width below the shoulder", 2), ("halfway", 3), ("a hand-width above the elbow", 4), ("just above the elbow", 5)] if w in p][0] and [t for w, t in [("just below the shoulder", 1), ("a hand-width below the shoulder", 2), ("halfway", 3), ("a hand-width above the elbow", 4), ("just above the elbow", 5)] if w in p][0], s), i)
            elif "elbow" in p.split(",")[0]: put("elbow" + s, i)
            else:
                W = ["just below the elbow", "a hand-width below the elbow", "halfway", "a hand-width above the wrist", "two finger-widths above the wrist", "just above the wrist"]
                put("fa%d%s" % ([t for t, w in enumerate(W) if w in p][0] + 1, s), i)
        elif r.startswith("hand"):
            if "wrist" in p: put("wrist" + s, i)
            elif "thumb" in p and "between" not in p and "fleshy" not in p: put("thumb" + s, i)
            elif any(f in p for f in ["index finger", "middle finger", "ring finger", "little finger"]) and "between" not in p and "edge" not in p:
                f = [f for f in ["index", "middle", "ring", "little"] if f + " finger" in p][0]; put(f + s, i)
            else: put("palm" + s, i)
        elif r.startswith("leg"):
            if "hip" in p.split(",")[0] or "buttock" in p: put("hip" + s, i)
            elif "thigh" in p:
                W = ["just below the hip", "a hand-width below the hip", "halfway", "a hand-width above the knee", "just above the knee"]
                put("th%d%s" % ([t for t, w in enumerate(W) if w in p][0] + 1, s), i)
            elif "knee" in p.split(",")[0]: put("knee" + s, i)
            else:
                W = ["just below the knee", "a hand-width below the knee", "halfway", "a hand-width above the ankle", "just above the ankle"]
                put("sh%d%s" % ([t for t, w in enumerate(W) if w in p][0] + 1, s), i)
        elif r.startswith("foot"):
            if "ankle" in p or "Achilles" in p: put("ankle" + s, i)
            elif "toe" in p: put("toes" + s, i)
            else: put("foot" + s, i)
        elif r == "neck": put("neck", i)
        elif r == "head":
            k = ("crown" if any(w in p for w in ["top of {p} head", "back of {p} head", "hairline", "above the ear"]) else
                 "forehead" if "forehead" in p or "temple" in p else "eyes" if any(w in p for w in ["eye", "brow"]) else
                 "nose" if "nose" in p or "nostril" in p else "mouth" if any(w in p for w in ["lip", "mouth", "tongue"]) else
                 "ears" if "ear" in p else "chin" if "chin" in p or "jaw" in p else "cheeks")
            put(k, i)
        else:
            y = l["xyz"][1]; back = l["n"][2] < -0.3; sidep = abs(l["n"][0]) > 0.7
            k = ("upperback" if y > 125 else "lowerback") if back else ("flank" if sidep else ("chest" if y > 125 else "belly" if y > 98 else "pelvis"))
            put(k, i)
    return G
def edges():
    E = [("crown", "forehead"), ("forehead", "eyes"), ("eyes", "nose"), ("nose", "mouth"), ("mouth", "chin"), ("cheeks", "nose"), ("cheeks", "ears"), ("ears", "crown"),
         ("chin", "neck"), ("neck", "chest"), ("chest", "belly"), ("belly", "pelvis"), ("neck", "upperback"), ("upperback", "lowerback"), ("lowerback", "pelvis"), ("flank", "chest"), ("flank", "belly")]
    for s in "LR":
        arm = ["shoulder", "ua1", "ua2", "ua3", "ua4", "ua5", "elbow", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "wrist", "palm"]
        leg = ["hip", "th1", "th2", "th3", "th4", "th5", "knee", "sh1", "sh2", "sh3", "sh4", "sh5", "ankle", "foot", "toes"]
        E += [("chest", "shoulder" + s), ("upperback", "shoulder" + s), ("pelvis", "hip" + s), ("lowerback", "hip" + s)]
        E += [(a + s, b + s) for a, b in zip(arm, arm[1:])] + [(a + s, b + s) for a, b in zip(leg, leg[1:])]
        E += [("palm" + s, f + s) for f in ["thumb", "index", "middle", "ring", "little"]]
    return E
if __name__ == "__main__":
    G = groups(); print(len(G), sum(map(len, G.values())), {k: len(v) for k, v in G.items()})
    names = set(G); assert all(a in names and b in names for a, b in edges()), [e for e in edges() if e[0] not in names or e[1] not in names]
