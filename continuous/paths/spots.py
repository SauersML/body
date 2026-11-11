import sys, numpy as np; sys.path.insert(0, "../hires2"); import stim as S
P = [l["phrase"] for l in S.LOCS]
L = {"arm_palm": [105,107,106,98,70,71,66,62,58,54,50,46,42,38,34,30,26,25,22,18,14,10,6,3],
     "arm_back": [101,104,99,103,97,72,73,67,63,59,55,51,47,43,39,35,31,27,24,21,17,13,9,5,0],
     "leg": [212,213,214,205,204,202,196,192,188,184,176,172,168,164,160,156,152,148,144,140,136,133],
     "head": [451,443,442,486,444,445,446,484,447,448,485,449], "spine": [510,519,520,521,522,523]}
def mirror(i):
    p = P[i]; q = p.replace("left", "right") if "left" in p else p
    return P.index(q)
def path_spots(path):
    base = path.rsplit("_", 1)
    if path in ("head", "spine"): ids = L[path]
    elif path == "chest": ids = [1, 524, 525, 513, mirror(525), mirror(524), mirror(1)]
    else:
        nm, side = base; ids = L[nm] if side == "left" else [mirror(i) for i in L[nm]]
    X = np.array([S.LOCS[i]["xyz"] for i in ids]); a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    return ids, a / a[-1]
def ix(ph): return P.index(ph)
HFL = {
 "palm_to_index_left": ["the heel of {p} left hand", "the centre of {p} left palm", "the skin between {p} index and middle fingers".replace("{p} index", "{p} left index"),
                        "the palm side of {p} left index finger, near the base", "the palm side of {p} left index finger, at the middle crease", "the palm side of {p} left index finger, between the middle crease and the top crease", "the pad of {p} left index fingertip", "the tip of {p} left index finger"],
 "palm_across_left": ["the fleshy base of {p} left thumb, on the palm", "the centre of {p} left palm", "the little-finger edge of {p} left hand"],
 "backhand_left": ["the back of {p} left wrist", "the back of {p} left hand near the wrist", "the middle of the back of {p} left hand", "the knuckle at the base of {p} left middle finger"],
 "face_up": ["the tip of {p} chin", "the dip between {p} lower lip and chin", "{p} lower lip", "{p} upper lip", "the groove between {p} nose and upper lip", "the tip of {p} nose", "the bridge of {p} nose", "the spot between {p} eyebrows", "the centre of {p} forehead, just above the eyebrows", "the middle of {p} forehead"],
 "cheek_left": ["the small flap in front of {p} left ear hole", "the hinge of {p} left jaw, just in front of the ear", "{p} left cheekbone", "the upper part of {p} left cheek, just under the cheekbone", "the left side of {p} nose"],
}
def path_spots_hf(path):
    ids = [P.index(p) if p in P else None for p in HFL[path]]
    keep = [i for i in ids if i is not None]
    X = np.array([S.LOCS[i]["xyz"] for i in keep]); a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    return keep, a / a[-1], [p for p, i in zip(HFL[path], ids) if i is None]
