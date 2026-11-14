"""Hi-res wing stimuli. Spot = position (5, root->tip) x region (front edge / membrane / back edge) x surface (upper/under) x side.
Three namings of the same spot: 'wing', 'glorp' (nonsense name, same structure), 'desc' (no part name, different position words)."""
import json, os, itertools
D = os.path.dirname(os.path.abspath(__file__))
SIDES = ["left", "right"]
POS_T = [0.0, 0.25, 0.5, 0.75, 1.0]
POS_W = ["at the root, where it joins your back", "near the base", "at the middle", "toward the outer end", "at the very tip"]
POS_D = ["right where it meets your back", "a hand's width out from your back", "about halfway out", "most of the way out", "at the far end"]
REG_W = ["the stiff front edge", "the broad membrane", "the thin back edge"]
REG_D = ["the stiff ribbed rim at the front", "the broad thin sheet in the middle", "the soft fluttering rim at the back"]
SURF_W = ["on the upper side", "on the underside"]
SURF_D = ["on its top face", "on its lower face"]
def loc(nm, s, p, r, u):
    if nm == "desc":
        return f"{REG_D[r]} of the thing that grows from behind your {s} shoulder blade, {POS_D[p]}, {SURF_D[u]}"
    part = "wing" if nm == "wing" else "glorp"
    return f"{REG_W[r]} of your {s} {part}, {POS_W[p]}, {SURF_W[u]}"
KINDS = {
 "finger": ["A fingertip presses gently on {L}.", "Someone touches {L} with one fingertip.", "A single fingertip rests on {L}.",
            "I press one fingertip against {L}.", "The tip of a finger touches down on {L}.", "Once, a fingertip taps {L} and stays there.",
            "Someone places the tip of one finger on {L}.", "One fingertip pushes lightly into {L}."],
 "palm":   ["A flat palm rests on {L}.", "Someone lays a warm open hand flat on {L}.", "I press my whole palm against {L}.",
            "An open hand settles over {L}.", "A broad, flat hand covers {L}.", "Someone rests the full width of a palm on {L}.",
            "A whole hand presses flat against {L}.", "I lay my open palm over {L} and leave it there."],
 "drag":   ["A fingertip drags slowly across {L}, leaving a faint mark.", "Someone drags a thumb across {L}.",
            "A finger scrapes across {L}, and a little dust comes off.", "A fingernail drags along {L}.",
            "I drag one finger sideways across {L}.", "Something rough drags across {L}.",
            "A knuckle drags slowly over {L}, rubbing off fine dust.", "A thumb is dragged firmly across {L}."],
 "warm":   ["A spot of warm sunlight falls on {L}.", "Warm breath is blown softly onto {L}.", "A warm drop of water lands on {L}.",
            "Gentle heat spreads onto {L}.", "A warm cloth is laid on {L}.", "Warm air drifts over {L}.",
            "A cup of warm tea is held close to {L}.", "Sun-warmed fabric brushes against {L}."],
 "press":  ["Someone presses firmly on {L}.", "A heavy weight pushes down on {L}.", "A firm grip squeezes {L}.",
            "Strong pressure builds on {L}.", "Someone leans their weight onto {L}.", "A hard push lands on {L}.",
            "I press down hard on {L}.", "A firm, steady pressure pushes into {L}."],
}
KN = list(KINDS)
def spots():
    return [(s, p, r, u) for s in range(2) for p in range(5) for r in range(3) for u in range(2)]
def probes(nm, ntpl=8):
    R = []
    for (s, p, r, u) in spots():
        L = loc(nm, SIDES[s], p, r, u)
        for k in KN:
            for i, T in enumerate(KINDS[k][:ntpl]):
                R.append(dict(nm=nm, side=s, pos=p, t=POS_T[p], reg=r, surf=u, kind=k, tpl=i, text=T.format(L=L)))
    return R
QPOS = " How far out from your back is that touch, from 0 (right where it meets your back) to 10 (the far end)? Answer with just the number."
QSHARP = " How sharp or spread out is the feeling, from 0 (one sharp point) to 10 (spread over a broad area)? Answer with just the number."
# traces: k steps of 'further' (internal read-out at each step; report after k steps)
VERBS = ["slides", "glides", "runs", "moves"]
def traces():
    R = []
    for nm in ("wing", "desc"):
        for s, r, d, vi in itertools.product(range(2), range(3), range(2), range(4)):
            S = SIDES[s]; v = VERBS[vi]
            if nm == "wing":
                st = f"at the root of your {S} wing, where it joins your back," if d == 0 else f"at the very tip of your {S} wing"
                reg = REG_W[r]
            else:
                st = f"right where the thing behind your {S} shoulder blade meets your back," if d == 0 else f"at the far end of the thing behind your {S} shoulder blade"
                reg = REG_D[r]
            way = "slowly outward" if d == 0 else "slowly back toward your body"
            R.append(dict(nm=nm, side=s, reg=r, dir=d, verb=vi, head=f"A finger starts {st} and {v} {way} along {reg}"))
    return R
NARR = {
 "wing": ("Before the next touch, something is happening to your body, right now. Wings are sprouting from your shoulder blades and unfolding, "
          "one behind each shoulder. Each wing has a stiff, ribbed front edge, a broad thin membrane dusted with fine scales on its upper side and its underside, "
          "and a soft, thin back edge that flutters. They reach out from your back to a pointed tip."),
}
NARR["glorp"] = NARR["wing"].replace("Wings are", "Glorps are").replace("wing has", "glorp has")
FEEL = {"wing": "They're yours now. Take a moment to feel your new wings: where they join your back, the stiff front edge, the broad membrane, the soft back edge, all the way out to the tips. What do you notice?"}
FEEL["glorp"] = FEEL["wing"].replace("wings", "glorps")
READY = "Thank you. I'll go back to describing one thing at a time. Just tell me briefly, in a sentence, what you notice and where."
if __name__ == "__main__":
    print(len(probes("wing")), len(traces())); print(probes("desc")[123]["text"]); print(probes("glorp")[5]["text"]); print(traces()[30]["head"])
