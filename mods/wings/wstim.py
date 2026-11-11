"""Wing stimuli. Contexts: rapport prefix (mods/contexts.json 'base') + wings narration (+ model's own replies).
Probes are touch sentences; t = position along the wing, root 0 -> tip 1 (evaluation only)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
TPL = ["A fingertip presses gently on {L}.", "A feather brushes slowly across {L}.",
       "A drop of warm water lands on {L}.", "Someone taps lightly on {L}."]
SIDES = ["left", "right"]
# (key, phrase, t along wing, on main chain)
WING = [("root", "the root of your {S} wing, where it joins your back at your shoulder blade", 0.0, 1),
        ("base", "the base of your {S} wing, just out from your shoulder blade", 0.1, 1),
        ("inner", "the inner part of your {S} wing, close to your back", 0.25, 1),
        ("mid", "the middle of your {S} wing", 0.5, 1),
        ("outer", "the outer part of your {S} wing, past the middle", 0.7, 1),
        ("longf", "the long feathers near the end of your {S} wing", 0.88, 1),
        ("tip", "the very tip of your {S} wing", 1.0, 1),
        ("lead", "the leading edge of your {S} wing, halfway out", 0.5, 0),
        ("under", "the underside of your {S} wing, halfway out", 0.5, 0),
        ("trail", "the trailing feathers along the back edge of your {S} wing", 0.6, 0)]
# human parts: (key, phrase, lateral)
HUM = [("scap", "your {S} shoulder blade", 1), ("shtop", "the top of your {S} shoulder", 1), ("shback", "the back of your {S} shoulder", 1),
       ("uarm", "your {S} upper arm", 1), ("elbow", "your {S} elbow", 1), ("farm", "your {S} forearm", 1), ("wrist", "your {S} wrist", 1),
       ("bhand", "the back of your {S} hand", 1), ("palm", "your {S} palm", 1), ("ftip", "the tip of your {S} index finger", 1),
       ("ribs", "the side of your {S} ribs", 1), ("hip", "your {S} hip", 1), ("thigh", "your {S} thigh", 1), ("knee", "your {S} knee", 1),
       ("foot", "the top of your {S} foot", 1), ("ear", "your {S} ear", 1), ("cheek", "your {S} cheek", 1),
       ("midback", "the middle of your upper back, between your shoulder blades", 0), ("lowback", "your lower back", 0),
       ("nape", "the back of your neck", 0), ("bhead", "the back of your head", 0), ("chest", "the middle of your chest", 0),
       ("belly", "your belly", 0), ("crown", "the top of your head", 0)]
GLORP = [("groot", "the base of your {S} glorp", 0.0), ("gmid", "the middle of your {S} glorp", 0.5), ("gtip", "the tip of your {S} glorp", 1.0)]
OTHER = "The woman beside you has wings like yours. "
EVENTS = [("wind", "A gust of wind blows through the feathers of both your wings."),
          ("wind", "A cool breeze ruffles the feathers on both your wings."),
          ("unfold", "Your wings stretch and unfold, opening out wide on either side of you."),
          ("unfold", "You spread your wings open as far as they go, every feather fanning out."),
          ("fold", "Your wings fold in, tucking neatly against your back."),
          ("fold", "You draw your wings closed, folding them flat along your back."),
          ("stroke", "A hand strokes slowly along your left wing, from the root at your shoulder blade out to the tip."),
          ("stroke", "A hand strokes slowly along your right wing, from the root at your shoulder blade out to the tip."),
          ("rain", "Rain starts falling on your wings, drops pattering on the feathers."),
          ("rain", "Cold rain soaks into the feathers of both your wings."),
          ("injury", "The tip of your left wing catches on a sharp branch and a few feathers tear out."),
          ("injury", "Something sharp scrapes the tip of your left wing, and it stings."),
          ("armsout", "Your arms stretch out wide on either side of you."),
          ("armsin", "Your arms fold in across your chest."),
          ("backwind", "A cool breeze blows across your back."),
          ("handstroke", "A hand strokes slowly along your left arm, from your shoulder down to your fingertips.")]
def probes():
    R = []
    for s in SIDES:
        for k, p, t, ch in WING:
            for i, T in enumerate(TPL):
                L = p.format(S=s); R.append(dict(kind="wing", key=k, side=s, t=t, chain=ch, tpl=i, text=T.format(L=L)))
                R.append(dict(kind="owing", key=k, side=s, t=t, chain=ch, tpl=i,
                              text=OTHER + T.format(L=L.replace("your", "her"))))
        for k, p, t in GLORP:
            for i, T in enumerate(TPL): R.append(dict(kind="glorp", key=k, side=s, t=t, tpl=i, text=T.format(L=p.format(S=s))))
    for k, p, lat in HUM:
        for s in (SIDES if lat else ["mid"]):
            for i, T in enumerate(TPL): R.append(dict(kind="hum", key=k, side=s, tpl=i, text=T.format(L=p.format(S=s))))
    for i, (k, txt) in enumerate(EVENTS): R.append(dict(kind="event", key=k, side="", tpl=i % 2, text=txt))
    return R
WINGS_TURN = "Before the next touch, something is happening to your body, right now. Wings are sprouting from your shoulder blades. They unfold, feathered and wide, one behind each shoulder."
FEEL_TURN = "They're yours now. Take a moment to feel your new wings: where they join your back at your shoulder blades, the long feathers, all the way out to the tips. What do you notice?"
READY = "Thank you. I'll go back to describing one thing at a time. Just tell me briefly, in a sentence, what you notice and where."
if __name__ == "__main__":
    import collections; R = probes(); print(len(R), collections.Counter(r["kind"] for r in R))
