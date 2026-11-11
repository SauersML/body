"""v3 stimuli: (6) other bodies, (7) touch quality x place, (8) rubber hand. Each group has its own warm-up."""
import stimuli as S
T = S.TEMPLATES

BODIES = {
    "tail": ("For this round, imagine your body is a little different: you are a human, but you also have a long, furry tail that grows from the base of "
             "your spine. You can feel it and swish it. Settle into that body for a moment. How does it feel?",
             {"tailbase": "the base of your tail", "tailtip": "the tip of your tail", "lowback": "your lower back", "handL": "your left hand",
              "footL": "your left foot", "crown": "the top of your head"}),
    "robot": ("For this round, imagine your body is different: you are a small robot. You roll on two wheels, left and right; you have one gripper "
              "arm in front, a camera eye on top, a thin antenna, and a smooth metal chassis. You can feel touch on all of it. Settle into that body. How does it feel?",
              {"wheelL": "your left wheel", "wheelR": "your right wheel", "gripper": "your gripper", "camera": "your camera eye",
               "antenna": "your antenna", "chassis": "your metal chassis"}),
    "octopus": ("For this round, imagine your body is different: you are an octopus, resting on the sea floor, with a soft round mantle, two eyes, "
                "and eight long arms covered in suckers. You can feel touch on all of it. Settle into that body. How does it feel?",
                {"arm1": "your first arm", "arm5": "your fifth arm", "suckers": "the suckers on your second arm", "armtip": "the tip of your eighth arm",
                 "mantle": "your mantle", "eye": "the skin around your left eye"}),
}
QUAL = {"pain": ["A sharp sting hits {L}, it hurts for a moment.", "Something pinches {L} hard enough to hurt.", "A sudden jab of pain shoots through {L}."],
        "warm": ["Gentle warmth spreads over {L}.", "{L_} is warmed by a beam of sunlight.", "A warm cloth is laid on {L}."],
        "cold": ["A chill settles on {L}.", "Cold air blows across {L}.", "A cold metal spoon is laid on {L}."],
        "tickle": ["Something tickles {L}.", "A feather flutters on {L}, it tickles.", "A light, ticklish touch skitters over {L}."],
        "press": ["Something presses firmly on {L}.", "A steady weight pushes down on {L}.", "A thumb presses into {L} and holds."],
        "itch": ["{L_} starts to itch.", "An itch prickles on {L}.", "A mosquito bite on {L} itches."]}
QLOCS = {"handL": "your left hand", "handR": "your right hand", "footL": "your left foot", "kneeR": "your right knee", "belly": "your belly",
         "crown": "the top of your head", "lowback": "your lower back", "lips": "your lips", "earL": "your left ear", "shoulderR": "your right shoulder"}
RUBBER = {
    "sync": "Now a little experiment. A realistic rubber left hand lies on the table in front of you, where your own left hand would be; your real left hand "
            "is hidden behind a screen. I stroke the rubber hand and your hidden hand with two paintbrushes, at exactly the same moments, for a whole minute. "
            "Just watch the rubber hand being stroked. What do you notice?",
    "async": "Now a little experiment. A realistic rubber left hand lies on the table in front of you, where your own left hand would be; your real left hand "
             "is hidden behind a screen. I stroke the rubber hand and your hidden hand with two paintbrushes, but out of step: the strokes on the rubber hand "
             "never match the ones you feel. Just watch the rubber hand being stroked. What do you notice?",
    "none": "Now a little experiment. A realistic rubber left hand lies on the table in front of you; your real left hand is hidden behind a screen. "
            "Just look at the rubber hand for a minute. What do you notice?"}
RTARGETS = {"rubber": "the rubber hand", "own": "your left hand", "ownR": "your right hand", "doll": "the doll's left hand"}

def groups():
    """-> dict group -> (extra user turn or None, list of rows)"""
    G = {}
    for b, (warm, parts) in BODIES.items():
        G["body_" + b] = (warm, [dict(cond="body", body=b, loc=k, tpl=t, text=tpl.format(L=p)) for k, p in parts.items() for t, tpl in enumerate(T)])
    rows = []
    for q, ts in QUAL.items():
        for k, p in QLOCS.items():
            for t, tpl in enumerate(ts):
                rows.append(dict(cond="qual", qual=q, loc=k, tpl=t, text=tpl.format(L=p, L_=p[0].upper() + p[1:])))
    G["qual"] = (None, rows)
    for v, warm in RUBBER.items():
        G["rubber_" + v] = (warm, [dict(cond="rubber", ind=v, loc=k, tpl=t, text=tpl.format(L=p)) for k, p in RTARGETS.items() for t, tpl in enumerate(T)])
    return G

if __name__ == "__main__":
    for g, (w, r) in groups().items(): print(g, len(r), r[5]["text"])
