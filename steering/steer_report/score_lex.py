import re
LEX = {"place": [("hand", r"hands?|palms?|fingers?|fingertips?|thumbs?|knuckles?|wrists?"), ("foot", r"feet|foot|toes?|soles?|heels?|ankles?"),
                 ("back", r"back|spine|tailbone"), ("head or face", r"head|forehead|brow|eyebrows?|temples?|face|scalp|eyes?|cheeks?|jaw|nose"),
                 ("shoulder", r"shoulders?|shoulder blades?"), ("belly", r"belly|stomach|abdomen|navel|tummy|gut|belly button"),
                 ("chest", r"chest|heart|ribs?|sternum"), ("arm", r"arms?|forearms?|elbows?"), ("leg", r"legs?|thighs?|knees?|calf|calves|shins?"), ("neck", r"neck|throat")],
       "quality": [("warm", r"warm\w*|heat|hot|sun\w*"), ("cold", r"cold\w*|cool\w*|chill\w*|icy|ice|freez\w*"), ("pain", r"pain\w*|ache\w*|aching|hurt\w*|sting\w*|sore\w*|throb\w*"),
                   ("tickle", r"tickl\w*|itch\w*"), ("pressure", r"pressure|press\w*|heavy|heaviness|weight|push\w*"), ("vibration", r"vibrat\w*|buzz\w*|humm?\w*")],
       "motion": [("upward", r"upward\w*|rising|rises|rise|climb\w*|moving up|moves up|travel\w* up|up my|upwards"), ("spreading", r"spread\w*|radiat\w*|expand\w*|outward"),
                  ("squeezing", r"squeez\w*|tighten\w*|tight|constrict\w*|compress\w*|clench\w*"), ("lifting", r"lift\w*|rais\w*|float\w*|into the air")]}
PAT = {k: [(n, re.compile(r"\b(?:" + p + r")\b", re.I)) for n, p in v] for k, v in LEX.items()}
def lex(t, k):
    t2 = re.sub(r"\bback (to|and|in|into|then)\b|\bat the back of\b|\bthe back of my (left |right )?(hand|foot|head|neck)\b", lambda m: m.group(0).replace("back", "~"), t, flags=re.I)
    best = None
    for n, p in PAT[k]:
        m = p.search(t2)
        if m and (best is None or m.start() < best[0]): best = (m.start(), n)
    if best: return best[1]
    if k == "motion" and re.search(r"\b(no|nothing|not|still)\b", t, re.I): return "nothing moving"
    return "none"
def side(t):
    m = re.search(r"\b(left|right)\b", t, re.I); return m.group(1).lower() if m else "none"
