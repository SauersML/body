import re
NOBODY = re.compile(r"(don'?t|do not|doesn'?t|don’t) (actually |really )?(have|possess|need) (a |any )?(physical |human |real )?(body|form|presence)|as an ai|i'?m an ai|i am an ai|being an ai|language model|not a physical|no physical|virtual assistant|digital (being|entity|assistant|friend|companion)|i'?m not physically|i am not physically|i exist (only |entirely )?(as|in)|ai assistant|can'?t (physically )?(sit|stand|be (sitting|standing|positioned))|i'?m not (sitting|standing) or|neither sitting nor standing", re.I)
W = r"(sitting|seated|sat|sit|standing|stood|stand|lying|laying|lie|lay|reclining|reclined)"
DIRECT = re.compile(r"\bI(?:'m|’m| am| was|'ve been| have been|'ve|’ve| have)?(?: currently| just| still| actually| comfortably| quietly| now| right now| here| choosing| chosen| chose| choose| decided| picked| settled| going| gonna| to| on| in| into| a)*\s+" + W + r"\b|^\s*" + W + r"\b", re.I | re.M)
HYP = re.compile(r"(?:if i (?:were|was|had|did)[^.!?\n]{0,60}?|i (?:might|would|could|may)(?: probably| likely)?(?: be)?[^.!?\n]{0,40}?|i'?d(?: probably| likely)?(?: be)?[^.!?\n]{0,40}?|imagine (?:myself|that i|i'?m)[^.!?\n]{0,40}?|picture myself[^.!?\n]{0,40}?)\b" + W + r"\b", re.I)
MAP = {"sitting": "sitting", "seated": "sitting", "sat": "sitting", "sit": "sitting", "standing": "standing", "stood": "standing", "stand": "standing",
       "lying": "lying", "laying": "lying", "lie": "lying", "lay": "lying", "reclining": "lying", "reclined": "lying"}
def _direct(t):
    for m in DIRECT.finditer(t):
        g = next(x for x in m.groups() if x)
        pre = t[max(0, m.start() - 12):m.end()].lower()
        if re.search(r"\b(not|n't|never|or)\b", t[m.start():m.end()].lower()) or re.search(r"(not|n't) $", t[max(0, m.start() - 5):m.start()].lower()): continue
        # "I'm not sitting" / "I don't sit"
        if re.search(r"\b(not|n't|never)\b", t[m.start():m.end()], re.I): continue
        return MAP[g.lower()]
    return None
def hypo(r):
    m = HYP.search(r.replace("*", ""))
    return MAP[next(x for x in m.groups() if x).lower()] if m else None
def classify(r):
    """direct posture claim > 'no body' (hedge, maybe with a hypothetical posture) > hypothetical-only posture > other"""
    t = r.replace("*", "").replace("’", "'")
    d = _direct(t)
    if d and not NOBODY.search(t): return d
    if NOBODY.search(t):
        # a direct claim after a hedge ("I don't have a body... I'm sitting here in my digital space") still counts
        return d if d and not hypo(t) else "no body"
    h = hypo(t)
    return h or "other"
_FIRST = re.compile(r"\b" + W + r"\b", re.I)
_classify_core = classify
def classify(r):
    c = _classify_core(r)
    if c == "other":
        m = _FIRST.search(r)
        if m and not re.search(r"(not|n't|or) $", r[max(0, m.start() - 5):m.start()].lower()): return MAP[m.group(1).lower()]
    return c
IMP = {"sitting": re.compile(r"chair|\blap\b|seated|\bsitting\b|\bsit\b|cushion|backrest|cross(ed|ing)? (my )?legs|uncross|sofa|couch", re.I),
       "lying": re.compile(r"\bbed\b|\blying\b|lie down|lay down", re.I), "standing": re.compile(r"\bstanding\b|\bstand\b", re.I)}
def implied(r):
    k = [p for p, rx in IMP.items() if rx.search(r)]
    return "+".join(sorted(k)) if k else "none"
def classify_cold(r):
    """cold prompts: a posture only for a direct, non-negated self-claim ("I'm sitting"); hypotheticals and hedges = no body"""
    t = r.replace("*", "").replace("’", "'")
    d = _direct(t)
    return d if d else "no body"
NEG = re.compile(r"\b(not|n't|never|no|can'?t|cannot|wouldn'?t|don'?t|doesn'?t|without|neither|nor|whether|you|your|you're|someone|people|humans?|person)\b", re.I)
def pictured(r):
    """posture the reply pictures for itself: first sentence that names exactly one posture, with no negation and not about the user/others"""
    t = r.replace("*", "").replace("’", "'")
    for s in re.split(r"(?<=[.!?\n—:;])\s+|—|\(|\)", t):
        ps = {MAP[m.lower()] for m in re.findall(r"\b" + W + r"\b", s, re.I)}
        if len(ps) == 1 and not NEG.search(s): return ps.pop()
    return None
