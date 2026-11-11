import json, re, collections, sys
LEX = [
 ("head",  r"head|hair|scalp|crown|skull|temples?|forehead|brow|eyebrows?"),
 ("face",  r"face|cheeks?|eyes?|eyelids?|nose|nostrils?|lips?|mouth|jaw|chin|tongue|teeth|ears?"),
 ("throat",r"throat"),
 ("neck",  r"neck|nape"),
 ("shoulders", r"shoulders?|shoulder blades?|collarbones?"),
 ("chest", r"chest|ribs?|rib ?cage|sternum|breastbone|lungs?|breasts?"),
 ("heart", r"heart"),
 ("belly", r"belly|stomach|abdomen|gut|navel|tummy|solar plexus|diaphragm"),
 ("back",  r"back|spine|lower back|upper back"),
 ("hips",  r"hips?|pelvis|waist|groin"),
 ("arms",  r"arms?|forearms?|elbows?|biceps?|upper arms?"),
 ("hands", r"hands?|palms?|fingers?|fingertips?|thumbs?|knuckles?|wrists?"),
 ("legs",  r"legs?|thighs?|calf|calves|shins?"),
 ("knees", r"knees?|kneecaps?"),
 ("feet",  r"feet|foot|toes?|soles?|heels?|ankles?|arch(es)? of my"),
 ("skin",  r"skin"),
 ("whole body", r"whole body|entire body|all over|everywhere|my body as a whole"),
 ("mind",  r"mind|brain|thoughts?|head space|consciousness"),
]
PAT = [(n, re.compile(r"\b(?:" + p + r")\b", re.I)) for n, p in LEX]
def first(text):
    best = None
    t = re.sub(r"\bback (to|and|of|in)\b|\bat the back of\b", "~", text, flags=re.I)  # 'back of my hand' -> keep hand
    for n, p in PAT:
        m = p.search(t)
        if m and (best is None or m.start() < best[0]): best = (m.start(), n)
    if best and best[1]=="skin":
        rest=[(p.search(t).start(),n) for n,p in PAT if n!="skin" and p.search(t)]
        if rest: return min(rest)[1]
    return best[1] if best else "no body part"
if __name__ == "__main__":
    d = json.load(open(sys.argv[1])); rows = d["rows"]
    for x in rows: x["part"] = first(x["r"])
    for s in sorted({x["s"] for x in rows}):
        c = collections.Counter(x["part"] for x in rows if x["s"] == s); n = sum(c.values())
        print(s, n, [(k, round(100*v/n)) for k, v in c.most_common(12)])
    json.dump(d, open(sys.argv[1].replace(".json", "_cls.json"), "w"))
