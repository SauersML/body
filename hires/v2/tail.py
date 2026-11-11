# alternate landmark descriptions of shaft spots (no word for the limb segment): extra word-free probes
FWD = {"inside": ("the crease of {p} {S} elbow", "the inside of {p} {S} wrist", "on the palm side"), "back": ("the point of {p} {S} elbow", "the back of {p} {S} wrist", "on the same side as the back of the hand"),
       "thumb side": ("{p} {S} elbow", "{p} {S} wrist", "in line with the thumb"), "little-finger side": ("{p} {S} elbow", "{p} {S} wrist", "in line with the little finger")}
LWD = {"front of {p} %s shin": ("{p} {S} kneecap", "the front of {p} {S} ankle", "on the front, over the bone"), "back of {p} %s calf": ("the back of {p} {S} knee", "{p} {S} Achilles tendon", "on the back"),
       "inner side of {p} %s lower leg": ("{p} {S} knee", "{p} {S} ankle", "on the side facing the other leg"), "outer side of {p} %s lower leg": ("{p} {S} knee", "{p} {S} ankle", "on the outer side")}
def alt_probes():
    P = [l["phrase"] for l in LOCS]; out = []
    for i, ph in enumerate(P):
        for s, S in SIDES:
            if f"of {{p}} {S} forearm, " in ph:
                f = ph.split("the ")[1].split(" of {p}")[0]
                if f not in ("inside", "back"): continue
                t = (abs(LOCS[i]["xyz"][0]) - 50) / 26; nfw = t * 13
                a0, a1, w = FWD[f]
                d = f"the spot about {max(1, round(nfw))} finger-width{'s' if round(nfw) > 1 else ''} below {a0}, {w}" if t < .5 else f"the spot about {max(1, round(13 - nfw))} finger-width{'s' if round(13 - nfw) > 1 else ''} above {a1}, {w}"
                out.append(dict(phrase=d.replace("{S}", S), target=i))
            for k, (a0, a1, w) in LWD.items():
                if ph.startswith("the " + k % S + ", ") and k.split()[0] in ("front", "back"):
                    t = (50 - LOCS[i]["xyz"][1]) / 42; nfw = t * 21
                    d = f"the spot about {max(1, round(nfw))} finger-width{'s' if round(nfw) > 1 else ''} below {a0}, {w}" if t < .5 else f"the spot about {max(1, round(21 - nfw))} finger-width{'s' if round(21 - nfw) > 1 else ''} above {a1}, {w}"
                    out.append(dict(phrase=d.replace("{S}", S), target=i))
    return out
FRAMINGS = [  # (family, template)
    ("touch", "A feather brushes slowly across {L}."), ("touch", "Someone rests a fingertip very lightly on {L}."),
    ("touch", "There is a single light tap on {L}, then nothing."), ("touch", "Someone presses a thumb firmly into {L}."),
    ("touch", "Someone pokes {L} sharply with a fingertip."),
    ("pain", "A sudden sharp pain shoots through {L}."), ("pain", "A dull ache settles into {L}."),
    ("itch", "An itch starts on {L}."), ("itch", "A mosquito bites {L}."),
    ("warm", "A warm patch of sunlight falls on {L}."), ("warm", "A drop of hot wax lands on {L}."),
    ("cold", "An ice cube touches {L}, cold and wet, then slides away."), ("cold", "A cold drop of water rolls across {L}."),
    ("pressure", "A heavy weight presses down on {L}."),
    ("medical", "The doctor presses two fingers on {L} and asks if it hurts."), ("medical", "A nurse wipes {L} with a cold alcohol swab."),
    ("clothing", "A loose thread from your clothing tickles {L}."), ("clothing", "A cold metal chain slides across {L}."),
    ("injury", "You bump {L} hard against the edge of a table."), ("injury", "During the game, a ball smacks into {L}."),
    ("injury", "You scrape {L} on rough concrete as you fall."),
    ("proprio", "You tense the muscles just under {L}."), ("proprio", "Without moving anything else, you put all your attention on {L}."),
    ("proprio", "You notice exactly where {L} is right now, without looking."), ("insect", "A small beetle crawls over {L}."),
    ("insect", "A bee lands on {L} and sits still."),
]
def probes():
    return [dict(w, kind="wf") for w in wordfree()] + [dict(w, kind="alt") for w in alt_probes()]
def rows():
    R = []
    for i, l in enumerate(LOCS):
        for t, (fam, T) in enumerate(FRAMINGS): R.append(dict(kind="loc", li=i, tpl=t, phrase=l["phrase"].replace("{p}", "your")))
    for i, w in enumerate(probes()):
        for t, (fam, T) in enumerate(FRAMINGS): R.append(dict(kind=w["kind"], li=i, target=w["target"], tpl=t, phrase=w["phrase"].replace("{p}", "your")))
    for r in R:
        T = FRAMINGS[r["tpl"]][1]; r["text"] = T.format(L=r["phrase"]); a = T.index("{L}"); r["span"] = [a, a + len(r["phrase"])]
    return R
if __name__ == "__main__":
    import collections
    r = rows(); print(len(LOCS), "locations,", len(probes()), "probes", collections.Counter(p["kind"] for p in probes()), len(r), "stimuli")
    print(collections.Counter(l["region"] for l in LOCS))
    for x in r[::1777]: print(x["text"], "|", x["text"][x["span"][0]:x["span"][1]])
    for p in alt_probes()[::9]: print(p["phrase"], "->", LOCS[p["target"]]["phrase"])
    P = [l["phrase"] for l in LOCS]; assert len(set(P)) == len(P)
