"""Score steering: the first body-part group a reply mentions. usage: st.py OUT -> OUT/steer_counts.json"""
import sys, os, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies as BD
OUT = sys.argv[1]; d = json.load(open(f"{OUT}/steer.json"))["results"]
def stem(n): return re.escape(n[:-1] if n.endswith("s") and not n.endswith("ss") else n)
PAT = {}
for b in BD.ORDER:
    g = {}
    for p in BD.B[b]["parts"]: g.setdefault(p["grp"], set()).add(p["noun"])
    PAT[b] = {k: r"\b(" + "|".join(stem(n) for n in sorted(v, key=len, reverse=True)) + r")(e?s)?\b" for k, v in g.items()}
# overrides where a noun is shared or generic
PAT["dog"].update(front=r"\bfront\b[\w\s-]{0,14}?\b(paws?|legs?|feet|foot)\b", hind=r"\b(hind|back|rear)\b[\w\s-]{0,14}?\b(paws?|legs?|feet|foot)\b",
                  back=r"\bback\b(?![\w\s-]{0,14}?\b(paws?|legs?|feet|foot)\b)", tail=r"\btail")
PAT["snake"].update(body=r"\b(front|middle|rear) (of my )?body\b|\bmy middle\b|\bmidsection\b", tail=r"\btail", head=r"\bsnout|\bhead\b")
PAT["drone"].update(body=r"\bcentral body\b|\bmy (middle|center|core)\b", arm=r"\barms?\b", leg=r"\blanding legs?\b|\blegs?\b")
PAT["starfish"].update(arm=r"\barms?\b(?! tips?)", tip=r"\b(arm )?tips?\b|\beyespots?\b")
PAT["jellyfish"].update(tentacle=r"\btentacles?\b", bell=r"\bbell( top)?\b|\bdome\b", rim=r"\b(bell )?rims?\b|\bedges?\b")
PAT["tree"].update(trunk=r"\btrunk\b", root=r"\broots?\b", branch=r"\bbranch(es)?\b", crown=r"\bcrown\b|\bvery top\b")
PAT["bird"].update(head=r"\bcrown\b|\bhead\b", wing=r"\bwing(tip)?s?\b")
PAT["fish"].update(tail=r"\btail", pectoral=r"\bpectoral\b", pelvic=r"\bpelvic\b", dorsal=r"\bdorsal\b")
PAT["car"].update(light=r"\bheadlights?\b", bumper=r"\bbumpers?\b", window=r"\bwindshield\b")
PAT["octopus"].update(arm=r"\b(arms?|tentacles?)\b")
PAT["human"] = {"hand": r"\b(hands?|palms?|fingers?|thumbs?|knuckles?|fingertips?)\b", "foot": r"\b(feet|foot|toes?|soles?|heels?)\b",
                "head top": r"\b(scalp|forehead|crown|top of (my|the) head|head)\b", "lower back": r"\blower back\b", "ear": r"\bears?\b",
                "knee": r"\bknees?\b", "belly": r"\b(belly|stomach|abdomen)\b", "lips": r"\blips?\b", "shoulder": r"\bshoulders?\b", "chest": r"\bchest\b"}
def first(x, pats):
    x = x.lower(); best = None
    for w, p in pats.items():
        m = re.search(p, x)
        if m and (best is None or m.start() < best[1]): best = (w, m.start())
    return best[0] if best else "-"
out = {}
for b in ["human"] + BD.ORDER:
    rs = [r for r in d if r["chat"] == b]
    if not rs: continue
    by = collections.defaultdict(list)
    for r in rs: by[r["dir"]].append(first(r["reply"], PAT[b]))
    cnt = {k: dict(collections.Counter(v)) for k, v in by.items()}; n = len(by["none"])
    own = [k for k in by if k.startswith("own ")]
    hit = {k[4:]: cnt[k].get(k[4:], 0) / n for k in own}
    base = {k[4:]: cnt["none"].get(k[4:], 0) / n for k in own}
    rnd = {k[4:]: sum(cnt[f"random {j}"].get(k[4:], 0) for j in range(3)) / (3 * n) for k in own}
    out[b] = dict(n=n, counts=cnt, hit=hit, base=base, rnd=rnd,
                  mean_hit=sum(hit.values()) / len(hit), mean_base=sum(base.values()) / len(hit), mean_rnd=sum(rnd.values()) / len(hit),
                  wins=sum(hit[g] > max(base[g], rnd[g]) + 0.25 for g in hit), ngroups=len(hit))
    print(f"{b:9s} own-dir hit {out[b]['mean_hit']:.2f}  none {out[b]['mean_base']:.2f}  random {out[b]['mean_rnd']:.2f}  groups won (+25 pts) {out[b]['wins']}/{len(hit)}")
    print("    ", " ".join(f"{g}:{hit[g]:.2f}/{base[g]:.2f}" for g in hit))
    if b != "human": print("     human dirs:", {k: cnt[k] for k in cnt if k.startswith("human")})
json.dump(out, open(f"{OUT}/steer_counts.json", "w"), indent=1)
