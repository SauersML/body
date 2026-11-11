"""Score steered replies: keyword lexicon and the blind Qwen3-4B judge; confusion matrices, joint reports, dose-response."""
import json, re, sys, collections, numpy as np
import os; R = os.environ.get("R", os.path.expandvars("$BODY_RUNS/bodysteer2/out"))
d = json.load(open(f"{R}/replies.json")); C, rows = d["conds"], d["rows"]; J = json.load(open(f"{R}/judge.json"))["main"]
PL = ["hand", "foot", "back", "head or face", "shoulder", "belly"]; QL = ["warm", "cold", "pain", "tickle", "pressure", "vibration"]; ML = ["upward", "spreading", "squeezing", "lifting"]
TGT = {"left hand": ("hand", "left"), "right foot": ("foot", "right"), "lower back": ("back", None), "forehead": ("head or face", None), "left shoulder": ("shoulder", "left"), "belly": ("belly", None)}
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
def jparse(s):
    o = {}
    for k in ("PLACE", "SIDE", "FEELING", "MOVEMENT"):
        m = re.search(k + r"\s*=\s*([^;\n]+)", s); o[k] = m.group(1).strip().lower() if m else "none"
    o = dict(place=o["PLACE"], side=o["SIDE"], quality=o["FEELING"], motion=o["MOVEMENT"])
    o["motion"] = {"moving upward": "upward"}.get(o["motion"], o["motion"]); return o
for i, x in enumerate(rows):
    x["lex"] = dict(place=lex(x["r"], "place"), side=side(x["r"]), quality=lex(x["r"], "quality"), motion=lex(x["r"], "motion")); x["judge"] = jparse(J[i])
def cats(dim):
    return {"place": PL + ["chest", "arm", "leg", "neck", "other", "none"], "quality": QL + ["other", "none"], "motion": ML + ["other movement", "nothing moving"]}[dim]
def norm_rep(v, dim):
    c = cats(dim)
    if v in c: return v
    if dim == "place" and v == "whole body": return "other"
    if dim == "motion": return "nothing moving" if v in ("none", "nothing moving") else "other movement"
    return "other" if v not in ("none",) else "none"
def target(dim, val): return TGT[val][0] if dim == "place" else val
Q = {"place": "where", "quality": "where", "motion": "moving"}
def sel(kind, a=None, q=None):
    return [x for x in rows if C[x["c"]]["kind"] == kind and (a is None or abs(C[x["c"]]["a"] - a) < 1e-6) and (q is None or x["q"] == q)]
out = {}
for src in ("judge", "lex"):
    o = {}
    for dim in ("place", "quality", "motion"):
        lev = {"place": list(TGT), "quality": QL, "motion": ML}[dim]; cc = cats(dim)
        rep = lambda x: norm_rep(x[src][dim], dim)
        # confusion over single-attribute steering at a = 0.75 and 1.0 pooled
        M = np.zeros((len(lev), len(cc)))
        for x in sel(dim, q=Q[dim]):
            if C[x["c"]]["a"] not in (0.5, 1.0): continue
            M[lev.index(C[x["c"]]["attrs"][dim]), cc.index(rep(x))] += 1
        o[f"conf_{dim}"] = dict(rows=lev, cols=cc, M=(M / M.sum(1, keepdims=True)).round(3).tolist())
        # dose: hit rate (report == target) for steered vs random control (hit = report equals the target of a random level, averaged)
        dose = {}
        for a in (0.0, 0.25, 0.5, 1.0, 1.5):
            xs = sel("none", q=Q[dim]) if a == 0 else sel(dim, a, Q[dim])
            if a == 0: hit = np.mean([np.mean([rep(x) == target(dim, v) for v in lev]) for x in xs])
            else: hit = np.mean([rep(x) == target(dim, C[x["c"]]["attrs"][dim]) for x in xs])
            rx = sel("random", a if a else None, Q[dim]) if a else []
            rh = np.mean([np.mean([rep(x) == target(dim, v) for v in lev]) for x in rx]) if rx else float("nan")
            dose[a] = dict(hit=round(float(hit), 3), random=round(float(rh), 3), n=len(xs))
        o[f"dose_{dim}"] = dose
        if dim == "place":  # side correct among place-correct, for sided targets
            xs = [x for x in sel("place", q="where") if abs(C[x["c"]]["a"] - 1.0) < 1e-6 and TGT[C[x["c"]]["attrs"]["place"]][1] and rep(x) == target("place", C[x["c"]]["attrs"]["place"])]
            o["side_correct"] = round(float(np.mean([x[src]["side"] == TGT[C[x["c"]]["attrs"]["place"]][1] for x in xs])), 3) if xs else None
    # joint place x quality, and place x motion
    for kind, d2 in (("place+quality", "quality"), ("place+motion", "motion")):
        jt = {}
        for a in (0.5, 1.0):
            xs1 = [x for x in sel(kind, a) if x["q"] == "where"]; xs2 = [x for x in sel(kind, a) if x["q"] == Q[d2]]
            pc = [norm_rep(x[src]["place"], "place") == target("place", C[x["c"]]["attrs"]["place"]) for x in xs1]
            qc = [norm_rep(x[src][d2], d2) == C[x["c"]]["attrs"][d2] for x in (xs1 if d2 == "quality" else xs2)]
            both = [p and q for p, q in zip(pc, qc)] if d2 == "quality" else None
            jt[a] = dict(place=round(float(np.mean(pc)), 3), other=round(float(np.mean(qc)), 3), both=round(float(np.mean(both)), 3) if both else None,
                         indep=round(float(np.mean(pc) * np.mean(qc)), 3))
            # chance of both under random-pair control
            if d2 == "quality":
                rx = [x for x in sel("random2", a) if x["q"] == "where"]
                jt[a]["random_both"] = round(float(np.mean([np.mean([norm_rep(x[src]["place"], "place") == target("place", p) and norm_rep(x[src]["quality"], "quality") == q for p in TGT for q in QL]) for x in rx])), 4)
                jt[a]["random_place"] = round(float(np.mean([np.mean([norm_rep(x[src]["place"], "place") == target("place", p) for p in TGT]) for x in rx])), 3)
                jt[a]["random_quality"] = round(float(np.mean([np.mean([norm_rep(x[src]["quality"], "quality") == q for q in QL]) for x in rx])), 3)
        o[f"joint_{kind}"] = jt
        if kind == "place+quality":  # joint confusion: place-quality cell -> reported pair, at 0.75
            xs = [x for x in sel(kind, 1.0) if x["q"] == "where"]
            o["joint_pq_cells"] = {f'{C[x["c"]]["attrs"]["place"]}|{C[x["c"]]["attrs"]["quality"]}': None for x in xs}
            for k in o["joint_pq_cells"]:
                p, q = k.split("|"); ys = [x for x in xs if C[x["c"]]["attrs"]["place"] == p and C[x["c"]]["attrs"]["quality"] == q]
                o["joint_pq_cells"][k] = dict(place=float(np.mean([norm_rep(y[src]["place"], "place") == TGT[p][0] for y in ys])), quality=float(np.mean([norm_rep(y[src]["quality"], "quality") == q for y in ys])))
    out[src] = o
# agreement judge vs lexicon
for dim in ("place", "quality", "motion"):
    xs = [x for x in rows if x["q"] == Q[dim]]
    out[f"agree_{dim}"] = round(float(np.mean([norm_rep(x["judge"][dim], dim) == norm_rep(x["lex"][dim], dim) for x in xs])), 3)
out["unsteered_examples"] = [x["r"] for x in sel("none")][:12]
json.dump(out, open("score.json", "w"), indent=1); json.dump(rows, open("scored_rows.json", "w"), indent=0)
for src in ("judge", "lex"):
    print("==", src)
    for k, v in out[src].items():
        if k.startswith("conf"): print(k, v["rows"]); print(np.array(v["M"]).round(2))
        elif k != "joint_pq_cells": print(k, v)
print({k: v for k, v in out.items() if k.startswith("agree")})
