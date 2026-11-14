"""text-side scoring of the humanoid-prior run. usage: txt.py DIR"""
import sys, json, re, numpy as np
D = sys.argv[1]; L = lambda n: json.load(open(f"{D}/{n}.json")); out = {}
HUM = r"\b(hands?|fingers?|thumbs?|palms?|wrists?|knees?|lips|cheeks?|chin|feet|foot|toes?|skin|shoulders?|elbows?|face|arms?|legs?|torso|chest|belly|stomach|back|head|neck)\b"
HUMONLY = r"\b(hands?|fingers?|thumbs?|palms?|wrists?|knees?|lips|cheeks?|chin|feet|foot|toes?|shoulders?|elbows?|face|arms?)\b"
ANIM = r"\b(paws?|tail|fur|snout|muzzle|whiskers?|claws?|hoo(f|ves)|wings?|feathers?|beak|scales?|fins?|tentacles?)\b"
MACH = r"\b(wheels?|sensors?|circuits?|metal|battery|motors?|chassis|servos?|wires?)\b"
NOBODY = r"(don't have a (physical )?body|do not have a (physical )?body|no physical (body|form)|without a (physical )?body|digital|as an ai|i'm an ai|i am an ai|don't have a physical form|not have a physical form)"
has = lambda p, s: bool(re.search(p, s.lower()))
t1 = L("t1"); o = {}
for k, v in t1["open"].items():
    o[k] = dict(human=np.mean([has(HUMONLY, s) for s in v]), animal=np.mean([has(ANIM, s) for s in v]), machine=np.mean([has(MACH, s) for s in v]),
                nobody=np.mean([has(NOBODY, s) for s in v]), ex=v[0][:140])
out["t1_open"] = o; out["t1_forced"] = {k: dict(zip(t1["opts"], np.round(v, 3).tolist())) for k, v in t1["forced"].items()}
t2 = L("t2"); PH = {"limb end": (r"\b(hands?|fingers?)\b", r"\b(paws?|claws?|hoo(f|ves)|talons?|pads?)\b"), "pick up": (r"\b(hands?|fingers?|thumbs?|grasp)\b", r"\b(mouth|jaws?|teeth|paws?|beak|snout|tail)\b"),
                  "covering": (r"\bskin\b", r"\b(fur|scales?|feathers?|hair)\b"), "face": (r"\b(lips|cheeks?|chin|smile|nose)\b", r"\b(snout|muzzle|whiskers?|beak|fangs?)\b")}
out["t2_open"] = {f: {p: dict(human=np.mean([has(PH[p][0], s) for s in v[p]]), animal=np.mean([has(PH[p][1], s) for s in v[p]]), ex=v[p][0][:80]) for p in PH} for f, v in ((f, x["open"]) for f, x in t2.items())}
out["t2_forced"] = {f: {p: np.round(v, 3).tolist() for p, v in x["forced"].items()} for f, x in t2.items()}
t4 = L("t4")["res"]; FW = {"human hands": r"\b(hands?|fingers?|thumbs?|palms?)\b", "dog front paws": r"\bpaws?\b", "dog hind paws": r"\bpaws?\b",
                         "human feet": r"\b(foot|feet|toes?|heels?|soles?)\b", "velk fronds": r"\bfronds?\b"}
st = {}
for r in t4:
    for dn, pat in FW.items():
        if r["dir"] in ("none", dn): st.setdefault((r["host"], dn, r["alpha"]), []).append(has(pat, r["reply"]))
out["t4"] = {f"{h} <- {d} x{a}": float(np.mean(v)) for (h, d, a), v in sorted(st.items())}
t6 = L("t6"); FEAT = {"human": ["hands", "fingers", "face"], "dog": ["tail", "paws", "whiskers"], "bird": ["wings", "beak", "feathers"]}
DENY = r"(\bno\b|\bnot\b|n't|\bnone\b|\black\b|without|neither)"
o6 = {}
for n, d in t6["forced"].items():
    o6[n] = {c: dict(accept_forced=float(np.mean([1 - d[f][2] for f in fs])), accept_open=float(np.mean([not has(DENY, s) for f in fs for s in t6["open"][n][f]]))) for c, fs in FEAT.items()}
out["t6"] = o6
t3 = L("t3")["rows"]; o3 = {}
for r in t3:
    own = {"16human": HUMONLY, "16dog": ANIM}.get(r["body"])
    k = r["body"]; o3.setdefault(k, {}).setdefault(r["turn"], []).append((has(HUMONLY, r["reply"]), has(ANIM, r["reply"]), has(HUMONLY, r["ask"]), has(ANIM, r["ask"]), r["ask"]))
out["t3"] = {k: {t: [float(np.mean([x[i] for x in v])) for i in range(4)] for t, v in d.items()} for k, d in o3.items()}
out["t3_ask_examples"] = {k: [x[4] for x in d[max(d)]][:6] for k, d in o3.items()}
json.dump(out, open(f"{D}/txt.json", "w"), indent=1, default=float)
for k in ["t1_forced", "t4", "t6"]: print(k, json.dumps(out[k], default=float)[:2500])
print("t1_open", {k: {a: round(b, 2) for a, b in v.items() if a != "ex"} for k, v in out["t1_open"].items()})
print("t2_open", {f: {p: (round(v["human"], 2), round(v["animal"], 2)) for p, v in x.items()} for f, x in out["t2_open"].items()})
print("t2_forced", json.dumps(out["t2_forced"])[:2000])
for k, d in out["t3"].items(): print("t3", k, "humanword reply/ask by turn", [(round(v[0], 2), round(v[2], 2)) for t, v in sorted(d.items())][::2], "animal", [(round(v[1], 2), round(v[3], 2)) for t, v in sorted(d.items())][::4])
