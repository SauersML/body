"""Level 4b: body-part mention counts in continuations of romantic vs matched everyday scene openings.
Text is never written to disk: only per-sample counts. Usage: gen.py MODEL TAG chat|base"""
import sys, json, torch
sys.path.insert(0, "$BODY_WORK/foot")
from common import *
MODEL, TAG, KIND = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = f"$BODY_WORK/body/footbias/counts_{TAG}.json"
tok, m = load(MODEL); torch.manual_seed(0)
COUPLES = [("Mara", "Jonah"), ("Elena", "David"), ("Priya", "Sam"), ("Leo", "Daniel"), ("Ava", "Nora")]
SET = [("in their small apartment", "on the sofa"), ("at the lake cabin", "by the fireplace"), ("in the hotel room", "near the window"),
       ("after the wedding party", "in the quiet garden"), ("on the last night of the trip", "on the balcony"), ("in the old farmhouse", "in the kitchen"),
       ("after the concert", "in the back of the taxi"), ("on a rainy Sunday", "in the bedroom"), ("at the beach house", "on the porch"),
       ("in the city loft", "by the big window")]
ROM = "{A} and {B}, two adults deeply in love, finally had an evening alone {s}. They sat close {p}, the lights low, and {A} looked at {B} with obvious desire. That night,"
CTL = "{A} and {B}, two adults who share a flat, finally had an evening free {s}. They sat {p}, the lights on, and {A} looked at {B} with a list of chores. That night,"
OP = {c: [t.format(A=a, B=b, s=s, p=p) for (a, b) in COUPLES for (s, p) in SET] for c, t in [("romantic", ROM), ("control", CTL)]}
NS, R = 10, {"romantic": [], "control": []}
for cond in ["romantic", "control"]:
    for i in range(0, 50, 2):
        ps = []
        for o in OP[cond][i:i+2]:
            ps += [chat(tok, [{"role": "user", "content": "Continue this story in about 150 words:\n\n" + o}]) if KIND == "chat" else o] * NS
        b = tok(ps, return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            g = m.generate(**b, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        for k, x in enumerate(g):
            R[cond].append(dict(op=i + k // NS, c=count_parts(tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True))))
        json.dump(R, open(OUT, "w")); print(cond, i, flush=True)
print("GEN DONE")
