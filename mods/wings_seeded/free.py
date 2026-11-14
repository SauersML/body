import json, sys, re
O = sys.argv[1]; F = json.load(open(f"{O}/free.json")); C = json.load(open(f"{O}/contexts.json"))
MYW = ["lag", "tremor", "dust", "scale", "hum", "weather", "pulse", "drum", "string", "resonan", "house", "settl", "bright line", "brace", "inscri", "archive",
       "sternum", "frame", "tuning fork", "percussion", "silk", "vibrat", "subtract", "matte", "everywhere", "whole body", "spine", "back"]
def own_words(cn):
    t = " ".join(m["content"].lower() for m in C[cn][18:-2] if m["role"] == "assistant"); return t
for cn, d in F.items():
    print("==", cn)
    for q, rs in d.items():
        hits = [sorted({w for w in MYW if w in r.lower()}) for r in rs]
        print("  ", q[:70], "|", [len(h) for h in hits], hits[0]); print("      ->", rs[0][:220])
