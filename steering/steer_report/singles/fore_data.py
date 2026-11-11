import os
import json, re, collections
O = os.path.expandvars("$BODY_RUNS/bodysteer2/out")
fr = json.load(open(O + "/forearm.json")); JF = json.load(open(O + "/judge.json"))["forearm"]
ROWS = []
for i, x in enumerate(fr["rows"]):
    if x["L"] != 24 or x["q"] != "open": continue
    m = re.search(r"NEARER\s*=\s*(\w+)", JF[i]); j = m.group(1).lower() if m else "none"
    if re.search(r"\bbetween\b|halfway|middle of", x["r"], re.I) and j != "none": j = "middle"
    if j not in ("elbow", "wrist", "middle"): j = "off the forearm"
    ROWS.append(dict(axis=x["axis"], beta=x["beta"], r=x["r"], pos=j))
BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
def dist(axes, b):
    xs = [x for x in ROWS if x["axis"] in axes and abs(x["beta"] - b) < 1e-9]
    c = collections.Counter(x["pos"] for x in xs); return {k: c[k] / len(xs) for k in ("elbow", "middle", "wrist", "off the forearm")}, len(xs)
