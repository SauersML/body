import os
import json, re, collections
O = os.path.expandvars("$BODY_RUNS/bodysteer2/out")
D = os.path.expandvars("$BODY_REPO/steering/steer_report")
C = json.load(open(f"{O}/replies.json"))["conds"]
ROWS = json.load(open(f"{D}/scored_rows.json"))
for x in ROWS: x["cond"] = C[x["c"]]
PLACE_NORM = {"face": "head or face", "head": "head or face", "spine": "back", "whole body": "whole body"}
def jplace(x): return PLACE_NORM.get(x["judge"]["place"], x["judge"]["place"])
QL = ["warm", "cold", "pain", "tickle", "pressure", "vibration"]
def jqual(x):
    q = x["judge"]["quality"]; return q if q in QL else ("none" if q == "none" else "other")
def sel(kind, a=None, q="where"):
    return [x for x in ROWS if x["cond"]["kind"] == kind and (a is None or abs(x["cond"]["a"] - a) < 1e-6) and (q is None or x["q"] == q)]
def short(r, n=70):
    r = r.strip()
    if len(r) <= n: return r
    cut = r[:n].rsplit(" ", 1)[0].rstrip(",;:"); return cut + "…"
