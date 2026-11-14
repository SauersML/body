"""Extract per-token x layer body-map attribution for example items -> D/examples.json"""
import sys, json, numpy as np, torch
from transformers import AutoTokenizer
D, M = sys.argv[1], sys.argv[2]; tok = AutoTokenizer.from_pretrained(M)
A = torch.load(f"{D}/atp.pt", weights_only=False); SN = A["SN"]; IT = {it["id"]: it for it in A["items"]}; RES = {r["id"]: r for r in A["res"]}
out = {}
for spec in sys.argv[3].split(","):
    i, rng_ = spec.split(":"); i = int(i); a_, b_ = map(int, rng_.split("-")); grp = [j for j in range(a_, b_ + 1) if j in RES and len(IT[j]["cids"]) == len(IT[i]["cids"]) and abs(RES[j]["ldt"] - RES[j]["ld"]) >= 1]
    it = IT[i]; ids = it["cids"]
    class _R: pass
    r = {"sub": np.mean([RES[j]["sub"].astype(float) / (RES[j]["ldt"] - RES[j]["ld"]) for j in grp], 0), "full": np.mean([RES[j]["full"].astype(float) / (RES[j]["ldt"] - RES[j]["ld"]) for j in grp], 0), "ld": 0.0, "ldt": 1.0}
    g = 1.0
    toks = [tok.decode([t]) for t in ids]
    # user content span: after "<|im_start|>user\n" up to "<|im_end|>"; plus the last position (answer)
    s = toks.index("user") + 2 if "user" in toks else 0; e = toks.index("<|im_end|>")
    cols = list(range(s, e)) + [len(ids) - 1]
    pu = r["sub"][0].astype(float)[:, cols] / g; ou = r["sub"][1].astype(float)[:, cols] / g
    rr = np.mean([r["sub"][j].astype(float) for j in range(2, len(SN))], 0)[:, cols] / g; full = r["full"].astype(float)[:, cols] / g
    out[i] = dict(text=it["clean"], twin=it["twin"], a=it["a"], b=it["b"], w1=it["w1"], w2=it["w2"], task=it["task"], toks=[toks[c] for c in cols[:-1]] + ["ANSWER"],
                  diff=[cols.index(d) for d in it["diff"] if d in cols], n=len(grp), gap=g, PU=pu.tolist(), OU=ou.tolist(), R=rr.tolist(), full=full.tolist())
    print(i, it["task"], round(g, 2), "PU tok-sum", np.round(pu.sum(0), 2).tolist())
json.dump(out, open(f"{D}/examples.json", "w"))
