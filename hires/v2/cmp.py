import json, sys
C = ["v1set_nn5_cm", "v1set_chain_rho", "v1set_partner_nn", "v1set_probe_wf_rank", "nn5_cm", "chain_rho", "probe_alt_rank"]
keys = sys.argv[2].split(",")
print("cond".ljust(16), "key".ljust(18), " ".join(c.replace("v1set_", "1:")[:9].rjust(9) for c in C))
for f in sys.argv[1].split(","):
    o = json.load(open(f if f.endswith(".json") else f"m8b_{f}.json"))
    for k in keys:
        if k in o: print(f.ljust(16), k.ljust(18), " ".join(f"{o[k].get(c, float('nan')):9.3g}" for c in C))
