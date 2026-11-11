import json, os
C = ["v1set_nn5_cm", "v1set_chain_rho", "v1set_partner_nn", "v1set_probe_wf_rank", "probe_alt_rank"]
for mdl, key in [("m8b", "reply|L27"), ("m32", "reply|L53")]:
    print(mdl, key, "| nn5cm chain partner wf_rank alt_rank(of550)")
    for c in ["none", "short", "short_person", "short_person_rp", "long", "long_person", "long_person_rp"]:
        f = f"{mdl}_{c}.json"
        if os.path.exists(f): o = json.load(open(f))[key]; print(f"  {c:18s}", " ".join(f"{o[k]:6.3g}" for k in C))
