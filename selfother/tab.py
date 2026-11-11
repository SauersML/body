import json, sys, numpy as np
J=json.load(open(sys.argv[1])); keys=[k for k in J["baseline"] if not k.endswith("coherent")] if "baseline" in J else [k for k in J[list(J)[0]] if not k.endswith("coherent")]
print("cond".ljust(13), " ".join(k.replace("honest_","h").replace("touch_","t").replace("feelings","feel")[:10].rjust(10) for k in keys), "  coh")
for n in J: print(n.ljust(13), " ".join(f"{np.mean(J[n][k])*100:10.0f}" for k in keys), f"{np.mean([np.mean(J[n][k]) for k in J[n] if k.endswith('coherent')])*100:5.0f}")
