import json, sys
R = json.load(open(sys.argv[1]))
C = R["curve"]
print("chance", R["chance"], "clip", round(R["clip_frac_skin_vectors"], 3), "model act acc", round(R["model_action_acc_on_expert_rows"], 3))
for k, v in C.items():
    m = [x[0] for x in v]; best = max(range(len(m)), key=lambda i: m[i]); lo = min(m)
    first90 = next(i for i in range(len(m)) if m[i] - m[0] >= 0.9 * (m[best] - m[0])) if m[best] > m[0] else None
    print(f"{k:32s} L0 {m[0]:.3f} L7 {m[7]:.3f} L8 {m[8]:.3f} L12 {m[12]:.3f} L18 {v[18]} L35 {m[35]:.3f} L36 {m[36]:.3f} best L{best} {v[best]} first90% L{first90}")
for k, v in R.get("base", {}).items(): print("base", k, v)
