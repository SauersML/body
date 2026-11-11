import json, sys, numpy as np
o, ohf, L = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(f"{o}/anaf.json"))[f"reply|{L}"]; h = json.load(open(f"{ohf}/ana_hf.json"))[f"reply|{L}"]; p = a["probe"]
hand = ["palm_to_index_left", "palm_across_left", "backhand_left"]; face = ["face_up", "cheek_left"]
m = lambda ks, k: float(np.mean([h[x][k] for x in ks]))
n = dict(body=dict(same_step=a["fr_same_step_cos_full"], same_place=a["fr_same_place_cos_full"]), hand=dict(same_step=m(hand, "same_step"), same_place=m(hand, "same_place")),
         face=dict(same_step=m(face, "same_step"), same_place=m(face, "same_place")),
         probe_paths=float(np.mean([p[k] for k in ["arm_ant", "arm_feather", "arm_finger", "leg", "head", "spine", "chest"]] + [h[x]["probe"] for x in hand + face])), probe_spots=p["landmark_spots_arms"])
json.dump(n, open(sys.argv[4], "w")); print(n)
