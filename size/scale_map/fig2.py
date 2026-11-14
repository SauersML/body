import json; from svg import *
N = json.load(open("q2_numbers.json")); F = Fig(2300, 1530); TOUCH, SENT, G = "#2B5FA8", "#E07B24", "#C9CDD3"
F.t(60, 90, "Qwen3-8B body controller: one body map for touch and sentences,", 58, 700)
F.t(60, 160, "but only touch-made directions move the hand", 58, 700)
F.t(60, 290, f"Map shape from sentences vs from skin touches: rank correlation {N['rsa']:.2f}".replace("0.", "."), FS, 700)
F.t(60, 400, "Add a place direction made from", FS); F.t(760, 400, "skin touches", FS, 700, TOUCH); F.t(1060, 400, "or", FS); F.t(1125, 400, "sentences", FS, 700, SENT)
F.t(1360, 400, "or at random:", FS)
pan = [("hand ends closer to the place", "than a random direction", [("touches", *N["hand_gain_sensor"], TOUCH), ("sentences", *N["hand_gain_word"], SENT)], 30, lambda v: f"{v:.0f} cm"),
       ("hand touches", "the place", [("touches", *N["hand_touch_sensor"], TOUCH), ("sentences", *N["hand_touch_word"], SENT), ("random", *N["hand_touch_random"], G)], 1, lambda v: f"{round(v*100)}%"),
       ("says it was touched", "at that place", [("touches", *N["name_sensor1.5"], TOUCH), ("sentences", *N["name_word1.5"], SENT), ("random", *N["name_random1.5"], G)], 1, lambda v: f"{round(v*100)}%")]
for k, (a, b, items, vm, fmt) in enumerate(pan):
    x = 90 + k * 760; F.t(x, 530, a, FS, 700); F.t(x, 585, b, FS, 700)
    F.bars(x + 20, 680, 720, [(l.replace("touches", "touches").replace("sentences", "sentences"), v, lo, hi, c) for l, v, lo, hi, c in items], vm, bw=180, gap=40, fmt=fmt)
F.save("../body_scale_shared_v1")
