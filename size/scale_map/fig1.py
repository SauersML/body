import json; from svg import *
N = json.load(open("q1_numbers.json")); F = Fig(2300, 2300); G = "#C9CDD3"
F.t(60, 90, "Bigger Qwen3: sharper body map, not better body control", 60, 700)
# panel A: text map
F.t(60, 200, "Map from touch sentences:", FS, 700); F.t(60, 255, "cm between neighbours (lower is better)", FS)
tm = N["text_map_cm"]
F.bars(110, 330, 420, [("Qwen3-8B", tm["8B"], None, None, C["8B"]), ("Qwen3-32B", tm["32B"], None, None, C["32B"]), ("word|vectors", 32.0, None, None, G)],
       34, bw=200, gap=90, fmt=lambda v: f"{v:.0f} cm")
# panel B: poke readout
F.t(1250, 200, "Body controller: where the poke", FS, 700); F.t(1250, 255, "was, read from its states", FS, 700)
px = N["poke_xyz_r2"]
F.bars(1380, 330, 420, [("Qwen3-0.6B", px["0.6B"], None, None, C["0.6B"]), ("Qwen3-8B", px["8B"], None, None, C["8B"])], 1.0, bw=260, gap=160,
       fmt=lambda v: f"{round(v*100)}% explained")
# panel C: control
F.t(60, 1010, "Body controller, same training, 192 new scenes: success rate", FS, 700)
tasks = ["reach", "touch where touched", "exact spot", "grasp"]; TL = {"reach": "touch the body part|it was asked to", "touch where touched": "touch where|you were touched",
      "exact spot": "within 5 cm of|the touched spot", "grasp": "grab the|handle"}
for row, (st, lab) in enumerate((("r0", "copied the teacher"), ("r2", "+ 2 rounds of practice"))):
    y0 = 1230 + row * 530; F.t(60, y0 + 30, lab, FS, 700)
    for k, t in enumerate(tasks):
        d = N["control"][f"{st}|{t}"]; x0 = 120 + k * 545
        F.bars(x0, y0 + 80, 330, [("8B", *d["8B"], C["8B"]), ("32B", *d["32B"], C["32B"])], 1.0, bw=170, gap=40,
               fmt=lambda v: f"{round(v*100)}%")
        if d["p"] < .05: F.t(x0 + 205, y0 + 75, "p=" + (f"{d['p']:.3f}".lstrip("0")), FS, 400, INK, "middle")
for k, t in enumerate(tasks):
    for j, part in enumerate(TL[t].split("|")): F.t(120 + k * 545 + 190, 1100 + j * 52, part, FS, 700, INK, "middle")
F.save("../body_scale_size_v1")
