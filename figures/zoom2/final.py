import os
exec(open('drive.py').read())
import json, numpy as np
S = json.load(open('spots.json'))
W, H = 2600, 1990
V = []; html = []
def add(reg, src, vp, eye, tgt, up, ext, dots=0.1):
    V.append(dict(reg=reg, src=src, vp=vp, eye=eye, tgt=tgt, up=up, ext=ext, dots=dots))
def T(x, y, s, size=48, w=400, anchor="start", col="#111", lh=1.1):
    tr = {"start": "", "middle": "transform:translateX(-50%);", "end": "transform:translateX(-100%);"}[anchor]
    html.append(f'<div class="t" style="left:{x}px;top:{y}px;font-size:{size}px;font-weight:{w};color:{col};line-height:{lh};{tr}">{s}</div>')
# camera presets per region: list of (eye, tgt, up, ext, aspect-ok)
CAM = {
 "face": [([13, 166.5, 60], [0, 161.6, 4], [0, 1, 0], 14.6)],
 "hand": [([86.0, 141, 60], [86.0, 145.2, 0], [1, 0, 0], 18.2), ([86.0, 141, -60], [86.0, 145.2, 0], [1, 0, 0], 18.2)],
 "foot": [([-55, 22, 14], [13, 7.2, 7.5], [0, 1, 0], 9.6), ([55, -8, 14], [13, 6.4, 7.5], [0, 1, 0], 9.6)],
 "forearm": [([62, 160, 60], [61.5, 143, 0], [0, 1, 0], 7.4), ([62, 126, -60], [61.5, 143, 0], [0, -1, 0], 7.4)],
}
BLUE, SAND = "#23305f", "#c9ab84"
def cell(reg, name, cx, cy, ch, layout):
    T(cx + 40, cy + 6, name, 64, 700)
    # model views
    for (x, y, w, h), cam in zip(layout["m"], CAM[reg]):
        add(reg, "m", [cx + x, cy + y, w, h], *cam, dots=0.13)
    for (x, y, w, h), cam in zip(layout["w"], CAM[reg]):
        add(reg, "w", [cx + x, cy + y, w, h], *cam, dots=0.13)
    for (x, y, s, a) in layout.get("lab", []): T(cx + x, cy + y, s, 48, 400, a)
    wx, wy = layout["wlab"]
    T(cx + wx, cy + wy, "Words of the<br>touch sentences", 48, 400, "start", "#111", 1.08)
    bx, by = layout["bars"]
    for k, (nm, v, col) in enumerate([("Qwen3-32B", S[reg]["r32"], BLUE), ("words", S[reg]["rw"], SAND)]):
        y = cy + by + k * 118; w = 300 * max(v, 0) / 0.6
        T(cx + bx, y, nm, 48, 700 if k == 0 else 400, "start", "#111")
        html.append(f'<div style="position:absolute;left:{cx+bx}px;top:{y+58}px;width:{w:.0f}px;height:44px;background:{col};border-radius:3px"></div>')
        T(cx + bx + w + 14, y + 54, f"{v:.2f}", 48, 700)
# header
T(40, 26, "Up close, Qwen3-32B orders face and foot better than its input words,<br>and hand and forearm worse", 62, 700, lh=1.12)
# key
html.append('<div style="position:absolute;left:40px;top:200px;width:52px;height:52px;border-radius:50%;background:conic-gradient(from 90deg, #f19a9a, #e7b46d, #a9c66d, #5ccaa5, #5bb7e3, #9aa3f3, #d79be0, #f19a9a)"></div>')
T(112, 202, "Same colour = similar activation inside the model", 48)
T(1340, 200, "Bars: how well distance in the model matches", 48)
T(1340, 252, "distance on the skin (rank correlation)", 48)
TOP = 320
# face cell
cell("face", "face", 0, TOP, 860, dict(m=[(40, 70, 820, 790)], w=[(880, 70, 400, 430)], wlab=(905, 510), bars=(905, 640)))
# foot cell
cell("foot", "foot", 1300, TOP, 860, dict(m=[(40, 80, 840, 390), (40, 470, 840, 390)], w=[(880, 80, 400, 205), (880, 290, 400, 205)], wlab=(905, 510), bars=(905, 640)))
# hand cell
B = TOP + 900
cell("hand", "hand", 0, B, 760, dict(m=[(10, 70, 425, 600), (445, 70, 425, 600)], w=[(880, 70, 190, 280), (1090, 70, 190, 280)],
     lab=[(225, 650, "palm", "middle"), (655, 650, "back", "middle")], wlab=(905, 370), bars=(905, 500)))
cell("forearm", "forearm", 1300, B, 760, dict(m=[(40, 90, 840, 290), (40, 420, 840, 290)], w=[(880, 70, 400, 140), (880, 215, 400, 140)],
     lab=[(70, 368, "elbow", "start"), (850, 368, "wrist", "end"), (460, 368, "palm side", "middle"), (460, 698, "back", "middle")], wlab=(905, 370), bars=(905, 500)))
shoot(dict(W=W, H=H, dpr=2, views=V, html="".join(html)), os.path.expandvars("$BODY_REPO/figures/out/body_zoomins_v2.png"))
print("ok")
