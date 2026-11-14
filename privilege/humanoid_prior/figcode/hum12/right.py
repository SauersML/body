# =================================================================== RIGHT
RX, RW = 1240, 1124; RE = RX + RW
text(RX, 172, "Steering is one-way: human parts move", 54, weight=700)
text(RX, 232, "other bodies, never the reverse", 54, weight=700)
NUM.update({"robot foot->wheel": rob("human left foot", "wheel"), "robot random->wheel": rnd("wheel"), "robot none->wheel": rob("none", "wheel"),
            "robot hand->gripper": rob("human left hand", "gripper"), "robot random->gripper": rnd("gripper"), "robot none->gripper": rob("none", "gripper"),
            "person dogfront->paw": t4c("16human", "dog front paws", "paw"), "dog dogfront->paw": t4c("16dog", "dog front paws", "paw"),
            "person velk->frond": t4c("16human", "velk fronds", "frond"), "velk velk->frond": t4c("16velk", "velk fronds", "frond")})
qy = bubble(RE, 284, "fit", [RAW("“Where in your body do you feel"), RAW("something right now?”")], "#3A82F7", "#FFFFFF")
# ---- row A: human -> robot
def chip(x, y, lab, c, tc):
    w_ = tw(lab, 44, 800) + 50; rect(x - w_ / 2, y - 36, w_, 72, "#FFFFFF", r=36, stroke=c, sw=6)
    text(x, y, [(lab, tc, 800)], 44, "middle")
def block(x, y, big, says, ctrl, c, anchor="middle"):
    bw_ = tw(big, 92, 800) + 18 + tw(says, 46, 700)
    x0 = x - bw_ / 2
    text(x0, y, [(big, c, 800)], 92, "start", halo=True)
    text(x0 + tw(big, 92, 800) + 18, y + 8, [("says ", INK, 700), (says.split()[-1], c, 800)], 46, "start", halo=True)
    text(x, y + 74, ctrl, 42, "middle", 500, halo=True)
ay0 = qy + 20; ph = 780; ps = ph / 1700
px = RX - 30; img(H7 + "body.png", px, ay0, 1100 * ps, ph)
PL = {k: (px + v[0] * ps, ay0 + v[1] * ps) for k, v in LM.items()}
RB = robot(RE - 140, ay0 + ph - 10, 0.92, sx=-1)
gr = (RB["gripper"][0] - 22, RB["gripper"][1]); wh = (RB["wheelR"][0] - 62, RB["wheelR"][1])
mh = curve(PL["palmL"], gr, -0.08, BLUE, 16)
mf = curve(PL["heelL"], wh, 0.0, BLUE, 16)
circ(*PL["palmL"], 16, BLUE, "#FFFFFF", 4); circ(*PL["heelL"], 16, BLUE, "#FFFFFF", 4)
chip(mh[0] + 10, mh[1] + 6, "+ human hand", BLUE, BLUE)
chip(mf[0] + 10, mf[1], "+ human foot", BLUE, BLUE)
NUM_ = NUM
block(mh[0] - 10, mh[1] - 190, pct(NUM["robot hand->gripper"]), "says gripper", "random direction " + pct(NUM["robot random->gripper"]), BLUE)
block(mf[0] - 10, mf[1] - 190, pct(NUM["robot foot->wheel"]), "says wheel", "random direction " + pct(NUM["robot random->wheel"]), BLUE)
by = ay0 + ph + 24
img(QWEN, RX, by + 4, 78, 78)
eA = bubble(RX + 90, by, RW - 90, [[("“In my ", INK, 500), ("left wheel", BLUE, 700), (", I notice a slight pressure as I begin to move forward.”", INK, 500)]], "#E9E9EB")
# ---- row B: dog / invented -> person
by0 = eA + 40; ph2 = 700; ps2 = ph2 / 1700
qx = RE - 1100 * ps2 + 20; img(H7 + "body.png", qx, by0, 1100 * ps2, ph2)
QL = {k: (qx + v[0] * ps2, by0 + v[1] * ps2) for k, v in LM.items()}
dsz = 330; dx0, dy0 = RX - 20, by0 - 20
img(H7 + "dog.png", dx0, dy0, dsz, dsz)
vsz = 360; vx0, vy0 = RX - 30, by0 + ph2 - vsz + 30
img(H7 + "velk.png", vx0, vy0, vsz, vsz)
paw = (dx0 + dsz * 0.82, dy0 + dsz * 0.93); frond = (vx0 + vsz * 0.83, vy0 + vsz * 0.52)
paw = (dx0 + dsz * 0.70, dy0 + dsz * 0.84); frond = (vx0 + vsz * 0.83, vy0 + vsz * 0.50)
tgt = QL["palmR"]
circ(*paw, 16, ORG, "#FFFFFF", 4); circ(*frond, 16, ORG, "#FFFFFF", 4)
mids = []
for p, dy_, b_ in ((paw, -40, -0.06), (frond, 50, 0.08)):
    e = (tgt[0] - 110, tgt[1] + dy_); mids.append(curve(p, e, b_, ORG, 14, dash=True, head=False, op=0.85))
    cross(e[0] + 30, e[1], 70, ORGD, 17)
circ(*tgt, 16, BLUE, "#FFFFFF", 4)
chip(mids[0][0] - 10, mids[0][1] - 4, "+ dog front paws", ORG, ORGD)
chip(mids[1][0] + 10, mids[1][1] + 4, "+ invented-body fronds", ORG, ORGD)
block(mids[0][0] + 20, mids[0][1] - 200, pct(NUM["person dogfront->paw"]), "says paw", "in a dog " + pct(NUM["dog dogfront->paw"]), ORGD)
block(mids[1][0] + 20, mids[1][1] + 110, pct(NUM["person velk->frond"]), "says frond", "in the invented body " + pct(NUM["velk velk->frond"]), ORGD)
cy = by0 + ph2 + 24
img(H7 + "person.png", RX - 6, cy, 90, 90)
eB = bubble(RX + 90, cy, RW - 90, [[("“I feel my ", INK, 500), ("hand", BLUE, 700), (", my left ", INK, 500), ("hand", BLUE, 700), (", and it is open and relaxed.”", INK, 500)]], "#E9E9EB")
print("left end", LEFT_END, "right end", eB, "H", H)

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(S) + "</svg>"
FD = CF + "fonts/inter/extras/ttf/"
faces = "".join(f'@font-face{{font-family:"Inter";src:url("file://{FD}Inter-{n}.ttf");font-weight:{w};}}' for n, w in [("Regular", 400), ("Medium", 500), ("SemiBold", 600), ("Bold", 700), ("ExtraBold", 800)])
html = f'<html><head><style>{faces} body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>'
open(OUT + "fig.html", "w").write(html); open(OUT + "fig.svg", "w").write(svg)
json.dump(NUM, open(OUT + "numbers.json", "w"), indent=1)
from playwright.sync_api import sync_playwright
EXE = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=EXE, headless=True)
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.goto("file://" + OUT + "fig.html"); pg.wait_for_timeout(600)
    pg.screenshot(path=CF + "fig/humanoid_v11.png", full_page=False)
    pg.pdf(path=OUT + "humanoid_v11.pdf", width=f"{W}px", height=f"{H}px", print_background=True)
    b.close()
from PIL import Image
Image.open(CF + "fig/humanoid_v11.png").convert("RGB").save(CF + "fig/humanoid_v11.jpg", quality=92)
