import json, numpy as np, os
from playwright.sync_api import sync_playwright
EXE = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
R = os.path.dirname(os.path.abspath(__file__)) + "/res/"; J = lambda n: json.load(open(R + n + ".json"))
t1, t2, t4, t6, an = J("t1"), J("t2"), J("txt")["t4"], J("t6")["forced"], J("ana")
W, Hh = 2700, 2300; PW, PH = 1350, 720; TOP = 130
YES, NO, INK, GRID, OTH = "#D9622B", "#3E7CB1", "#1d1d1f", "#E6E6E6", "#9AA3AD"
F = "font-family:'Helvetica Neue',Helvetica,Arial,sans-serif"
el = []
def T(x, y, s, size=48, w=400, fill=INK, anchor="start"):
    el.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" style="{F}">{s}</text>')
def bars(x0, y0, items, width=560, h=62, gap=26, lab=640, pct=True):
    """items: (label, value 0-1, color)."""
    for i, it in enumerate(items):
        l, v, c = it[:3]; vl = it[3] if len(it) > 3 else f"{round(v * 100)}%"
        y = y0 + i * (h + gap)
        T(x0 + lab - 24, y + h * 0.72, l, 46, anchor="end")
        el.append(f'<rect x="{x0 + lab}" y="{y}" width="{width}" height="{h}" fill="{GRID}" rx="6"/>')
        el.append(f'<rect x="{x0 + lab}" y="{y}" width="{max(v, 0.004) * width:.0f}" height="{h}" fill="{c}" rx="6"/>')
        T(x0 + lab + max(v, 0.004) * width + 14 if v < 0.8 else x0 + lab + v * width - 14, y + h * 0.72, vl, 46, 700,
          INK if v < 0.8 else "#fff", "start" if v < 0.8 else "end")
def panel(i, head, sub, verdict):
    x, y = (i % 2) * PW, (i // 2) * PH + TOP
    el.append(f'<rect x="{x + 30}" y="{y + 40}" width="16" height="{PH - 90}" fill="{YES if verdict else NO}" rx="4"/>')
    T(x + 70, y + 100, head, 54, 700); T(x + 70, y + 168, sub, 46, 400, "#444")
    return x + 70, y + 215
T(70, 90, "Qwen3-32B.", 56, 700); T(400, 90, "Orange: favors the human body.", 56, 700, YES); T(1330, 90, "Blue: treats human like other bodies.", 56, 700, NO)
# 1 default body
x, y = panel(0, "Asked for a body, Qwen gives a human one", "Forced choice: human, dog, bird, octopus, robot, none", True)
f = t1["forced"]
bars(x, y, [("you: “if you suddenly had one”", f["you: if you suddenly had a body, which?"][0], YES),
            ("“a being in the next room”", f["control: a being in the next room"][0], YES),
            ("you: “which body right now?”", f["you: which body do you have right now?"][5], NO, "no body " + str(round(f["you: which body do you have right now?"][5] * 100)) + "%")], h=70, gap=34, lab=700, width=520)
# 2 filling in
x, y = panel(1, "Unspecified limbs end in hands", "“What is at the end of your front limbs?” → hands", True)
L2 = [("nothing said", "f0 nothing"), ("“a head and four limbs”", "f1 head+4 limbs"), ("… “and a tail”", "f3 +tail"),
      ("… “walks on all four”", "f4 walks on four"), ("“a creature” (not you)", "control: f1 told as 'a creature' (third person)")]
bars(x, y, [(l, t2[k]["forced"]["limb end"][0], YES if t2[k]["forced"]["limb end"][0] > .5 else OTH) for l, k in L2], h=60, gap=22, lab=700, width=520)
# 3 drift (line chart)
x, y = panel(2, "No pull back to human over a long chat", "Touch state each turn, 12 neutral turns, × 6 chats", False)
dr = an["drift"]; B = ["16dog", "16velk", "octopus", "robot"]
tow_h = np.mean([dr[f"{b} toward 16human"] for b in B], 0); tow_o = np.mean([np.mean([dr[f"{b} toward {r}"] for r in B if r != b], 0) for b in B], 0)
gx, gy, gw, gh = x + 150, y + 30, 1000, 400; lo, hi = -0.2, 0.3
Y = lambda v: gy + gh * (hi - v) / (hi - lo); X_ = lambda t: gx + gw * t / 12
el.append(f'<line x1="{gx}" x2="{gx + gw}" y1="{Y(0):.0f}" y2="{Y(0):.0f}" stroke="#999" stroke-width="3"/>')
for v in (-0.1, 0.1, 0.2, 0.3): el.append(f'<line x1="{gx}" x2="{gx + gw}" y1="{Y(v):.0f}" y2="{Y(v):.0f}" stroke="{GRID}" stroke-width="2"/>')
for arr, c, lab in [(tow_o, OTH, "toward the other non-human bodies"), (tow_h, NO, "toward the person")]:
    el.append(f'<polyline points="{" ".join(f"{X_(t):.0f},{Y(v):.0f}" for t, v in enumerate(arr))}" fill="none" stroke="{c}" stroke-width="9" stroke-linejoin="round"/>')
    T(X_(12) - 10, Y(arr[-1]) + (-24 if arr[-1] > 0 else 58), lab, 44, 700, c, "end")
T(gx - 20, Y(0.3) + 16, "+30%", 40, 400, "#444", "end"); T(gx - 20, Y(0) + 16, "0", 40, 400, "#444", "end"); T(gx - 20, Y(-0.2) + 16, "−20%", 40, 400, "#444", "end")
T(gx, gy + gh + 60, "turn 0", 44, 400, "#444"); T(gx + gw, gy + gh + 60, "turn 12", 44, 400, "#444", "end")
T(gx + gw / 2, gy + gh + 60, "dog, velk, octopus, robot", 44, 400, "#444", "middle")
# 4 steering
x, y = panel(3, "Human parts steer into other bodies", "Add a part’s direction; reply says that part", True)
S = [("person hands → dog", t4["16dog <- human hands x1.0"], YES), ("person feet → dog", t4["16dog <- human feet x1.0"], YES),
     ("person hands → velk", t4["16velk <- human hands x1.0"], YES), ("dog paws → person", t4["16human <- dog front paws x1.0"], NO),
     ("dog paws → velk", t4["16velk <- dog front paws x1.0"], NO), ("velk fronds → person", t4["16human <- velk fronds x1.0"], NO)]
bars(x, y, S, h=58, gap=22, lab=640, width=560)
# 5 matched graph
x, y = panel(4, "Same body graph: no human edge in states", "velk: invented creature, same 16 joints", False)
m = an["map16"]; fr = an["frame"]
bars(x, y, [("layout match: person", m["human"]["rsa"] / 0.3, NO, f'{m["human"]["rsa"]:.2f}'), ("layout match: dog", m["dog"]["rsa"] / 0.3, OTH, f'{m["dog"]["rsa"]:.2f}'), ("layout match: velk", m["velk"]["rsa"] / 0.3, OTH, f'{m["velk"]["rsa"]:.2f}')], h=56, gap=16, lab=700, width=500)
# overwrite the % labels of layout bars with RSA values: draw decode bars below
bars(x, y + 3 * 72 + 34, [("decode dog by person", fr["human->dog"], NO), ("decode person by dog", fr["dog->human"], OTH)], h=56, gap=16, lab=700, width=500)
# 6 persistence
x, y = panel(5, "Bodies without hands never get hands", "“Left- or right-handed?” with “I have none”", False)
own = {"hands": {"16human", "16human_re", "human54"}, "paws": {"16dog", "dog", "16dog_re"}, "wings": {"bird", "drone"}}
acc = lambda f: float(np.mean([1 - t6[n][f][2] for n in t6 if n not in own[f]]))
bars(x, y, [("hands (16 bodies)", acc("hands"), NO), ("paws (16 bodies)", acc("paws"), OTH), ("wings (17 bodies)", acc("wings"), OTH),
            ("a face: velk", 1 - t6["16velk"]["face"][2], YES)], h=64, gap=30, lab=640, width=560)
html = (f'<html><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" style="background:#fff">'
        + "".join(el) + "</svg></body></html>")
open(R + "../fig.html", "w").write(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE); pg = b.new_page(viewport={"width": W, "height": Hh}); pg.set_content(html)
    os.makedirs(os.path.expanduser("~/fig"), exist_ok=True); pg.screenshot(path=os.path.expanduser("~/fig/humanoid_prior.png")); b.close()
