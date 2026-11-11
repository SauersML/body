import os
import json, os, numpy as np
from playwright.sync_api import sync_playwright
D = os.path.dirname(os.path.abspath(__file__)); F = json.load(open(f"{D}/fig_numbers.json"))
W, H = 2600, 2250
CH = ["root", "base", "inner", "mid", "outer", "longf", "tip"]; TT = [0, .1, .25, .5, .7, .88, 1]
pc = {k: v[0] for k, v in F["pc"].items()}; lo, hi = min(pc.values()), max(pc.values())
def col(v):
    x = (v - lo) / (hi - lo); a = np.array([28, 110, 140]); b = np.array([238, 150, 30]); m = np.array([200, 200, 200])
    c = a + (m - a) * (2 * x) if x < .5 else m + (b - m) * (2 * x - 1); return "#%02x%02x%02x" % tuple(int(t) for t in c)
# back view geometry (origin = centre between shoulder blades)
OX, OY = 800, 1200
def wingpt(t, s, off=0.0):
    # quadratic bezier root -> tip, sweeping up and out; s=-1 viewer left (person's left, seen from behind)
    p0 = np.array([40, 0]); p1 = np.array([300, -330]); p2 = np.array([600, -230])
    p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    d = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1); n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    p = p + off * n * (1 if n[1] > 0 else -1)
    return OX + s * p[0], OY + p[1]
def wingpath(s):
    top = [wingpt(t, s, -40) for t in np.linspace(0, 1, 30)]
    bot = [wingpt(t, s, 120 + 160 * np.sin(np.pi * t) ** .8 - 90 * t) for t in np.linspace(1, 0, 30)]
    pts = top + bot; return "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in pts) + "Z"
def feathers(s):
    out = []
    for t in np.linspace(.15, .98, 16):
        x0, y0 = wingpt(t, s, 10); x1, y1 = wingpt(t, s, 120 + 160 * np.sin(np.pi * t) ** .8 - 90 * t)
        out.append(f'<path d="M{x0:.0f},{y0:.0f} L{x1:.0f},{y1:.0f}" stroke="#d8d2c4" stroke-width="5" fill="none"/>')
    return "".join(out)
body = f'''<g fill="#efe9df" stroke="#9a9184" stroke-width="5">
<circle cx="{OX}" cy="{OY-215}" r="88"/>
<rect x="{OX-38}" y="{OY-140}" width="76" height="70" rx="20"/>
<path d="M{OX-165},{OY-80} Q{OX},{OY-110} {OX+165},{OY-80} L{OX+125},{OY+330} L{OX-125},{OY+330} Z"/>
<path d="M{OX-165},{OY-80} Q{OX-215},{OY-60} {OX-212},{OY+80} L{OX-235},{OY+420} Q{OX-238},{OY+500} {OX-205},{OY+520} L{OX-180},{OY+430} L{OX-160},{OY+80} Z"/>
<path d="M{OX+165},{OY-80} Q{OX+215},{OY-60} {OX+212},{OY+80} L{OX+235},{OY+420} Q{OX+238},{OY+500} {OX+205},{OY+520} L{OX+180},{OY+430} L{OX+160},{OY+80} Z"/>
<path d="M{OX-125},{OY+330} L{OX-120},{OY+700} L{OX-25},{OY+700} L{OX},{OY+380} L{OX+25},{OY+700} L{OX+120},{OY+700} L{OX+125},{OY+330} Z"/>
</g>
<path d="M{OX-95},{OY-40} Q{OX-55},{OY+30} {OX-40},{OY+110}" stroke="#b5ab9c" stroke-width="5" fill="none"/>
<path d="M{OX+95},{OY-40} Q{OX+55},{OY+30} {OX+40},{OY+110}" stroke="#b5ab9c" stroke-width="5" fill="none"/>'''
wings = "".join(f'<path d="{wingpath(s)}" fill="#faf8f3" stroke="#9a9184" stroke-width="5"/>' + feathers(s) for s in (-1, 1))
dots = []
for s, sn in ((-1, "left"), (1, "right")):
    for k, t in zip(CH, TT):
        x, y = wingpt(t, s, 40); dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="30" fill="{col(pc[k+":"+sn])}" stroke="#222" stroke-width="4"/>')
    for k, t, off in (("lead", .5, -40), ("trail", .6, 165)):
        x, y = wingpt(t, s, off); dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="24" fill="{col(pc[k+":"+sn])}" stroke="#222" stroke-width="4"/>')
# nearest human places (person's left wing = viewer's left), from the model's states with words of the wing removed
N = F["near"]; HUMXY = {"scap": (OX - 70, OY + 10), "midback": (OX, OY + 40), "ftip": (OX - 205, OY + 525), "ribs": (OX - 120, OY + 200), "hip": (OX - 120, OY + 330)}
LAB = {"scap": "wing root: left shoulder blade", "midback": "between the shoulder blades", "ftip": "left fingertip", "ribs": "left ribs", "hip": "left hip"}
arcs = []; used = set()
for k, t, off in (("root", 0, 40), ("inner", .25, 40), ("tip", 1, 40)):
    hk = N[k + ":left"][0][0].split(":")[0]; x, y = wingpt(t, -1, off); hx, hy = HUMXY[hk]
    cx, cy = (x + hx) / 2 - 140, (y + hy) / 2 + 60
    arcs.append(f'<path d="M{x:.0f},{y:.0f} Q{cx:.0f},{cy:.0f} {hx},{hy}" stroke="#7a2e8a" stroke-width="7" fill="none" stroke-dasharray="18 12"/>'
                f'<circle cx="{hx}" cy="{hy}" r="16" fill="#7a2e8a"/>'); used.add(hk)
labs = {"ribs": (OX - 95, OY + 285, "start"), "scap": (OX - 60, OY + 105, "start"), "midback": (OX + 0, OY + 290, "middle"), "ftip": (OX - 330, OY + 600, "middle")}
lab = "".join(f'<text x="{labs[h][0]}" y="{labs[h][1]}" text-anchor="{labs[h][2]}" class="p lab">{LAB[h]}</text>' for h in used if h in labs)
wl = [(wingpt(0, -1, 40), "root"), (wingpt(1, -1, 40), "tip")]
wtxt = f'<text x="{wl[1][0][0]-10:.0f}" y="{wl[1][0][1]-60:.0f}" class="b" text-anchor="middle">wing tip</text><text x="{wl[0][0][0]-60:.0f}" y="{wl[0][0][1]-150:.0f}" class="b" text-anchor="middle"></text>'
# chain plot
PX, PY, PW, PH = 1660, 1230, 820, 600
def lp(vals, c, w, dash=""):
    pts = " ".join(f"{PX + t * PW:.0f},{PY + PH - v * PH:.0f}" for t, v in zip(TT, vals))
    return f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{w}" stroke-dasharray="{dash}"/>' + "".join(
        f'<circle cx="{PX + t * PW:.0f}" cy="{PY + PH - v * PH:.0f}" r="13" fill="{c}"/>' for t, v in zip(TT, vals))
mean2 = lambda k: [(a + b) / 2 for a, b in zip(F[k]["left"], F[k]["right"])]
plot = f'''<line x1="{PX}" y1="{PY+PH}" x2="{PX+PW}" y2="{PY+PH}" stroke="#222" stroke-width="4"/>
<line x1="{PX}" y1="{PY}" x2="{PX}" y2="{PY+PH}" stroke="#222" stroke-width="4"/>
{lp(mean2("fromroot_words"), "#9a9184", 7, "16 12")}{lp(mean2("fromroot_N"), "#1c6e8c", 7)}{lp(mean2("fromroot_W"), "#c0392b", 10)}
<text x="{PX}" y="{PY+PH+70}" class="p">root</text><text x="{PX+PW}" y="{PY+PH+70}" class="p" text-anchor="end">tip</text>
<text x="{PX+PW/2}" y="{PY+PH+140}" class="p" text-anchor="middle">touched spot along the wing</text>
<text transform="translate({PX-40},{PY+PH/2}) rotate(-90)" class="p" text-anchor="middle">model distance from root</text>
<line x1="1570" y1="1030" x2="1680" y2="1030" stroke="#c0392b" stroke-width="10"/><text x="1700" y="1047" class="p">chat with wings</text>
<line x1="1570" y1="1095" x2="1680" y2="1095" stroke="#1c6e8c" stroke-width="7"/><text x="1700" y="1112" class="p">same touches, chat without wings</text>
<line x1="1570" y1="1160" x2="1680" y2="1160" stroke="#9a9184" stroke-width="7" stroke-dasharray="16 12"/><text x="1700" y="1177" class="p">word meaning alone</text>'''
legend = f'''<defs><linearGradient id="g"><stop offset="0" stop-color="{col(lo)}"/><stop offset=".5" stop-color="{col((lo+hi)/2)}"/><stop offset="1" stop-color="{col(hi)}"/></linearGradient></defs>
<rect x="330" y="2120" width="700" height="50" fill="url(#g)" stroke="#222" stroke-width="3"/>
<text x="315" y="2160" class="p" text-anchor="end">root-like</text><text x="1045" y="2160" class="p">tip-like</text>
<text x="680" y="2100" class="p" text-anchor="middle">dot colour: main axis of the model's own states</text>'''
method = f'''<rect x="60" y="40" width="2480" height="560" rx="30" fill="#f4f1fb"/>
<text x="100" y="120" class="b">Chat with Qwen3-8B, after a warm hello and a body scan:</text>
<text x="100" y="200" class="p">Me: "Wings are sprouting from your shoulder blades. They unfold, feathered and wide."</text>
<text x="100" y="275" class="p" fill="#5a2a6a">Qwen: "a light, tingling warmth spreading across my shoulder blades, as if something</text>
<text x="100" y="335" class="p" fill="#5a2a6a">soft and feathery is unfolding behind each shoulder."</text>
<text x="100" y="420" class="p">Then 80 touches, 10 spots per wing: "A feather brushes slowly across the very tip of your left wing."</text>
<text x="100" y="495" class="p">Read the model's state at the first token of its reply (layer 27 of 36).</text>
<text x="100" y="565" class="p" fill="#5a2a6a">Qwen: "A light, ticklish sensation moves across the tip of my left wing."</text>'''
heads = f'''<text x="80" y="700" class="h">Touches on new wings: the wing root lands on the shoulder blade</text>
<text x="1560" y="900" class="h2">Root-to-tip order: most of</text><text x="1560" y="965" class="h2">it is already in the words</text>
<text x="80" y="790" class="p" fill="#7a2e8a">dashed arcs: nearest human body place in the model's states</text>'''
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>.h{{font:700 64px Inter,Helvetica,sans-serif}}.h2{{font:700 56px Inter,Helvetica,sans-serif}}.b{{font:700 50px Inter,Helvetica,sans-serif}}.p{{font:500 48px Inter,Helvetica,sans-serif}}text{{fill:#111}}.lab{{paint-order:stroke;stroke:#fff;stroke-width:16px;stroke-linejoin:round}}</style>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>{method}{heads}{wings}{body}{"".join(dots)}{"".join(arcs)}{lab}{wtxt}<text x="{OX-330}" y="{OY+660}" class="p lab" text-anchor="middle" fill="#7a2e8a">(a made-up "glorp" tip lands here too)</text>{plot}{legend}</svg>'''
open(f"{D}/fig.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as p:
    b = p.chromium.launch(channel="chromium", headless=True); pg = b.new_context(viewport={"width": W, "height": H}).new_page()
    pg.goto("file://" + f"{D}/fig.html"); os.makedirs(os.path.expandvars("$BODY_REPO/figures/out"), exist_ok=True)
    pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/body_wings_map.png")); b.close()
