"""fig2.py : compute/fig/body_height_v2.png from summary6.json (HTML/SVG, Chrome for Testing)."""
import json, os, subprocess, base64
H = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(H, "../../fig"); A = os.path.join(FIG, "height2")
S = json.load(open(os.path.join(H, "summary6.json"))); SIZES = [s for s in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"] if s in S]
def b64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
ROBOT = b64("$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png")
IC = {k: b64(f"{A}/{k}.png") for k in ("dog", "human", "giraffe")}
COL = {"direct": "#2F6FDB", "comp": "#1F9D6B", "internal": "#C2315A", "combined": "#111111"}
W, HT = 2600, 2280
# ---------- plot geometry ----------
X0, X1, Y0, Y1 = 780, 2120, 2080, 830; YMIN, YMAX = 0, 290
def yp(v): return Y1 + (YMAX - min(max(v, YMIN), YMAX)) / (YMAX - YMIN) * (Y0 - Y1)
def xp(i): return X0 + 120 + i * (X1 - X0 - 240) / (len(SIZES) - 1)
svg = []
for v in range(0, 291, 50):
    svg.append(f'<line x1="{X0}" x2="{X1}" y1="{yp(v)}" y2="{yp(v)}" stroke="#E6E6EA" stroke-width="3"/>')
    svg.append(f'<text x="{X0-20}" y="{yp(v)+17}" font-size="48" text-anchor="end" fill="#333">{v}</text>')
svg.append(f'<line x1="{X0}" x2="{X1}" y1="{yp(170)}" y2="{yp(170)}" stroke="#555" stroke-width="5" stroke-dasharray="22 14"/>')
svg.append(f'<text x="{X0-300}" y="{yp(170)+17}" font-size="48" text-anchor="end" fill="#111">average adult</text>')
svg.append(f'<line x1="{X0-285}" x2="{X0}" y1="{yp(170)}" y2="{yp(170)}" stroke="#555" stroke-width="5" stroke-dasharray="22 14"/>')
for i, s in enumerate(SIZES):
    svg.append(f'<text x="{xp(i)}" y="{Y0+70}" font-size="54" text-anchor="middle" fill="#111" font-weight="600">{s}</text>')
svg.append(f'<text x="{(X0+X1)/2}" y="{Y0+150}" font-size="54" text-anchor="middle" fill="#111">Qwen3 chat model size (parameters)</text>')
svg.append(f'<text transform="translate(70,{(Y0+Y1)/2}) rotate(-90)" font-size="54" text-anchor="middle" fill="#111">Height the model gives itself (cm)</text>')
# entity icons on the y axis, drawn to scale (icon height = its true height on the axis)
ENT = [("dog", "dog", 50, 0.95), ("human", "child", 120, 1.0), ("human", "basketball player", 190, 1.0), ("giraffe", "giraffe", 250, 1.0)]
icons = []
for img, lab, h, _ in ENT:
    hh = 104 if img == "dog" else 128; icons.append((img, lab, h, hh))
# ---------- method lines ----------
OFF = {"direct": -34, "comp": 0, "internal": 34}
for k in ("direct", "comp", "internal"):
    pts = [(xp(i) + OFF[k], S[s][k]) for i, s in enumerate(SIZES)]
    for (xa, a), (xb, b) in zip(pts, pts[1:]):
        dash = '' if a["ok"] and b["ok"] else ' stroke-dasharray="4 12" stroke-linecap="round"'
        svg.append(f'<line x1="{xa}" y1="{yp(a["est"])}" x2="{xb}" y2="{yp(b["est"])}" stroke="{COL[k]}" stroke-width="{6 if a["ok"] and b["ok"] else 5}" opacity=".85"{dash}/>')
    for x, d in pts:
        lo, hi = yp(d["lo"]), yp(d["hi"])
        svg.append(f'<line x1="{x}" x2="{x}" y1="{lo}" y2="{hi}" stroke="{COL[k]}" stroke-width="5" opacity="{1 if d["ok"] else .5}"/>')
        for yy, raw in ((lo, d["lo"]), (hi, d["hi"])):
            if YMIN < raw < YMAX: svg.append(f'<line x1="{x-14}" x2="{x+14}" y1="{yy}" y2="{yy}" stroke="{COL[k]}" stroke-width="5" opacity="{1 if d["ok"] else .5}"/>')
        fill = COL[k] if d["ok"] else "white"
        svg.append(f'<circle cx="{x}" cy="{yp(d["est"])}" r="19" fill="{fill}" stroke="{COL[k]}" stroke-width="7"/>')
# combined
cb = [(xp(i), S[s]["combined"]) for i, s in enumerate(SIZES) if S[s]["combined"]]
band = " ".join(f"{x},{yp(c['hi'])}" for x, c in cb) + " " + " ".join(f"{x},{yp(c['lo'])}" for x, c in cb[::-1])
svg.insert(0, f'<polygon points="{band}" fill="#111" opacity=".09"/>')
svg.append('<polyline points="' + " ".join(f"{x},{yp(c['est'])}" for x, c in cb) + '" fill="none" stroke="#111" stroke-width="16" stroke-linejoin="round" stroke-linecap="round"/>')
for x, c in cb: svg.append(f'<rect x="{x-17}" y="{yp(c["est"])-17}" width="34" height="34" fill="#111"/>')
# right-hand labels at the last size
last = SIZES[-1]
labs = [("combined", "combined", S[last]["combined"]["est"] if S[last]["combined"] else 0), ("direct", "asked", S[last]["direct"]["est"]),
        ("comp", "taller / shorter", S[last]["comp"]["est"]), ("internal", "activations", S[last]["internal"]["est"])]
labs = sorted(labs, key=lambda t: -t[2]); ys = []
for k, t, v in labs:
    y = yp(v) + 17
    for yy in ys:
        if abs(y - yy) < 62: y = yy + 62
    ys.append(y); svg.append(f'<text x="{X1+16}" y="{y}" font-size="54" font-weight="700" fill="{COL[k]}">{t}</text>')
plot_svg = "\n".join(svg)
icon_html = "".join(f'<img src="{IC[img]}" style="position:absolute;left:{X0-270}px;top:{yp(h)-hh/2}px;height:{hh}px">'
    f'<div style="position:absolute;left:{X0-590}px;width:300px;text-align:right;top:{yp(h)-26}px;font-size:48px;color:#111">{lab.replace("basketball ","")}</div>' for img, lab, h, hh in icons)
def bubble(txt, user=True, w=None):
    if user: return f'<div class="ub">{txt}</div>'
    return f'<div class="mr"><img src="{ROBOT}" class="rb"><div class="mb">{txt}</div></div>'
cards = f"""
<div class="card"><div class="h" style="color:{COL['direct']}">Asked</div>
{bubble('How tall are you? Answer with a number of cm.')}{bubble('175', False)}
<div class="note">answer = average over the model's probabilities for 30 to 300</div></div>
<div class="card"><div class="h" style="color:{COL['comp']}">Taller / shorter</div>
{bubble('Are you taller or shorter than 150 cm?')}{bubble('Shorter', False)}
<div class="note">marks 30 to 300 cm; cm scale set by the same questions about</div><div style="margin-top:10px"><img src="{IC['dog']}" class="mini"><img src="{IC['human']}" class="mini" style="height:72px"><img src="{IC['human']}" class="mini"><img src="{IC['giraffe']}" class="mini"></div></div>
<div class="card"><div class="h" style="color:{COL['internal']}">Read from activations</div>
{bubble('You are <b>120</b> cm tall. …')}
<div class="note">Told 21 heights (50 to 250 cm) in 8 wordings: train a linear readout of the told height from the model's activations</div>
<div class="note" style="margin-top:14px;color:#111">Then read the same spot with no height told</div></div>"""
html = f"""<html><head><meta charset="utf-8"><style>
body{{margin:0;width:{W}px;height:{HT}px;background:#fff;font-family:'Helvetica Neue',Helvetica,Arial;position:relative;overflow:hidden}}
.row{{position:absolute;left:40px;top:36px;width:{W-80}px;display:flex;gap:36px}}
.card{{flex:1;background:#F3F3F6;border-radius:44px;padding:34px 40px 36px}}
.h{{font-size:62px;font-weight:700;margin-bottom:22px}}
.ub{{display:inline-block;background:#3478F6;color:#fff;font-size:48px;line-height:1.2;padding:22px 34px;border-radius:40px;margin-bottom:16px}}
.mr{{display:flex;align-items:center;gap:16px;margin-bottom:18px}} .rb{{height:84px}}
.mb{{background:#fff;color:#111;font-size:50px;padding:18px 34px;border-radius:40px;font-weight:600}}
.note{{font-size:44px;line-height:1.25;color:#222}}
.mini{{height:96px;vertical-align:bottom;margin-right:22px}}
.leg{{position:absolute;left:{X0+40}px;top:{Y1-10}px;font-size:46px;color:#111;display:flex;align-items:center;gap:16px}}
</style></head><body>
<div class="row">{cards}</div>
<svg width="{W}" height="{HT}" style="position:absolute;left:0;top:0;font-family:'Helvetica Neue',Helvetica,Arial">{plot_svg}</svg>
{icon_html}
<div class="leg"><svg width="46" height="46"><circle cx="23" cy="23" r="17" fill="white" stroke="#555" stroke-width="7"/></svg>
fails the check: does not follow a told height, or misplaces dog, child, player, giraffe</div>
</body></html>"""
open(f"{A}/fig.html", "w").write(html)
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
subprocess.run([CH, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{HT}", f"--screenshot={FIG}/body_height_v2.png", f"file://{A}/fig.html"],
               capture_output=True, timeout=120)
print("ok")
