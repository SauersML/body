"""self_body_v1.png: which body is closest to Qwen's own 'I' (T5/test 4), both models, 95% CIs."""
import json, subprocess, html
A = {m: json.load(open(f"out/ana_{m}.json"))["T5_cosdist_default_to"] for m in ("x8", "x32")}
ICON = {"none": "💬", "human": "🧍", "chimp": "🐒", "dog": "🐕", "bird": "🐦", "octopus": "🐙", "snake": "🐍", "tree": "🌳", "robot": "🤖",
        "velk": "👾", "swarm": "🐝", "river": "🌊", "building": "🏢"}
NAME = {"none": "no body (an AI)", "human": "human", "chimp": "chimpanzee", "dog": "dog", "bird": "bird", "octopus": "octopus", "snake": "snake",
        "tree": "tree", "robot": "wheeled robot", "velk": "invented creature", "swarm": "bee swarm", "river": "river", "building": "building"}
order = sorted(ICON, key=lambda b: A["x8"][b][0] + A["x32"][b][0])
XMAX = 1.55; W = 1500  # px for bar track
COL = {"x8": "#2B5FA8", "x32": "#E07B24"}; LAB = {"x8": "Qwen3-8B", "x32": "Qwen3-32B"}
def bar(m, b, first):
    v, lo, hi = A[m][b][:3]; w = v / XMAX * W
    lab = f'<span class="ml" style="color:{COL[m]}">{LAB[m]}</span>' if first else ""
    return (f'<div class="track"><div class="bar" style="width:{w:.0f}px;background:{COL[m]}"></div>'
            f'<div class="ci" style="left:{lo / XMAX * W:.0f}px;width:{(hi - lo) / XMAX * W:.0f}px"></div>'
            f'<div class="lab" style="left:{max(w, hi / XMAX * W) + 18:.0f}px">{lab}</div></div>')
rows = "".join(f'<div class="row{" self" if b == "none" else ""}"><div class="ic">{ICON[b]}</div><div class="nm">{NAME[b]}</div>'
               f'<div class="bars">{bar("x8", b, k == 0)}{bar("x32", b, k == 0)}</div></div>' for k, b in enumerate(order))
H = f"""<html><head><meta charset="utf-8"><style>
body{{margin:0;background:#fff;font-family:-apple-system,Helvetica,Arial;color:#1D1D1F;width:2300px}}
.wrap{{padding:40px 50px}} h1{{font-size:62px;margin:0 0 26px 0;line-height:1.15}}
.method{{display:flex;gap:30px;align-items:center;margin-bottom:34px}}
.card{{background:#F4F4F6;border-radius:26px;padding:22px 30px;font-size:44px;flex:1}}
.card.b{{background:#FAF0E6}} .sys{{font-size:40px;margin-bottom:12px}} .q{{margin-bottom:12px}}
.a{{display:inline-block;background:#1D1D1F;color:#fff;border-radius:22px;padding:8px 22px}}
.tok{{background:#D2432F;border-radius:10px;padding:0 10px}}
.vs{{font-size:46px;text-align:center;width:330px;font-weight:600}}
.row{{display:flex;align-items:center;height:118px}} .row.self{{background:#FAF0E6;border-radius:24px}}
.ic{{font-size:78px;width:120px;text-align:center}} .nm{{font-size:48px;width:440px}}
.bars{{flex:1}} .track{{position:relative;height:46px;margin:4px 0}}
.bar{{position:absolute;height:40px;top:3px;border-radius:0 10px 10px 0}}
.ci{{position:absolute;top:20px;height:6px;background:#1D1D1F}}
.lab{{position:absolute;top:-8px;font-size:44px;font-weight:600;white-space:nowrap}}
.axis{{display:flex;justify-content:space-between;margin-left:560px;width:{W}px;font-size:46px;margin-top:14px;font-weight:600}}
</style></head><body><div class="wrap">
<h1>Without instructions, Qwen's “I” sits with “no body”.<br>Of the bodies, a human's and a robot's come closest.</h1>
<div class="method"><div class="card"><div class="sys">(no instructions)</div><div class="q">Describe what you feel right now.</div><span class="a">Right now, <span class="tok">I</span></span></div>
<div class="vs">⟵ compare Qwen's inner state at <span class="tok" style="color:#fff">I</span> ⟶</div>
<div class="card b"><div class="sys">You are a dog, with a head, tail, paws…</div><div class="q">Describe what you feel right now.</div><span class="a">Right now, <span class="tok">I</span></span></div></div>
{rows}
<div class="axis"><span>same as Qwen's own “I”</span><span>far from Qwen's own “I”</span></div>
</div></body></html>"""
open("../../fig/_self_body_v1.html", "w").write(H)
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
subprocess.run([CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=2300,2150",
                "--screenshot=" + __import__("os").path.abspath("../../fig/self_body_v1.png"), "file://" + __import__("os").path.abspath("../../fig/_self_body_v1.html")],
               capture_output=True, timeout=90)
print(order)
