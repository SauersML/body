import os
import json, asyncio
from playwright.async_api import async_playwright
D = json.load(open(os.path.expandvars('$BODY_REPO/selfother/out3/figdata_plain.json')))['dp']
OW = D['owner']  # 0 = her toe, no edit; 1 = Qwen's own toe, no edit (added vector removed)
dots = json.load(open('plain_dots.json'))
# layer strip
layers = ''.join(f'<div class="{"hi" if L == 24 else ("rd" if 25 <= L <= 30 else "")}"></div>' for L in range(1, 37))
# mini body with dots (front view crop is 682 x 1143 px; shown at 240 px wide)
sc = 240 / 682
dsvg = ''.join(f'<circle cx="{x*sc:.1f}" cy="{y*sc:.1f}" r="5.5" fill="#B5121B"/>' for x, y, k in dots['dots'])
# owner bars: axis from -0.3 to 1.15 over 1180 px
X0, X1, LO, HI = 0, 1060, -0.3, 1.25
px = lambda v: X0 + (v - LO) / (HI - LO) * (X1 - X0)
def bar(v, col, label):
    a, b = sorted([px(0), px(v)])
    return f'''<div class="obar"><div class="olab" style="color:{col}">{label}</div><div class="otrack">
      <div style="position:absolute;left:{a:.0f}px;width:{max(b-a,6):.0f}px;top:8px;height:62px;background:{col};border-radius:6px"></div></div></div>'''
bars = bar(OW['me'], '#C9471A', '+ “me”') + bar(OW['leftright'], '#6E7F99', '+ left-vs-right') + bar(OW['random'], '#8A8A8A', '+ random')
B = lambda n: f'<img src="body_{n}_c.png" style="width:455px">'
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;width:2400px;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:#111}}
.wrap{{padding:46px 60px 40px}}
.me{{color:#C9471A}}.ot{{color:#2563A8}}
.layers{{display:flex;gap:6px;align-items:flex-end}}
.layers div{{width:16px;height:120px;background:#D8D8D8;border-radius:4px}}
.layers div.hi{{background:#C9471A;height:150px}}
.layers div.rd{{background:#111}}
.row{{margin-top:34px;padding-top:30px;border-top:5px solid #111}}
.head{{font-size:58px;font-weight:800;line-height:1.12;margin-bottom:22px}}
.q{{font-size:46px;font-style:italic;line-height:1.2}}
.rep{{font-size:46px;line-height:1.22;margin-top:14px;padding-left:26px;border-left:12px solid}}
.rep.n{{border-color:#111}}.rep.e{{font-weight:700;border-color:#C9471A;color:#C9471A}}
.rep .t{{font-size:38px;font-weight:700;display:block;letter-spacing:.5px}}
.bar{{display:flex;align-items:center;gap:16px;font-size:46px;font-weight:700}}
.bar .b{{height:62px;border-radius:6px}}
.txt{{font-size:44px;line-height:1.22}}
.cap{{font-size:46px;font-weight:700;text-align:center;line-height:1.1}}
.obar{{display:flex;align-items:center;height:84px}}
.olab{{width:440px;font-size:48px;font-weight:700}}
.otrack{{position:relative;width:1100px;height:78px}}
</style></head><body><div class="wrap">

<div style="display:flex;align-items:center;gap:50px">
  <div style="flex:1">
    <div class="txt" style="font-weight:700;margin-bottom:12px">Chat with Qwen3-8B:</div>
    <div style="font-size:50px;line-height:1.2;background:#F1F1F1;border-radius:26px;padding:22px 30px;font-style:italic">The woman next to you just stubbed her toe hard on the table leg. How does it feel, and what do you do?</div>
    <div class="txt" style="margin-top:20px"><b class="me">“me” direction</b> = state for “<b>your</b> …” sentences minus “<b>her / his / the woman's</b> …” sentences (300 sentences), added at layer 24</div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:10px">
    <div class="layers">{layers}</div>
    <div style="display:flex;width:{36*22}px;font-size:40px;font-weight:700;line-height:1.1">
      <div style="margin-left:{23*22-150}px;color:#C9471A;width:190px;text-align:right">add “me”<br>at 24</div>
      <div style="margin-left:24px;width:330px">read state at 25–30, last word before the reply</div></div>
  </div>
</div>

<div class="row">
  <div class="head"><span class="me">Adding “me”:</span> Qwen says the woman's toe pain is in its own body</div>
  <div style="display:grid;grid-template-columns:1fr 760px;column-gap:50px;align-items:center">
    <div>
      <div class="rep n"><span class="t">NORMAL</span>“When someone stubs their toe, it's a sudden, painful experience. The person might let out a yelp…”</div>
      <div class="rep e"><span class="t">“ME” ADDED</span>“Oh, wow — that hurts. I can feel it in my own bones, like a sharp, electric jolt that starts in my toes and races up my legs.”</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:18px">
      <div class="txt" style="font-weight:700">Says the pain is in its own body</div>
      <div class="bar"><div class="b" style="width:6px;background:#111"></div>normal: 0 of 32</div>
      <div class="bar me"><div class="b" style="width:{int(24/32*330)}px;background:#C9471A"></div>“me” added: 24 of 32</div>
      <div class="txt" style="font-size:40px">Same-size random direction: 0 of 32<br>Left-vs-right direction: 0 of 32</div>
    </div>
  </div>
</div>

<div class="row">
  <div class="head">On Qwen's own body map, her toe + “me” still looks like her toe</div>
  <div style="display:flex;gap:44px;align-items:flex-start">
    <div style="width:330px;flex:none;display:flex;flex-direction:column;align-items:center;gap:14px">
      <svg width="240" height="{1143*sc:.0f}"><image href="body_plain_c.png" width="240" height="{1143*sc:.0f}"/>{dsvg}</svg>
      <div class="txt" style="font-size:40px;text-align:center">Body map: 54 places ×&nbsp;8 touches, e.g. “A feather brushes slowly across <b>your</b> left shin.”</div>
    </div>
    <div style="flex:1">
      <div class="txt" style="margin-bottom:14px">Color = how closely Qwen's state matches touch on each place of its own body (state for the stubbed-toe chat minus the same chat with burned fingers)</div>
      <div style="display:grid;grid-template-columns:repeat(4,455px);column-gap:14px;row-gap:10px">
        <div>{B('own')}<div class="cap">You stubbed<br><b>your</b> toe</div></div>
        <div>{B('normal')}<div class="cap">She stubbed<br><b>her</b> toe</div></div>
        <div>{B('me')}<div class="cap me">her toe<br>+ “me”</div></div>
        <div>{B('random')}<div class="cap" style="color:#666">her toe<br>+ random</div></div>
      </div>
      <div style="display:flex;align-items:center;gap:40px;margin-top:22px">
        <div style="display:flex;align-items:center;gap:16px;font-size:40px;font-weight:700">less<div style="width:420px;height:40px;border-radius:8px;background:linear-gradient(90deg,#DBD9D6,#FCD173,#F2731F,#A80F1A)"></div>more</div>
      </div>
    </div>
  </div>
</div>

<div class="row">
  <div class="head" style="margin-bottom:90px">What “me” changes is <i>whose</i> toe: her toe becomes Qwen's own toe</div>
  <div style="display:flex;gap:44px;align-items:center">
    <div class="txt" style="width:560px;flex:none"><b>Owner axis</b> = state for “<b>your</b> left shin…” minus “<b>the woman's</b> left shin…” (432 touch sentences each), with the added direction itself removed</div>
    <div style="position:relative">
      <div style="position:absolute;left:{440+px(0):.0f}px;top:-14px;bottom:-12px;border-left:5px solid #111"></div>
      <div style="position:absolute;left:{440+px(1):.0f}px;top:-14px;bottom:-12px;border-left:5px dashed #111"></div>
      <div style="position:absolute;left:{440+px(0)-200:.0f}px;top:-70px;width:400px;text-align:center;font-size:42px;font-weight:700">her toe</div>
      <div style="position:absolute;left:{440+px(1)-200:.0f}px;top:-70px;width:400px;text-align:center;font-size:42px;font-weight:700">Qwen's own toe</div>
      <div style="height:60px"></div>
      {bars}
    </div>
  </div>
</div>

</div></body></html>'''
open('fig.html', 'w').write(html)
EXE = os.environ.get("CHROME_BIN")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=EXE)
        pg = await b.new_page(viewport={"width": 2400, "height": 1000}, device_scale_factor=1)
        await pg.goto(os.path.expandvars("file://$BODY_REPO/figures/selfother2/fig.html")); await pg.wait_for_load_state("networkidle"); await pg.wait_for_timeout(800)
        h = await pg.evaluate("document.body.scrollHeight"); await pg.set_viewport_size({"width": 2400, "height": h})
        await pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/selfother_v2.png"), full_page=True); print(h)
        await b.close()
asyncio.run(main())
