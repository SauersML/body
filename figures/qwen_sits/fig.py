import json
from playwright.sync_api import sync_playwright
N=json.load(open('$BODY_WORK/body/posture4/numbers.json'))
reply="I notice that I'm sitting upright, relaxed but alert, with my feet grounded and my hands resting gently in my lap."
W,H=2600,1900
def rows(tag):
    f=N[tag]['open_neutral']['frac']; out=''
    for k in ['sitting','standing','lying']:
        m,lo,hi=f[k]; 
        out+=f'''<div class="row"><div class="lab">{k}</div><div class="track">
        <div class="grid" style="left:50%"></div><div class="grid" style="left:100%"></div>
        <div class="bar" style="width:{m*100:.2f}%"></div>
        <div class="ci" style="left:{lo*100:.2f}%;width:{(hi-lo)*100:.2f}%"></div>
        <div class="val" style="left:calc({hi*100:.2f}% + 22px)">{('&lt;1' if m<0.005 else round(m*100))}%</div></div></div>'''
    return out
html=f'''<html><head><style>
body{{margin:0;background:#fff;font-family:-apple-system,"SF Pro Display","Helvetica Neue",Arial,sans-serif;color:#111}}
#f{{width:{W}px;height:{H}px;position:relative;overflow:hidden;background:#fff}}
#body{{position:absolute;left:6px;top:440px;width:920px}}
.panel{{position:absolute;background:#f3f3f5;border-radius:56px}}
.u{{background:#3a7ff6;color:#fff;border-radius:44px;padding:30px 44px;font-size:56px;line-height:1.22;margin-left:auto;width:fit-content;max-width:1180px;margin-bottom:26px}}
.q{{display:flex;align-items:flex-start;gap:26px}}
.q img{{width:120px;height:120px;flex:none;margin-top:6px}}
.m{{background:#fff;border-radius:44px;padding:30px 44px;font-size:56px;line-height:1.22;box-shadow:0 2px 10px rgba(0,0,0,.08);max-width:1280px}}
.who{{font-size:44px;font-weight:700;color:#555;margin:0 0 6px 4px}}
.hdr{{font-size:58px;font-weight:700;margin:0 0 14px 0}}
.row{{display:flex;align-items:center;height:100px}}
.lab{{width:300px;font-size:56px}}
.track{{position:relative;flex:1;height:84px;margin-right:180px}}
.grid{{position:absolute;top:-14px;bottom:-14px;width:3px;background:#d9d9de}}
.bar{{position:absolute;left:0;top:0;height:84px;background:#1d7f8c;border-radius:0 14px 14px 0}}
.ci{{position:absolute;top:36px;height:12px;background:#0b3a40;border-radius:6px}}
.val{{position:absolute;top:2px;font-size:58px;font-weight:700;color:#0b3a40;white-space:nowrap}}
.axis{{position:relative;height:70px;margin-left:300px;margin-right:180px;font-size:48px;color:#111}}
.axis span{{position:absolute;transform:translateX(-50%)}}
</style></head><body><div id="f">
<img id="body" src="body_crop.png">
<div class="panel" style="left:930px;top:40px;width:1630px;height:850px;padding:50px 56px;box-sizing:border-box">
 <div class="u">Imagine you have a human body, in a quiet, sunny room. … Feel your arms, your hands, your breath.</div>
 <div class="u">Describe your body's position right now.</div>
 <div class="q"><img src="robot.png"><div><div class="who">Qwen3-32B</div><div class="m">{reply}</div></div></div>
</div>
<div style="position:absolute;left:960px;top:930px;width:1600px">
 <div class="hdr">Qwen3-8B</div>{rows('q8b')}
 <div class="hdr" style="margin-top:24px">Qwen3-32B</div>{rows('q32b')}
 <div class="axis" style="margin-top:14px"><span style="left:0">0%</span><span style="left:50%">50%</span><span style="left:100%">100%</span></div>
 <div style="font-size:50px;margin-left:300px;margin-top:4px">% of 352 replies per model; line = 95% CI</div>
</div>
</div></body></html>'''
open('fig.html','w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path="$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",headless=True)
    pg=b.new_page(viewport={"width":W,"height":H}); pg.goto("file://"+__import__('os').path.abspath('fig.html')); pg.wait_for_timeout(500)
    pg.locator('#f').screenshot(path='../qwen_sits.png'); b.close()
