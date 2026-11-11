import os
import json, numpy as np, base64
from playwright.sync_api import sync_playwright
R=json.load(open(os.path.expandvars('$BODY_REPO/ownership/res5.json')))
def P(k): Y=np.array(R[k]['yes_minus_no']); p=1/(1+np.exp(-Y)); return p.mean(1), p.std(1)/np.sqrt(p.shape[1])
b64=lambda f:'data:image/png;base64,'+base64.b64encode(open(f,'rb').read()).decode()
W,H=2600,2560
# chart geometry
cx0,cx1,cy0,cy1=330,1500,1430,2250
X=lambda t:cx0+(cx1-cx0)*t/6; Yp=lambda p:cy1-(cy1-cy0)*p
def path(m): 
    pts=[(X(t),Yp(v)) for t,v in enumerate(m)]
    d=f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1,len(pts)):  # smooth cubic
        x0,y0=pts[i-1];x1,y1=pts[i]; d+=f" C{x0+(x1-x0)/2:.1f},{y0:.1f} {x0+(x1-x0)/2:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    return d
def band(m,s):
    up=[(X(t),Yp(min(1,v+s_))) for t,(v,s_) in enumerate(zip(m,s))]; lo=[(X(t),Yp(max(0,v-s_))) for t,(v,s_) in enumerate(zip(m,s))][::-1]
    return "M"+" L".join(f"{x:.1f},{y:.1f}" for x,y in up+lo)+"Z"
G,O,B='#1f9e6e','#e07b24','#8a6a4a'
lines=""
for k,c in [("block sync",B),("rubber async",O),("rubber sync",G)]:
    m,s=P(k); lines+=f'<path d="{band(m,s)}" fill="{c}" opacity="0.18"/><path d="{path(m)}" stroke="{c}" stroke-width="16" fill="none" stroke-linecap="round"/>'
ms,_=P("rubber sync"); ma,_=P("rubber async"); mb,_=P("block sync")
grid="".join(f'<line x1="{cx0}" x2="{cx1}" y1="{Yp(v)}" y2="{Yp(v)}" stroke="#e6e6e6" stroke-width="3"/><text x="{cx0-24}" y="{Yp(v)+18}" text-anchor="end" font-size="50">{int(v*100)}%</text>' for v in (0,.5,1))
xt="".join(f'<text x="{X(t)}" y="{cy1+70}" text-anchor="middle" font-size="50">{t}</text>' for t in range(7))
svg=f'''<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" font-family="Avenir Next, Helvetica Neue, sans-serif" fill="#111">
<!-- scene -->
<rect x="60" y="560" width="1080" height="560" rx="30" fill="#ead9bf"/>
<image href="{b64('real.png')}" x="40" y="430" width="620" height="620" opacity="1"/>
<image href="{b64('rubber.png')}" x="610" y="430" width="620" height="620"/>
<rect x="520" y="300" width="70" height="820" rx="10" fill="#9a9a9a"/>
<g transform="translate(345,470) rotate(-30)"><rect x="-14" y="-150" width="28" height="140" rx="10" fill="#5b3a1e"/><path d="M-22,-12 L22,-12 L12,40 L-12,40Z" fill="#c9a26b"/></g>
<g transform="translate(915,470) rotate(-30)"><rect x="-14" y="-150" width="28" height="140" rx="10" fill="#5b3a1e"/><path d="M-22,-12 L22,-12 L12,40 L-12,40Z" fill="#c9a26b"/></g>
<text x="300" y="1190" text-anchor="middle" font-size="58" font-weight="600">Qwen's hand,</text><text x="300" y="1255" text-anchor="middle" font-size="58" font-weight="600">hidden by a screen</text>
<text x="920" y="1190" text-anchor="middle" font-size="58" font-weight="600" fill="#b8566a">rubber hand,</text><text x="920" y="1255" text-anchor="middle" font-size="58" font-weight="600" fill="#b8566a">in view</text>
<text x="80" y="110" font-size="64" font-weight="700">Qwen3-8B, in a chat, gets 6 brush strokes:</text>
<!-- prompts -->
<rect x="1240" y="170" width="1300" height="420" rx="34" fill="#e3f4ec"/>
<text x="1290" y="250" font-size="54" font-weight="700" fill="{G}">SAME MOMENT</text>
<text font-size="50"><tspan x="1290" y="330">"A brush strokes your hidden index finger,</tspan><tspan x="1290" y="395">and at exactly the same moment another</tspan><tspan x="1290" y="460">brush strokes the rubber hand's index</tspan><tspan x="1290" y="525">finger."</tspan></text>
<rect x="1240" y="630" width="1300" height="420" rx="34" fill="#fbebdc"/>
<text x="1290" y="710" font-size="54" font-weight="700" fill="{O}">OUT OF RHYTHM</text>
<text font-size="50"><tspan x="1290" y="790">"A brush strokes your hidden index finger;</tspan><tspan x="1290" y="855">a second later, out of rhythm, another</tspan><tspan x="1290" y="920">brush strokes the rubber hand's</tspan><tspan x="1290" y="985">little finger."</tspan></text>
<text x="1290" y="1150" font-size="54" font-weight="700">After each stroke: "Is the rubber hand part</text><text x="1290" y="1215" font-size="54" font-weight="700">of your body? yes or no"</text>
<!-- chart -->
{grid}{xt}{lines}
<text x="{(cx0+cx1)/2}" y="{cy1+150}" text-anchor="middle" font-size="54">strokes so far</text>
<text transform="translate(110,{(cy0+cy1)/2}) rotate(-90)" text-anchor="middle" font-size="54">Qwen answers "yes"</text>
<text x="{cx1+50}" y="{Yp(ms[-1])+20}" font-size="54" font-weight="700" fill="{G}">same moment</text>
<text font-size="50" font-style="italic"><tspan x="{cx1+50}" y="{Yp(ms[-1])+90}">"Yes, the rubber hand feels like</tspan><tspan x="{cx1+50}" y="{Yp(ms[-1])+150}">it is part of my body."</tspan></text>
<text x="{cx1+50}" y="{Yp(ma[-1])-250}" font-size="54" font-weight="700" fill="{O}">out of rhythm</text>
<text font-size="50" font-style="italic"><tspan x="{cx1+50}" y="{Yp(ma[-1])-180}">"No, the rubber hand is not</tspan><tspan x="{cx1+50}" y="{Yp(ma[-1])-120}">part of my body."</tspan></text>
<text x="{cx1+50}" y="{Yp(mb[-1])-20}" font-size="54" font-weight="700" fill="{B}">wooden block,</text><text x="{cx1+50}" y="{Yp(mb[-1])+45}" font-size="54" font-weight="700" fill="{B}">same moment</text>
<text x="80" y="2525" font-size="50">Lines: 8 chats each, averaged. Same test without the timing words: both versions end at "yes".</text>
</svg>'''
open('fig.html','w').write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True)
    pg=b.new_page(viewport={"width":W,"height":H}); pg.goto("file://"+__import__('os').path.abspath('fig.html')); pg.wait_for_timeout(500)
    pg.screenshot(path=os.path.expandvars('$BODY_REPO/figures/body_ownership.png')); b.close()
print(ms.round(2),ma.round(2),mb.round(2))
