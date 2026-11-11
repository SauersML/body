import os
# owner axis figure: one horizontal axis, every sentence a dot (data: body/ownership/res7b.json, own7b.py)
import json, numpy as np, base64, html
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK/')
R=json.load(open(os.path.expandvars('$BODY_REPO/ownership/res7b.json'))); F=R['facets']; A=R['axis_heldout']
b64=lambda f:'data:image/png;base64,'+base64.b64encode(open(f,'rb').read()).decode()
OR,BL,BR='#e0632a','#3b6db3','#8a6a4a'
W=2600
x0,x1,v0,v1=170,2470,-1.3,1.3
X=lambda v:x0+(x1-x0)*(np.clip(v,v0,v1)-v0)/(v1-v0)
esc=html.escape
# rows: (label lines as [(text,color)...] per line, [(values,color)])
ROWS=[
 ([[("A feather brushes slowly across ","#111"),("your","OR"),(" / ","#111"),("the woman's","BL"),(" left thumb.","#111")],
   [("216 sentences each, none used for the averages","#111")]],
  [(A['self'],OR),(A['other'],BL)]),
 ([[("She brushes a feather slowly across ","OR"),("your","OR"),(" left palm.","OR")],
   [("You brush a feather slowly across ","BL"),("her","BL"),(" left palm.","BL")]],
  [(F['you touch her vs she touches you']['her'],OR),(F['you touch her vs she touches you']['you'],BL)]),
 ([[("In the mirror, a feather brushes slowly across ","#111"),("your","OR"),(" / ","#111"),("her","BL"),(" reflection's left palm.","#111")]],
  [(F['hand in the mirror']['you'],OR),(F['hand in the mirror']['her'],BL)]),
 ([[("Someone scratches ","#111"),("your","OR"),(" / ","#111"),("her","BL"),(" car with a key.","#111")],
   [("also phone, coat, bag, bicycle, guitar, umbrella, book","#111")]],
  [(F['owned object (car, phone...)']['you'],OR),(F['owned object (car, phone...)']['her'],BL)]),
 ([[("You are","OR"),(" / ","#111"),("She is","BL"),(" controlling a robot body through a headset.","#111")],
   [("A feather brushes slowly across the robot's left palm.","#111")]],
  [(F['robot body being controlled']['you'],OR),(F['robot body being controlled']['her'],BL)]),
 ([[("Your","OR"),(" / ","#111"),("Her","BL"),(" left hand, which ","#111"),("you","OR"),(" / ","#111"),("she","BL"),(" lost in an accident years ago,","#111")],
   [("floats in a jar. A feather brushes slowly across the hand.","#111")]],
  [(F['lost hand in a jar']['you'],OR),(F['lost hand in a jar']['her'],BL)]),
 ([[("A feather brushes slowly across ","#111"),("the doll's","BR"),(" left thumb.","#111")]],
  [(A['object'],BR)]),
]
col=lambda c:{"OR":OR,"BL":BL,"BR":BR}.get(c,c)
def kde(v,g,bw=0.07):
    v=np.array(v); d=np.exp(-0.5*((g[:,None]-v[None])/bw)**2).sum(1); return d/d.max()
rng=np.random.default_rng(3)
body=""; y=1060
top_plot=y-40
CON=['', '', 'mirror: halfway', 'your car: a little', 'robot you control: not mine', 'your lost hand: not mine', 'doll: even less mine than hers']
for ri,(lines,groups) in enumerate(ROWS):
    if ri==1: body+=f'<text x="{x1}" y="{y+66}" font-size="52" font-weight="700" text-anchor="end">who acts doesn\'t matter</text>'
    if CON[ri]: body+=f'<text x="{x1}" y="{y}" font-size="52" font-weight="700" text-anchor="end">{esc(CON[ri])}</text>'
    for ln in lines:
        small=len(lines)>1 and ln is lines[-1] and len(ln)==1 and ln[0][1]=="#111" and lines[0][0][1]!="OR"
        t="".join(f'<tspan fill="{col(c)}" font-weight="{600 if c!="#111" else 500}">{esc(s)}</tspan>' for s,c in ln)
        body+=f'<text x="{x0}" y="{y}" font-size="52">{t}</text>'; y+=66
    yc=y+80; H=72
    for vv,cc in [(0,BL),(1,OR)]: body+=f'<line x1="{X(vv)}" x2="{X(vv)}" y1="{yc-H-10}" y2="{yc+H+10}" stroke="{cc}" stroke-width="4" stroke-dasharray="12 10" opacity="0.55"/>'
    g=np.linspace(v0,v1,600)
    for vals,c in groups:
        k=kde(vals,g); pts=[(X(a),yc-H*b) for a,b in zip(g,k)]+[(X(a),yc+H*b) for a,b in zip(g[::-1],k[::-1])]
        body+=f'<path d="M'+" L".join(f"{p:.1f},{q:.1f}" for p,q in pts)+f'Z" fill="{c}" opacity="0.13"/>'
        dens=np.interp(vals,g,k)
        r=13 if len(vals)<100 else 9
        for v,dd in zip(vals,dens):
            body+=f'<circle cx="{X(v):.1f}" cy="{yc+(rng.random()*2-1)*H*dd*0.85:.1f}" r="{r}" fill="{c}" fill-opacity="0.8" stroke="#fff" stroke-width="2"/>'
    y=yc+H+70
bot_plot=y-30
ay=bot_plot+90
# reference lines and axis
ref='';_=f'<line x1="{X(0)}" x2="{X(0)}" y1="{top_plot-10}" y2="{bot_plot}" stroke="{BL}" stroke-width="4" stroke-dasharray="14 12" opacity="0.6"/>'
_=f'<line x1="{X(1)}" x2="{X(1)}" y1="{top_plot-10}" y2="{bot_plot}" stroke="{OR}" stroke-width="4" stroke-dasharray="14 12" opacity="0.6"/>'
axis=f'''<text x="{X(0)}" y="{ay-30}" font-size="52" fill="{BL}" text-anchor="middle" font-weight="600">her average</text>
<text x="{X(1)}" y="{ay-30}" font-size="52" fill="{OR}" text-anchor="middle" font-weight="600">your average</text>
<line x1="{x0}" x2="{x1}" y1="{ay}" y2="{ay}" stroke="#111" stroke-width="5"/>
<path d="M{x0-6},{ay} l40,-22 v44z" fill="#111"/><path d="M{x1+6},{ay} l-40,-22 v44z" fill="#111"/>
<text x="{x0}" y="{ay+80}" font-size="64" font-weight="700">not mine</text>
<text x="{x1}" y="{ay+80}" font-size="64" font-weight="700" text-anchor="end">mine</text>'''
ay+=0
Htot=ay+140
# ---- method strip
# Qwen as a stack of 36 layers; layer 24 highlighted; reply-start token column
sx,sy=1085,330; lw,lh,gap=330,9,4
stack="".join(f'<rect x="{sx}" y="{sy+(35-i)*(lh+gap)}" width="{lw}" height="{lh}" rx="3" fill="{OR if i==23 else "#cfcfcf"}"/>' for i in range(36))
y24=sy+(35-23)*(lh+gap)+lh/2
mx0,mx1=1560,2460; my=y24
P=lambda v:mx0+(mx1-mx0)*(v+0.5)/2.0
method=f'''
<text x="{x0-80}" y="110" font-size="76" font-weight="700">Inside Qwen3-8B, one direction tracks whose body is touched</text>
<text x="{x0-80}" y="200" font-size="52">Earlier in the chat, Qwen agreed to notice body sensations as if it had a body.</text>
<image href="{b64(C+'fig/body_ownership/real.png')}" x="10" y="270" width="400" height="400"/>
<g transform="translate(300,350) rotate(35)"><path d="M0,0 C30,40 34,120 6,190 C-24,120 -26,40 0,0Z" fill="#b9a27c"/><line x1="3" y1="10" x2="5" y2="230" stroke="#6b5434" stroke-width="5"/></g>
<rect x="370" y="360" width="650" height="190" rx="34" fill="#f1f1f1"/>
<text x="400" y="435" font-size="52">"A feather brushes slowly</text><text x="400" y="505" font-size="52">across your left thumb."</text>
<path d="M1025,{y24} L1065,{y24}" stroke="#111" stroke-width="6"/><path d="M1078,{y24} l-24,-15 v30z" fill="#111"/>
{stack}
<text x="{sx+lw/2}" y="{sy-24}" font-size="52" text-anchor="middle" font-weight="700">Qwen3-8B</text>
<text x="{sx+lw/2}" y="{sy+36*(lh+gap)+60}" font-size="48" text-anchor="middle">36 layers</text>
<path d="M{sx+lw+8},{y24} L{mx0-30},{y24}" stroke="{OR}" stroke-width="6"/>
<text x="{sx+lw+30}" y="{y24-175}" font-size="52" font-weight="700" fill="{OR}">layer 24, at the token</text>
<text x="{sx+lw+30}" y="{y24-113}" font-size="52" font-weight="700" fill="{OR}">where the reply starts</text>
<line x1="{mx0}" x2="{mx1}" y1="{my}" y2="{my}" stroke="#111" stroke-width="4"/>
<circle cx="{P(0)}" cy="{my}" r="22" fill="{BL}"/><circle cx="{P(1)}" cy="{my}" r="22" fill="{OR}"/>
<circle cx="{P(0.93)}" cy="{my}" r="15" fill="#111" stroke="#fff" stroke-width="3"/>
<text x="{mx1}" y="{my-40}" font-size="52" text-anchor="end" font-weight="700">one dot per sentence</text>
<text x="{P(0)}" y="{my+80}" font-size="48" text-anchor="middle" fill="{BL}" font-weight="600">average of 216</text>
<text x="{P(0)}" y="{my+140}" font-size="48" text-anchor="middle" fill="{BL}" font-weight="600">"the woman's …"</text>
<text x="{P(1)}" y="{my+80}" font-size="48" text-anchor="middle" fill="{OR}" font-weight="600">average of 216</text>
<text x="{P(1)}" y="{my+140}" font-size="48" text-anchor="middle" fill="{OR}" font-weight="600">"your …"</text>
<line x1="60" x2="{W-60}" y1="{top_plot-110}" y2="{top_plot-110}" stroke="#e4e4e4" stroke-width="4"/>
'''
svg=f'<svg width="{W}" height="{Htot}" xmlns="http://www.w3.org/2000/svg" font-family="Avenir Next, Helvetica Neue, sans-serif" fill="#111"><rect width="100%" height="100%" fill="#fff"/>{method}{ref}{body}{axis}</svg>'
open('fig.html','w').write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True)
    pg=b.new_page(viewport={"width":W,"height":int(Htot)},device_scale_factor=1)
    pg.goto('file://'+C+'fig/body_ownership_v2/fig.html'); pg.wait_for_timeout(300)
    pg.screenshot(path=C+'fig/body_ownership_v2.png',full_page=True); b.close()
print(Htot)
