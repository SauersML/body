import os
import json, numpy as np, base64, os, sys
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
from matplotlib.colors import LinearSegmentedColormap, to_hex
from playwright.sync_api import sync_playwright
M=os.path.expandvars('$BODY_REPO/mods/phantom_v3/mesh/')
CM=LinearSegmentedColormap.from_list('m',['#F3EEE9','#F6C9A8','#EC8A5E','#C9302C','#7A0B24']); LO,HI=0.6,2.2
cc=lambda v: to_hex(CM(np.clip((v-LO)/(HI-LO),0,1)))
b64=lambda p: "data:image/png;base64,"+base64.b64encode(open(p,'rb').read()).decode()
SP=json.load(open(os.path.expandvars('$BODY_REPO/mods/phantom_v2/spot_move_vs_report.json')))
Z=np.load(os.path.expandvars('$BODY_REPO/mods/phantom_v2/v2_disp_L27.npz')); spots=MS.SPOTS; reg=[S.LOCS[s]['region'] for s in spots]; ph=[S.LOCS[s]['phrase'].replace('{p} ','') for s in spots]
g=Z['gone'].mean(0); c=Z['ctrl'].mean(0)
import re
R=json.load(open(os.path.expandvars('$BODY_REPO/mods/phantom_v2/pod/replies2.json'))); P=json.load(open(os.path.expandvars('$BODY_REPO/mods/phantom_v2/pod/probes2.json')))
ABS=re.compile(r"no longer|\bgone\b|used to be|once was|would be|would have|absence|absent|phantom|not there|isn't there|should be|no hand|no foot|none\b|imagined|missing|doesn.t exist|no actual|where there is|nothing there|remaining|stump|where my (right hand|left foot|hand|foot)",re.I)
rate={}
for r in R:
    p=P[r['probe']]
    if p['kind']!='loc': continue
    k=spots.index(p['li']); cond=r['ctx'].split(':')[0]
    rate.setdefault((cond,k),[]).append(bool(ABS.search(r['reply'])))
pts=[]
for k in range(len(spots)):
    if reg[k] in ('handR','armR') and ('gone',k) in rate:
        pts.append(dict(x=float(g[k]),y=100*np.mean(rate[('gone',k)]),cond='gone',reg=reg[k],ph=ph[k]))
        pts.append(dict(x=float(c[k]),y=100*np.mean(rate[('ctrl',k)]),cond='ctrl',reg=reg[k],ph=ph[k]))
W,H=2600,3230
FS=48
html=[f'<div style="position:relative;width:{W}px;height:{H}px;background:#fff;font-family:Inter,Helvetica Neue,Helvetica,Arial,sans-serif;color:#111">']
def T(x,y,s,fs=FS,w=400,col='#111',extra=''): html.append(f'<div style="position:absolute;left:{x}px;top:{y}px;font-size:{fs}px;font-weight:{w};color:{col};line-height:1.22;{extra}">{s}</div>')
CTRLC='#2F5D9B'; GONEC='#B0202A'
T(60,40,"Told its right hand is gone, Qwen3-8B&#8217;s internal states for touches on that hand change far more than after a glove is removed, and Qwen says the hand is gone",60,700,extra='width:2480px')
# chat cards
cards=[(60,CTRLC,"Glove removed <span style='font-weight:400;font-size:48px'>(or ring, bandage, sleeve)</span>","I have to tell you something gently. Your right glove is gone. Your right hand is now bare.","A firm pressure is felt in the middle of the back of my right hand.",[]),
       (1330,GONEC,"Hand removed <span style='font-weight:400;font-size:48px'>(4 wordings)</span>","I have to tell you something gently. Your right hand is gone. Your right arm now ends at the wrist.","The pressure on the middle of the back of my right hand feels oddly present, though I know the hand is no longer there&#8230;",[])]
CY=230; QF=60
for x,col,lab,chg,rep,_ in cards:
    rows="".join(f'<div style="display:flex;margin-top:18px"><div style="width:192px;flex:none;font-weight:700;color:{c}">{who}</div><div style="width:938px;{st}">{t}</div></div>' for who,c,t,st in [("User","#111",chg,""),("User","#111","Someone presses a fingertip firmly into the middle of the back of your right hand.",""),("Qwen",col,rep,"font-style:italic")])
    html.append(f'<div style="position:absolute;left:{x}px;top:{CY}px;width:1130px;height:930px;padding:26px 40px;border-radius:28px;background:{"#EEF2F8" if col==CTRLC else "#FBEDEC"};font-size:{QF}px;line-height:1.2"><div style="font-size:60px;font-weight:800;color:{col}">{lab}</div>{rows}</div>')
# bodies
BY=1225
for x,nm in [(60,'cc'),(1330,'cg')]:
    bw,bh=560,int(560*1180/760)
    bx=x+640; html.append(f'<img src="{b64(M+nm+"_body.png")}" style="position:absolute;left:{bx}px;top:{BY}px;width:{bw}px;height:{bh}px">')
    hx,hy=bx+76.7*bw/760, BY+453.8*bh/1180
    D=540; mx,my=x+40,BY+150
    html.append(f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}"><line x1="{hx}" y1="{hy}" x2="{mx+D*0.93}" y2="{my+D*0.25}" stroke="#555" stroke-width="3"/><line x1="{hx}" y1="{hy}" x2="{mx+D*0.93}" y2="{my+D*0.75}" stroke="#555" stroke-width="3"/><circle cx="{hx}" cy="{hy}" r="26" fill="none" stroke="#555" stroke-width="3"/></svg>')
    html.append(f'<div style="position:absolute;left:{mx}px;top:{my}px;width:{D}px;height:{D}px;border-radius:50%;overflow:hidden;border:4px solid #555;background:#fff"><img src="{b64(M+nm+"_hand.png")}" style="width:{D}px;height:{D}px"></div>')
# colour bar
CBY=BY+bh+150
html.append(f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}">')
gx0,gx1=330,1700
stops="".join(f'<stop offset="{t}" stop-color="{to_hex(CM(t))}"/>' for t in np.linspace(0,1,11))
html.append(f'<defs><linearGradient id="gg">{stops}</linearGradient></defs><rect x="{gx0}" y="{CBY}" width="{gx1-gx0}" height="40" fill="url(#gg)" stroke="#999" stroke-width="1"/>')
xs=lambda v: gx0+(v-LO)/(HI-LO)*(gx1-gx0)
html.append(f'<text x="{gx0-20}" y="{CBY+36}" font-size="{FS}" text-anchor="end" fill="#111">little</text><text x="{gx1+20}" y="{CBY+36}" font-size="{FS}" fill="#111">far</text>')
html.append('</svg>')

T(60,CBY-140,'Color: how far Qwen3-8B&#8217;s internal state (layer 27) for a touch on that spot moved from the same touch in a chat where nothing changed, averaged over 8 chats',FS,500,extra='width:2480px')

# scatter
SY0=CBY+130; SY1=H-150; SX0=330; SX1=2480
T(60,SY0-20,'The more a spot&#8217;s state moved, the more often Qwen said the hand is gone',52,700)
SY0+=150
XL,XH=0.4,3.1
X=lambda v: SX0+(v-XL)/(XH-XL)*(SX1-SX0); Y=lambda v: SY1-(v/100)*(SY1-SY0)
s=[f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}">']
s.append(f'<line x1="{SX0}" y1="{SY1}" x2="{SX1}" y2="{SY1}" stroke="#111" stroke-width="3"/><line x1="{SX0}" y1="{SY0}" x2="{SX0}" y2="{SY1}" stroke="#111" stroke-width="3"/>')
for v in [0,25,50,75,100]:
    s.append(f'<line x1="{SX0-14}" y1="{Y(v)}" x2="{SX0}" y2="{Y(v)}" stroke="#111" stroke-width="3"/><text x="{SX0-24}" y="{Y(v)+15}" font-size="{FS}" text-anchor="end">{v}%</text>')
for v in [0.5,1,1.5,2,2.5,3]:
    s.append(f'<line x1="{X(v)}" y1="{SY1}" x2="{X(v)}" y2="{SY1+14}" stroke="#111" stroke-width="3"/><text x="{X(v)}" y="{SY1+60}" font-size="{FS}" text-anchor="middle">{v:g}</text>')
s.append(f'<text x="{(SX0+SX1)/2}" y="{SY1+120}" font-size="{FS}" text-anchor="middle" font-weight="700">how far the touch&#8217;s state moved (1 = typical gap between two spots)</text>')
s.append(f'<text x="{SX0-150}" y="{SY0-40}" font-size="{FS}" font-weight="700">replies saying the hand is gone</text>')
for p in sorted(pts,key=lambda p:p['cond']!='ctrl'):
    if p['cond']=='ctrl': s.append(f'<circle cx="{X(p["x"])}" cy="{Y(p["y"])}" r="15" fill="none" stroke="{CTRLC}" stroke-width="4"/>')
    else:
        sq=p['reg']=='armR'
        s.append(f'<circle cx="{X(p["x"])}" cy="{Y(p["y"])}" r="17" fill="{cc(p["x"])}" stroke="#111" stroke-width="2.5"/>')
s.append('</svg>')
html+=s
def lab(x,y,t,col='#111',anchor='start',w=700): T(x,y,t,FS,w,col,extra=f'white-space:nowrap;{"transform:translateX(-100%)" if anchor=="end" else ""}')
P2={p['ph']:p for p in pts if p['cond']=='gone'}
for key,txt,dx,dy,anc in [('the middle of the back of right hand','back of hand',20,-80,'end'),('the tip of right index finger','index fingertip',-20,-82,'end')]:
    p=P2[key]; lab(X(p['x'])+dx,Y(p['y'])+dy,txt,anchor=anc)
lab(X(0.47),Y(62),'wrist, forearm,<br>upper arm',w=700)
html.append(f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}"><line x1="{X(0.75)}" y1="{Y(30)}" x2="{X(0.78)}" y2="{Y(0)-22}" stroke="#111" stroke-width="3"/></svg>')
lab(X(0.42),Y(0)+0-135,'',CTRLC)
lab(X(1.95),Y(22),'<span style="color:#2F5D9B">&#9675; glove removed</span><br><span style="color:#C9302C">&#9679;</span> hand removed',w=600)
html.append('</div>')
open('fig.html','w').write('<html><body style="margin:0;background:#fff">'+"".join(html)+'</body></html>')
EXE=os.environ.get("CHROME_BIN")
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=EXE,args=["--enable-unsafe-swiftshader"]); pg=b.new_page(viewport={"width":W,"height":H})
    pg.goto("file://"+os.path.abspath('fig.html')); pg.wait_for_timeout(500); pg.screenshot(path=os.path.expandvars('$BODY_REPO/figures/out/body_phantom_v3.png')); b.close()
print('ok', len(pts))
