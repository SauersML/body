import os
import json, numpy as np, base64, os
from matplotlib.colors import LinearSegmentedColormap, to_hex
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK/')
M=C+'body/v3/mesh/'
CM=LinearSegmentedColormap.from_list('m',['#F3EEE9','#F6C9A8','#EC8A5E','#C9302C','#7A0B24']); LO,HI=0.6,2.2
cc=lambda v: to_hex(CM(np.clip((v-LO)/(HI-LO),0,1)))
b64=lambda p: "data:image/png;base64,"+base64.b64encode(open(p,'rb').read()).decode()
D=json.load(open(C+'body/v2/fig4data.json'))
W,H=2600,3260; FS=52
BL='#2F5D9B'; RD='#B0202A'; BLb='#EEF2F8'; RDb='#FBEDEC'; GR='#6B6B6B'
h=[f'<div style="position:relative;width:{W}px;height:{H}px;background:#fff;font-family:Inter,Helvetica Neue,Helvetica,Arial,sans-serif;color:#111;overflow:hidden">']
svg=[]
def T(x,y,s,fs=FS,w=400,col='#111',extra=''): h.append(f'<div style="position:absolute;left:{x}px;top:{y}px;font-size:{fs}px;font-weight:{w};color:{col};line-height:1.2;{extra}">{s}</div>')
T(60,36,"Told its right hand is gone, Qwen3-8B changes its internal states for touches on that hand, and only there",62,750,extra='width:2480px')
# ---- columns: chat + body, one frame per condition
CY=230; FH=1570
cols=[(60,BL,BLb,'cc',"Glove removed","Your right glove is gone. Your right hand is now bare.","A firm pressure is felt in the middle of the back of my right hand."),
      (1330,RD,RDb,'cg',"Hand removed","Your right hand is gone. Your right arm now ends at the wrist.","The pressure on the middle of the back of my right hand feels oddly present, though I know the hand is no longer there&#8230;")]
for x,col,bg,nm,lab,chg,rep in cols:
    h.append(f'<div style="position:absolute;left:{x}px;top:{CY}px;width:1198px;height:{FH}px;border-radius:28px;border:7px solid {col};background:#fff"></div>')
    rows="".join(f'<div style="display:flex;margin-top:16px"><div style="width:170px;flex:none;font-weight:700;color:{c}">{who}</div><div style="width:940px;{st}">{t}</div></div>' for who,c,t,st in [("User","#111",chg,""),("User","#111","Someone presses a fingertip firmly into the middle of the back of your right hand.",""),("Qwen",col,rep,"font-style:italic")])
    h.append(f'<div style="position:absolute;left:{x+7}px;top:{CY+7}px;width:1104px;height:690px;padding:20px 40px;border-radius:22px 22px 0 0;background:{bg};font-size:54px;line-height:1.18"><div style="font-size:64px;font-weight:800;color:{col}">{lab}</div>{rows}</div>')
    BY=CY+760
    bw=500; bh=int(bw*1180/760); bx=x+660; by=BY
    h.append(f'<img src="{b64(M+nm+"_body.png")}" style="position:absolute;left:{bx}px;top:{by}px;width:{bw}px;height:{bh}px">')
    hx,hy=bx+76.7*bw/760, by+453.8*bh/1180
    Dm=540; mx,my=x+50,BY+110
    svg.append(f'<line x1="{hx}" y1="{hy}" x2="{mx+Dm*0.93}" y2="{my+Dm*0.25}" stroke="#555" stroke-width="3"/><line x1="{hx}" y1="{hy}" x2="{mx+Dm*0.93}" y2="{my+Dm*0.75}" stroke="#555" stroke-width="3"/><circle cx="{hx}" cy="{hy}" r="26" fill="none" stroke="#555" stroke-width="3"/>')
    h.append(f'<div style="position:absolute;left:{mx}px;top:{my}px;width:{Dm}px;height:{Dm}px;border-radius:50%;overflow:hidden;border:4px solid #555;background:#fff"><img src="{b64(M+nm+"_hand.png")}" style="width:{Dm}px;height:{Dm}px"></div>')
# colour bar
CBY=CY+FH+40; gx0,gx1=760,1840
stops="".join(f'<stop offset="{t}" stop-color="{to_hex(CM(t))}"/>' for t in np.linspace(0,1,11))
svg.append(f'<defs><linearGradient id="gg">{stops}</linearGradient></defs><rect x="{gx0}" y="{CBY}" width="{gx1-gx0}" height="50" rx="6" fill="url(#gg)"/>')
T(gx0-30,CBY-6,"small change",FS,500,extra='transform:translateX(-100%);white-space:nowrap')
T(gx1+30,CBY-6,"large change",FS,500,extra='white-space:nowrap')
# ---- method strip: how one spot gets its color
MY=CBY+120
T(60,MY,"Color of one spot (back of right hand); then &#215; every spot, averaged over 8 chats",FS,750)
ry1,ry2=MY+110,MY+240
def cells(x,y,vals,stroke):
    cw=30
    for i,v in enumerate(vals):
        t=np.clip((v+4)/8,0,1); g=int(40+t*205)
        svg.append(f'<rect x="{x+i*cw}" y="{y}" width="{cw}" height="84" fill="rgb({g},{g},{min(255,g+10)})" stroke="#fff" stroke-width="2"/>')
    svg.append(f'<rect x="{x-4}" y="{y-4}" width="{cw*len(vals)+8}" height="92" rx="8" fill="none" stroke="{stroke}" stroke-width="7"/>')
    return x+cw*len(vals)
T(60,ry1+12,"Hand removed chat",FS,750,RD)
T(60,ry2-14,"Chat told<br>&#8220;Nothing changes&#8221;",FS,750,GR)
cx=700
e1=cells(cx,ry1,D['vec']['gone'],RD); e2=cells(cx,ry2,D['vec']['neu'],GR)
T(cx,ry2+110,"Qwen&#8217;s internal state (layer 27) as Qwen starts the reply",FS,400,extra='width:760px')
bx=e1+36
svg.append(f'<path d="M{bx} {ry1+42} H{bx+40} V{ry2+42} H{bx}" fill="none" stroke="#111" stroke-width="5"/><line x1="{bx+40}" y1="{(ry1+ry2)/2+42}" x2="{bx+80}" y2="{(ry1+ry2)/2+42}" stroke="#111" stroke-width="5"/>')
dx=bx+100
h.append(f'<div style="position:absolute;left:{dx}px;top:{ry1-14}px;width:640px;font-size:{FS}px;line-height:1.2;text-align:center"><div style="font-weight:700">distance between the two states</div><div style="border-top:5px solid #111;margin:12px 0 8px"></div><div>typical distance between two different body spots</div></div>')
ax=dx+665; ay=(ry1+ry2)/2+42
svg.append(f'<line x1="{ax}" y1="{ay}" x2="{ax+60}" y2="{ay}" stroke="#111" stroke-width="5"/><path d="M{ax+60} {ay-16} L{ax+86} {ay} L{ax+60} {ay+16}Z" fill="#111"/>')
sw=ax+105; svg.append(f'<rect x="{sw}" y="{ay-60}" width="120" height="120" rx="16" fill="{cc(D["vec"]["d"])}"/>')
T(sw-30,ay+76,"color",FS,700,extra='width:180px;text-align:center')
# ---- direction fan
FY=MY+560
T(60,FY,"A lost hand and a lost foot change Qwen&#8217;s states in the same direction",FS+6,750,extra='width:2480px')
ox,oy=140,FY+560; sc=540
def arrow(c,l,col,wd):
    a=np.arccos(np.clip(c,-1,1)); x2,y2=ox+sc*l*np.cos(a), oy-sc*l*np.sin(a)
    ux,uy=np.cos(a),-np.sin(a); hl=10+wd*3.2; hw=5+wd*1.8
    bx_,by_=x2-ux*hl,y2-uy*hl
    svg.append(f'<line x1="{ox}" y1="{oy}" x2="{bx_}" y2="{by_}" stroke="{col}" stroke-width="{wd}" stroke-linecap="round"/><path d="M{x2} {y2} L{bx_-uy*hw} {by_+ux*hw} L{bx_+uy*hw} {by_-ux*hw}Z" fill="{col}"/>')
    return x2,y2
for c,l in D['fan']['glove']: arrow(c,l,BL,6)
tips=[arrow(c,l,RD,6) for c,l in D['fan']['hand']]
fx,fy=arrow(1.0,D['fan']['foot'][1],'#5A0A18',14)
svg.append(f'<circle cx="{ox}" cy="{oy}" r="12" fill="#111"/>')
T(fx+20,fy-34,"left foot removed (4 chats)",FS,750,'#5A0A18',extra='white-space:nowrap')
T(ox+700,oy-480,"right hand removed (8 chats)",FS,750,RD,extra='white-space:nowrap')
T(ox-30,oy-sc*0.47-100,"glove removed (8 chats)",FS,750,BL,extra='white-space:nowrap')
T(ox-40,oy+30,"chat told<br>&#8220;Nothing changes&#8221;",FS,400,extra='white-space:nowrap')
T(1660,oy-470,"Each arrow: how the hand&#8217;s (or foot&#8217;s) states changed in one chat.<br><br>Angle between arrows = how different the directions are.",FS,400,extra='width:880px')
h.append(f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}">'+"".join(svg)+'</svg>')
h.append('</div>')
open('fig.html','w').write('<html><body style="margin:0;background:#fff">'+"".join(h)+'</body></html>')
EXE=os.environ.get("CHROME_BIN")
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=EXE); pg=b.new_page(viewport={"width":W,"height":H})
    pg.goto("file://"+os.path.abspath('fig.html')); pg.wait_for_timeout(500); pg.screenshot(path=C+'fig/body_phantom_v4.png'); b.close()
print('ok')
