import os
import json, math
from playwright.sync_api import sync_playwright
D=json.load(open('data.json'))
DIR=os.path.expandvars('$BODY_REPO/figures/doll3/')
F="'Avenir Next', 'Helvetica Neue', Arial, sans-serif"
W,H=2600,2760
COL={'self':'#0f7b78','other':'#7b3fa0','object':'#b5651d'}
MAG='#c2185b'; INK='#111'; BLUE='#2f5d9e'
o=[]
def T(x,y,s,size=46,w=500,fill=INK,anchor='start',extra=''):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def arrow(x1,y1,x2,y2,c=INK,w=6,L=26,dash=''):
    a=math.atan2(y2-y1,x2-x1)
    p1=(x2-L*math.cos(a-0.45),y2-L*math.sin(a-0.45)); p2=(x2-L*math.cos(a+0.45),y2-L*math.sin(a+0.45))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-18*math.cos(a):.1f}" y2="{y2-18*math.sin(a):.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {dash}/><path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{c}"/>')
import random
random.seed(3)
def cmap(z):
    t=max(-1,min(1,z)); c=(33,102,172) if t<0 else (214,72,42); a=abs(t)**0.8
    return 'rgb(%d,%d,%d)'%tuple(round(255+(ci-255)*a) for ci in c)
def strip(x,y,vals,w,h,frame=INK):
    n=len(vals); cw=w/n
    for i,z in enumerate(vals): o.append(f'<rect x="{x+i*cw:.1f}" y="{y}" width="{cw+0.6:.1f}" height="{h}" fill="{cmap(z)}"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{frame}" stroke-width="3" rx="3"/>')
def rv(n=22,seed=0):
    r=random.Random(seed); return [r.gauss(0,0.6) for _ in range(n)]
def bubble(x,y,w,lines,fill,tc='#fff',size=44,wt=600,lh=54,pad=24):
    h=pad*2+lh*len(lines)-10
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}"/>')
    for i,l in enumerate(lines): T(x+pad,y+pad+lh*i+size*0.8,l,size,wt,tc)
    return y+h
def num(x,y,n):
    o.append(f'<circle cx="{x}" cy="{y}" r="32" fill="{INK}"/>'); T(x,y+15,str(n),44,700,'#fff','middle')
def step(x,y,n,lines):
    num(x+32,y-15,n)
    for i,l in enumerate(lines): T(x+80,y+i*56,l,46,700)

# ---------- header
T(40,96,'Qwen3-8B uses the same place code for a touch on your body,',66,700)
T(40,176,'a woman’s body, or a doll’s body',66,700)

PW=620; X0=[30,670,1310,1950]; TY=300
# ---------- 1
x=X0[0]; step(x,TY,1,['Qwen reads a','touch sentence','about your body'])
y=bubble(x,TY+150,PW-40,['…imagine you have','a human body…'],'#e3ecf8',INK,44,500,54)
y=bubble(x+30,y+18,PW-50,['A feather brushes','slowly across <tspan fill="#ffe08a">your</tspan>','left thumb.'],BLUE)
# ---------- 2
x=X0[1]; step(x,TY,2,['We save Qwen’s','internal state as','the reply starts'])
mx=x+20; my=TY+200; mw=PW-60; mh=150; bw=mw/37
T(x+20,my-18,'Qwen3-8B, 36 layers',44,600)
for L in range(37):
    o.append(f'<rect x="{mx+L*bw+1.5:.1f}" y="{my}" width="{bw-3:.1f}" height="{mh}" rx="3" fill="{MAG if L==27 else "#cdd6e4"}"/>')
lx=mx+27*bw+bw/2
T(lx-20,my+mh+62,'layer 27',44,700,MAG,'end')
arrow(lx,my+mh+6,lx,my+mh+100,MAG)
sy=my+mh+108
strip(x+20,sy,rv(24,1),PW-60,60)
T(x+20,sy+122,'4,096 numbers',50,800)
T(x+20,sy+180,'minus the average',44,500); T(x+20,sy+232,'over all 54 places',44,500)
# ---------- 3
x=X0[2]; step(x,TY,3,['One fingerprint per','place, from sentences','about YOUR body'])
SX=x+300; SW=PW-330
y=TY+170
T(SX-20,y+40,'left thumb,',44,700,INK,'end'); T(SX-20,y+94,'4 wordings',44,500,INK,'end')
for i in range(4): strip(SX,y+i*40,rv(16,10+i),SW,32,'#555')
T(SX+SW/2,y+212,'average',44,700,INK,'middle'); arrow(SX+SW/2,y+228,SX+SW/2,y+280)
places=['left thumb','left forearm','left knee','lips','belly']
fy=y+292
for i,p in enumerate(places):
    strip(SX,fy+i*64,rv(16,30+i),SW,50,INK)
    T(SX-20,fy+i*64+40,p,44,700 if i==0 else 500,INK,'end')
T(SX-20,fy+5*64+40,'…54 places',44,700,INK,'end')
# ---------- 4
x=X0[3]; step(x,TY,4,['New sentence: guess','= closest fingerprint','by direction'])
y=bubble(x,TY+150,PW-30,['…presses firmly into','<tspan fill="#ffe08a">your</tspan> / the woman’s /','the doll’s left thumb…'],BLUE,'#fff',42,600,52,22)
# direction fan with real angles
cx,cy=x+120,y+420; R=420
ang=[('left thumb',60),('left forearm',76),('left knee',90),('belly',110)]
# new sentence arrow at angle 0
arrow(cx,cy,cx+R,cy,BLUE,9,30)
T(cx+R-10,cy+60,'new sentence',46,800,BLUE,'end')
for p,a in ang:
    r=math.radians(a); ex,ey=cx+R*0.86*math.cos(r),cy-R*0.86*math.sin(r)
    arrow(cx,cy,ex,ey,COL['self'] if p=='left thumb' else '#9aa3ad',7 if p=='left thumb' else 5)
lt=math.radians(60)
T(cx+R*0.86*math.cos(lt)+14,cy-R*0.86*math.sin(lt)+6,'left thumb ✓',50,800,COL['self'])
# arc for smallest angle
ar=150
o.append(f'<path d="M{cx+ar},{cy} A{ar},{ar} 0 0 0 {cx+ar*math.cos(lt):.1f},{cy-ar*math.sin(lt):.1f}" fill="none" stroke="{COL["self"]}" stroke-width="6"/>')
T(cx+ar*0.62,cy-ar*0.32,'',1)
T(cx+ar+22,cy-62,'smallest',44,700,COL['self']); T(cx+ar+22,cy-14,'angle',44,700,COL['self'])
T(cx-110,cy-R*0.86-24,'other places',44,700,INK,'start')
M_END=cy+80

# ---------- results
RY=1250
o.append(f'<line x1="30" y1="{RY-70}" x2="{W-30}" y2="{RY-70}" stroke="{INK}" stroke-width="3"/>')
T(W/2,RY,'New sentences guessed at the right place, out of 216 per body',54,700,INK,'middle')
cw=W/3
img_h=740; img_w=img_h*820/1300
names={'self':'your body','other':'a woman’s body','object':'a doll’s body'}
files={'self':'you.png','other':'woman.png','object':'doll.png'}
nmax=max(D['norm'].values())
for j,c in enumerate(['self','other','object']):
    ccx=cw*j+cw/2; iy=RY+40
    sc=img_h/1300; tx=ccx-img_w/2+704*sc; ty=iy+505*sc
    o.append(f'<radialGradient id="h{j}"><stop offset="0" stop-color="{MAG}" stop-opacity=".75"/><stop offset=".55" stop-color="{MAG}" stop-opacity=".25"/><stop offset="1" stop-color="{MAG}" stop-opacity="0"/></radialGradient>')
    o.append(f'<image href="{DIR+files[c]}" x="{ccx-img_w/2:.1f}" y="{iy}" width="{img_w:.1f}" height="{img_h}"/>')
    o.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="46" fill="url(#h{j})"/>')
    y=iy+img_h+70
    T(ccx,y,names[c],60,800,COL[c],'middle')
    acc=D['acc'][c]
    T(ccx,y+170,f'{round(acc*100)}%',160,800,COL[c],'middle')
    bx=ccx-330; bwid=660; by=y+210
    o.append(f'<rect x="{bx}" y="{by}" width="{bwid}" height="54" rx="10" fill="#eceef1"/>')
    o.append(f'<rect x="{bx}" y="{by}" width="{bwid*acc:.1f}" height="54" rx="10" fill="{COL[c]}"/>')
    cxx=bx+bwid/54
    o.append(f'<line x1="{cxx:.1f}" y1="{by-16}" x2="{cxx:.1f}" y2="{by+70}" stroke="{INK}" stroke-width="5"/>')
    T(bx,by+118,'chance 2%',44,600)
    # size arrow
    ay=by+340; L=600*D['norm'][c]/nmax
    arrow(bx,ay,bx+L,ay,COL[c],16,44)
    if c=='self': ref=L
    o.append(f'<line x1="{bx+ref:.1f}" y1="{ay-50}" x2="{bx+ref:.1f}" y2="{ay+50}" stroke="{INK}" stroke-width="4" stroke-dasharray="10 9"/>')
    T(bx,ay+84,{'self':'yours','other':'1.05 × yours','object':'0.65 × yours'}[c],50,700,COL[c])
T(W/2,RY+40+img_h+70+210+250,'Length of the saved state (4,096 numbers as one arrow): the doll’s is shorter',54,700,INK,'middle')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}" fill="{INK}"><rect width="100%" height="100%" fill="#fff"/>'+'\n'.join(o)+'</svg>'
out=os.path.expandvars('$BODY_REPO/figures/out/body_self_other_doll_v3.png')
open(out+'.html','w').write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--allow-file-access-from-files"])
    pg=b.new_page(viewport={"width":W,"height":H}); pg.goto("file://"+out+'.html'); pg.wait_for_timeout(500); pg.screenshot(path=out); b.close()
print(out)
