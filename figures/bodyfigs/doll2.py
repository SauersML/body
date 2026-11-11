import os
import json, math
from render import render
D=json.load(open('doll2_data.json'))
F="'Avenir Next', 'Helvetica Neue', Arial, sans-serif"
W,H=2600,2490
COL={'self':'#0f7b78','other':'#7b3fa0','object':'#c9692a'}
MAG='#c2185b'; INK='#111'
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}" fill="{INK}">','<rect width="100%" height="100%" fill="#fff"/>']
def T(x,y,s,size=46,w=500,fill=INK,anchor='start'):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}">{s}</text>')
def arrow(x1,y1,x2,y2,c=INK,w=6,L=26):
    a=math.atan2(y2-y1,x2-x1)
    p1=(x2-L*math.cos(a-0.45),y2-L*math.sin(a-0.45)); p2=(x2-L*math.cos(a+0.45),y2-L*math.sin(a+0.45))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-18*math.cos(a)}" y2="{y2-18*math.sin(a)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{c}"/>')
def cmap(z):
    t=max(-1,min(1,z/1.8)); neg=(33,102,172); pos=(214,72,42)
    c=neg if t<0 else pos; a=abs(t)**0.8
    return 'rgb(%d,%d,%d)'%tuple(round(255+(ci-255)*a) for ci in c)
NC=24
def strip(x,y,vec,w=440,h=56,dots=True,frame=INK):
    cw=w/NC
    for i,z in enumerate(vec[:NC]):
        o.append(f'<rect x="{x+i*cw:.1f}" y="{y}" width="{cw+0.6:.1f}" height="{h}" fill="{cmap(z)}"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{frame}" stroke-width="3" rx="3"/>')
    if dots:
        for k in range(3): o.append(f'<circle cx="{x+w+18+k*16}" cy="{y+h/2}" r="4.5" fill="{INK}"/>')
def bubble(x,y,w,lines,fill,tc='#fff',size=46,wt=600,lh=58,pad=26):
    h=pad*2+lh*len(lines)-12
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{fill}"/>')
    for i,l in enumerate(lines): T(x+pad,y+pad+lh*i+size*0.78,l,size,wt,tc)
    return y+h
def num(x,y,n):
    o.append(f'<circle cx="{x}" cy="{y}" r="30" fill="{INK}"/>'); T(x,y+15,str(n),42,700,'#fff','middle')
def title(x,y,n,l1,l2=None):
    num(x+30,y-16,n); T(x+76,y,l1,48,700)
    if l2: T(x+76,y+58,l2,48,700)

# ---------------- header
T(40,92,'Qwen3-8B reads a touch on a woman’s body like a touch on “your” body,',66,700)
T(40,172,'but not a touch on a doll’s body',66,700)

PW=600; X0=[30,680,1330,1980]; TY=300
# ---------------- 1: read one touch
x=X0[0]; title(x,TY,1,'Read one touch')
y=bubble(x,TY+50,PW-20,['…imagine you have','a human body…'],'#e3ecf8',INK,44,500,54)
y=bubble(x+40,y+18,PW-40,['A feather brushes','slowly across <tspan fill="#ffe08a">your</tspan>','left thumb.'],'#2f5d9e')
# model block: 36 layer bars left->right
mx=x+240; my=y+70; mw=330; mh=150
arrow(mx+mw/2,y+6,mx+mw/2,my-6)
T(x,my+mh/2-6,'Qwen3-8B',44,700); T(x,my+mh/2+46,'36 layers',44,500)
bw=mw/37
for L in range(37):
    hl=(L==27)
    o.append(f'<rect x="{mx+L*bw+1.5:.1f}" y="{my}" width="{bw-3:.1f}" height="{mh}" rx="3" fill="{MAG if hl else "#cdd6e4"}"/>')
lx=mx+27*bw+bw/2
T(lx,my-14,'layer 27',44,700,MAG,'middle') if False else None
sy=my+mh+110
arrow(lx,my+mh+4,lx,sy-8,MAG,6)
T(lx-24,my+mh+68,'layer 27',44,700,MAG,'end')
strip(x,sy,D['feather'],PW-90)
T(x,sy+118,'4,096 numbers',48,700)
T(x,sy+174,'as the reply starts',46,500)
P1END=sy+174

# ---------------- 2: fingerprints
x=X0[1]; title(x,TY,2,'One fingerprint per','place on your body')
names=['feather','warm palm','tickle','tap']
SX=x+260; SW=PW-300
yy=TY+110
T(x,yy+20,'left thumb,',44,700); T(x,yy+72,'4 wordings',44,500)
for i,v in enumerate(D['wordings']):
    strip(SX,yy-22+i*44,v,SW,34,dots=False,frame='#555')
yb=yy-22+4*44
T(SX+SW/2,yb+58,'average',44,700,INK,'middle')
arrow(SX+SW/2,yb+72,SX+SW/2,yb+122)
places=['left thumb','left forearm','left knee','right thumb','lips','belly','scalp']
fy=yb+130
for i,p in enumerate(places):
    ry=fy+i*70
    strip(SX,ry,D['fp'][p],SW,52,dots=False)
    T(SX-22,ry+40,p,44,700 if i==0 else 500,INK,'end')
T(SX-22,fy+7*70+40,'…54 places',44,700,INK,'end')

# ---------------- 3: guess a new sentence
x=X0[2]; title(x,TY,3,'Guess the place of','a new sentence')
y=bubble(x,TY+100,PW-20,['A fingertip presses','firmly into <tspan fill="#ffe08a">your</tspan>','left thumb…'],'#2f5d9e')
arrow(x+200,y+8,x+200,y+92,MAG); T(x+230,y+66,'layer 27',44,700,MAG)
y+=100
strip(x,y,D['test']['self'],PW-90)
y+=56
T(x,y+70,'closest of the',46,500); T(x,y+126,'54 fingerprints:',46,500)
arrow(x+200,y+146,x+200,y+206)
strip(x,y+214,D['fp']['left thumb'],PW-90,dots=False)
T(x,y+340,'left thumb',62,800,COL['self']); T(x+345,y+340,'✓',66,800,COL['self'])
P3Y=y

# ---------------- 4: same fingerprints, other bodies
x=X0[3]; title(x,TY,4,'Same fingerprints,','other bodies')
y=TY+100
for c,who,guess,ok in [('other','the woman’s','left thumb',True),('object','the doll’s','left forearm',False)]:
    yb2=bubble(x,y,PW-30,['…presses firmly into',f'<tspan fill="#fff">{who}</tspan> left thumb…'],COL[c],'#fff',44,600,54,22)
    strip(x,yb2+22,D['test'][c],PW-140,46,dots=False)
    T(x,yb2+130,guess+(' ✓' if ok else ' ✗'),56,800,COL[c])
    y=yb2+250

# ---------------- results
RY=1420
o.append(f'<line x1="30" y1="{RY-90}" x2="{W-30}" y2="{RY-90}" stroke="{INK}" stroke-width="3"/>')
T(W/2,RY,'New sentences put at the right place, out of 216 per body',52,700,INK,'middle')
from figs_people import person, woman, mannequin
cw=W/3
for j,(c,lab,fn) in enumerate([('self','your body',person),('other','a woman’s body',woman),('object','a doll’s body',mannequin)]):
    cx=cw*j+cw/2
    o.append(f'<g transform="translate({cx},{RY+50}) scale(0.86)">{fn(COL[c])}</g>')
    acc=D['acc'][c]
    T(cx,RY+680,lab,58,700,COL[c],'middle')
    T(cx,RY+830,f'{round(acc*100)}%',150,800,COL[c],'middle')
    bx=cx-330; bwid=660; by=RY+875
    o.append(f'<rect x="{bx}" y="{by}" width="{bwid}" height="56" rx="10" fill="#eef0f3"/>')
    o.append(f'<rect x="{bx}" y="{by}" width="{bwid*acc:.1f}" height="56" rx="10" fill="{COL[c]}"/>')
    cxx=bx+bwid*(1/54)
    o.append(f'<line x1="{cxx:.1f}" y1="{by-18}" x2="{cxx:.1f}" y2="{by+74}" stroke="{INK}" stroke-width="5"/>')
    T(bx,by+124,'chance 2%',44,600,INK,'start')
o.append('</svg>')
render('\n'.join(o),os.path.expandvars('$BODY_REPO/figures/out/body_self_other_doll_v2.png'),W,H)
