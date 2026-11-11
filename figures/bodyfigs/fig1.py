import os
import json, math, sys
sys.path.insert(0,'.'); from fig_body import body, pos
from render import render
J=json.load(open('../_selfother_L27.json'))
KEYS=['thumbL','indexL','middleL','ringL','littleL','palmL','backhandL','wristL','forearmL','elbowL','upperarmL','shoulderL','toesL','soleL','heelL','ankleL','shinL','kneeL','thighL','hipL','thumbR','indexR','middleR','ringR','littleR','palmR','backhandR','wristR','forearmR','elbowR','upperarmR','shoulderR','toesR','soleR','heelR','ankleR','shinR','kneeR','thighR','hipR','forehead','crown','cheekL','cheekR','lips','tongue','earL','earR','neckF','neckB','chest','belly','upback','lowback']
FONT="-apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
W,H=2400,2700
COL={'self':'#0e7c7b','other':'#7a3e9d','object':'#d0661c'}
NAME={'self':'your','other':"the woman's",'object':"the doll's"}
HEAD={'self':'your body','other':"a woman's body",'object':"a doll's body"}
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" fill="#111">','<rect width="100%" height="100%" fill="#fff"/>']
def T(x,y,s,size=50,w=400,fill="#111",anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}">{s}</text>'
def arrow(x1,y1,x2,y2,c="#222",w=7):
    a=math.atan2(y2-y1,x2-x1); L=30
    p1=(x2-L*math.cos(a-0.45),y2-L*math.sin(a-0.45)); p2=(x2-L*math.cos(a+0.45),y2-L*math.sin(a+0.45))
    return f'<line x1="{x1}" y1="{y1}" x2="{x2-20*math.cos(a)}" y2="{y2-20*math.sin(a)}" stroke="{c}" stroke-width="{w}"/><path d="M{x2},{y2} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z" fill="{c}"/>'
# ---------- method row
y0=40
o.append(f'<rect x="50" y="{y0}" width="860" height="420" rx="30" fill="#f1f2f5"/>')
o.append(T(85,y0+70,'Warm-up chat: “imagine you have',48))
o.append(T(85,y0+128,'a human body, standing in a quiet,',48))
o.append(T(85,y0+186,'sunny room”, then one touch:',48))
o.append(f'<rect x="85" y="{y0+222}" width="790" height="170" rx="24" fill="#2f5d9e"/>')
o.append(T(115,y0+292,'A feather brushes slowly',52,600,'#fff'))
o.append(T(115,y0+356,'across <tspan fill="#ffe08a">your</tspan> left thumb.',52,600,'#fff'))
o.append(arrow(925,y0+250,1000,y0+250))
lx=1015
o.append(T(lx,y0+52,'Qwen3-8B',52,700))
for i in range(36):
    yy=y0+380-i*8.6; hl=(i==27)
    o.append(f'<rect x="{lx}" y="{yy}" width="230" height="{8 if hl else 5}" rx="2" fill="{"#c2185b" if hl else "#c4c7ce"}"/>')
yl=y0+380-27*8.6
o.append(T(lx,y0+440,'layer 27 of 36,',48,700,'#c2185b'))
o.append(T(lx,y0+494,'state where the',48,400))
o.append(T(lx,y0+546,'reply starts',48,400))
o.append(arrow(lx+250,yl+4,1485,yl+4,"#c2185b"))
# decoder box
dx=1500
o.append(f'<rect x="{dx}" y="{y0}" width="860" height="470" rx="30" fill="#fff" stroke="#111" stroke-width="5"/>')
o.append(T(dx+40,y0+80,'Decoder: which of 54',56,700))
o.append(T(dx+40,y0+146,'touched places?',56,700))
o.append(T(dx+40,y0+234,'learned only from touches',48))
o.append(T(dx+40,y0+292,'on <tspan font-weight="700" fill="'+COL['self']+'">your</tspan> body, then tested',48))
o.append(T(dx+40,y0+350,'on new sentences about each',48))
o.append(T(dx+40,y0+408,'of the three bodies',48))
# ---------- three bodies
cw=780; bx0=30; by=800
for j,c in enumerate(['self','other','object']):
    X=bx0+j*cw
    cx=X+cw/2
    o.append(T(cx,by-130,HEAD[c],64,800,COL[c],'middle'))
    o.append(T(cx,by-60,f'“…across <tspan font-weight="700" fill="{COL[c]}">{NAME[c]}</tspan> left thumb.”',46,400,'#111','middle'))
    o.append(f'<g transform="translate({X+10},{by}) scale(1)">{body({'self':'you','other':'woman','object':'doll'}[c])}')
    per=J[c]['perloc']
    for k,key in enumerate(KEYS):
        x,y=pos(key); f=per[k]; r=15
        o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{COL[c]}" stroke-width="4"/>')
        if f>=1: o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL[c]}"/>')
        elif f>0:
            a=2*math.pi*f; ex=x+r*math.sin(a); ey=y-r*math.cos(a)
            o.append(f'<path d="M{x},{y} L{x},{y-r} A{r},{r} 0 {1 if f>.5 else 0} 1 {ex:.1f},{ey:.1f} Z" fill="{COL[c]}"/>')
    # example place ring
    x,y=pos('thumbL'); o.append(f'<circle cx="{x}" cy="{y}" r="27" fill="none" stroke="#111" stroke-width="5"/>')
    o.append('</g>')
    # accuracy bar
    acc=J['layers'][c][27]
    yb=by+1070
    o.append(T(cx,yb,f'{round(acc*100)}%',110,800,COL[c],'middle'))
    o.append(f'<rect x="{X+70}" y="{yb+35}" width="{cw-140}" height="44" rx="8" fill="#eceef2"/>')
    o.append(f'<rect x="{X+70}" y="{yb+35}" width="{(cw-140)*acc}" height="44" rx="8" fill="{COL[c]}"/>')
    o.append(f'<line x1="{X+70+(cw-140)*0.0185}" y1="{yb+20}" x2="{X+70+(cw-140)*0.0185}" y2="{yb+94}" stroke="#111" stroke-width="5"/>')
# doll thumb read as forearm: curved arrow
X=bx0+2*cw+10; (tx,ty)=pos('thumbL'); (fx,fy)=pos('forearmL')
o.append(f'<path d="M{X+tx+20},{by+ty-24} C{X+tx+90},{by+ty-150} {X+fx+90},{by+fy+10} {X+fx+24},{by+fy+4}" fill="none" stroke="#111" stroke-width="5"/>')
o.append(f'<path d="M{X+fx+18},{by+fy+4} l22,-20 l6,26 z" fill="#111"/>')
o.append(f'<line x1="{X+tx+5}" y1="{by+ty+30}" x2="{X+700}" y2="{by+748}" stroke="#111" stroke-width="4"/>')
o.append(T(X+775, by+800, 'thumb read', 46, 700, '#111', 'end'))
o.append(T(X+775, by+852, 'as forearm', 46, 700, '#111', 'end'))
# legend: glyphs
ly=by+1225
def pie(x,y,f,c,r=20):
    g=f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{c}" stroke-width="5"/>'
    if f>=1: return g+f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'
    if f<=0: return g
    a=2*math.pi*f; ex=x+r*math.sin(a); ey=y-r*math.cos(a)
    return g+f'<path d="M{x},{y} L{x},{y-r} A{r},{r} 0 {1 if f>.5 else 0} 1 {ex:.1f},{ey:.1f} Z" fill="{c}"/>'
o.append(T(70,ly+16,'Each circle: one touched place, 4 new test sentences',48,400))
lx2=1290
for i,(f,lab) in enumerate([(1,'4 of 4 right'),(.5,'2 of 4'),(0,'0 of 4')]):
    xx=lx2+i*350
    o.append(pie(xx,ly,f,'#444')); o.append(T(xx+36,ly+16,lab,48))
# ---------- layer strip
sy=ly+140; sx0=200; sx1=2060; sh=330
o.append(T(50,sy+10,'From layer 23 on, only the woman still reads like your body',58,800))
top=sy+60; bot=top+sh
xs=lambda L: sx0+(sx1-sx0)*L/36
ys=lambda v: bot-(bot-top)*v
o.append(f'<rect x="{xs(23)}" y="{top}" width="{xs(36)-xs(23)}" height="{sh}" fill="#f6f1ec"/>')
for v in (0,.5,1):
    o.append(f'<line x1="{sx0}" y1="{ys(v)}" x2="{sx1}" y2="{ys(v)}" stroke="#d5d7dc" stroke-width="2"/>')
    o.append(T(sx0-20,ys(v)+16,f'{int(v*100)}%',46,400,'#111','end'))
for c in ['self','other','object']:
    pts=' '.join(f'{xs(L):.1f},{ys(v):.1f}' for L,v in enumerate(J['layers'][c]))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{COL[c]}" stroke-width="9" stroke-linejoin="round"/>')
LAB={'self':'your body','other':"woman",'object':'doll'}
for c in ['self','other','object']:
    v=J['layers'][c][36]; o.append(T(xs(36)+20,ys(v)+16+(-8 if c=='self' else 0),LAB[c],48,700,COL[c]))
o.append(f'<line x1="{xs(27)}" y1="{top-10}" x2="{xs(27)}" y2="{bot}" stroke="#c2185b" stroke-width="4" stroke-dasharray="10 8"/>')
for L in (0,9,18,27,36):
    o.append(T(xs(L),bot+56,str(L),46,400,'#111','middle'))
o.append(T(xs(13.5),bot+56,'layer',46,700,'#111','middle'))
o.append(T(sx0+20,top+50,'test sentences decoded to the right place',46,400,'#111'))
o.append('</svg>')
render('\n'.join(o),os.path.expandvars('$BODY_REPO/figures/out/body_self_other_doll.png'),W,H)
