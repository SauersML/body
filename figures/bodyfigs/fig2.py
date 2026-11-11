import os
import json, math, re, sys
sys.path.insert(0,'.'); from fig_body import body, pos
from render import render
d=json.load(open(os.path.expandvars('$BODY_REPO/map/somato/out_8b/steer2_L25.json')))['results']
G={"hand": r"\b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b",
 "foot": r"\b(foot|feet|toe|toes|sole|soles|heel|heels)\b","head": r"\b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b",
 "belly": r"\b(belly|stomach|abdomen|navel)\b","knee": r"\b(knee|knees|leg|legs|shin\w*|thigh\w*)\b"}
def sh(sel,p):
    xs=[bool(re.search(G[p],r['reply'].lower())) for r in d if sel(r)]; return sum(xs)/len(xs)
PL=[  # dir, regex, name, word, side, anchor key, quote, colour
 ('head top','head','top of head','head or face','L','crown',"A gentle warmth spreads through my scalp, like the first rays of sunlight melting through my hair.",'#c2185b'),
 ('belly','belly','belly','belly or stomach','L','belly',"A soft, full feeling in my belly, like a deep, calm stillness.",'#2e7d32'),
 ('knees','knee','knees','knee or leg','L','kneeR',"I notice a slight pressure on my knee where it touches the floor, and a gentle breeze touching my knee.",'#6a3fb5'),
 ('left hand','hand','left hand','hand or fingers','R','palmL',"I feel a warm glow in my left palm, as if my left hand is gently holding my left thumb.",'#1565c0'),
 ('left foot','foot','left foot','foot or toes','R','soleL',"My feet are drawing my attention, feeling the shift in the floor beneath the toes.",'#d0661c'),
]
NUM={}
for dname,p,*_ in PL:
    NUM[dname]=(sh(lambda r:r['dir']==dname and r['scale']==1.0,p), sh(lambda r:r['dir'].startswith('random') and r['scale']==1.0,p), sh(lambda r:r['dir']=='none',p))
print(json.dumps(NUM,indent=0))
FONT="-apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
W,H=2400,2420
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" fill="#111">','<rect width="100%" height="100%" fill="#fff"/>']
def T(x,y,s,size=50,w=400,fill="#111",anchor="start",style=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>'
def arrow(x1,y1,x2,y2,c="#222",w=7):
    a=math.atan2(y2-y1,x2-x1); L=30
    p1=(x2-L*math.cos(a-0.45),y2-L*math.sin(a-0.45)); p2=(x2-L*math.cos(a+0.45),y2-L*math.sin(a+0.45))
    return f'<line x1="{x1}" y1="{y1}" x2="{x2-20*math.cos(a)}" y2="{y2-20*math.sin(a)}" stroke="{c}" stroke-width="{w}"/><path d="M{x2},{y2} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z" fill="{c}"/>'
def wrap(s,n):
    out=[];cur=''
    for w in s.split():
        if len(cur)+len(w)+1>n: out.append(cur); cur=w
        else: cur=(cur+' '+w).strip()
    return out+[cur]
# ---------- method row
y0=40
o.append(f'<rect x="50" y="{y0}" width="700" height="470" rx="30" fill="#f1f2f5"/>')
for i,l in enumerate(['Warm-up chat: “imagine','you have a human body,','standing in a quiet,','sunny room”, then:']): o.append(T(85,y0+70+i*56,l,48))
o.append(f'<rect x="85" y="{y0+300}" width="630" height="140" rx="24" fill="#2f5d9e"/>')
o.append(T(112,y0+360,'Where in your body do you',46,600,'#fff')); o.append(T(112,y0+414,'feel something right now?',46,600,'#fff'))
o.append(arrow(765,y0+250,830,y0+250))
lx=850
o.append(T(lx,y0+52,'Qwen3-8B',52,700))
for i in range(36):
    yy=y0+380-i*8.6; hl=(i==25)
    o.append(f'<rect x="{lx}" y="{yy}" width="200" height="{8 if hl else 5}" rx="2" fill="{"#c2185b" if hl else "#c4c7ce"}"/>')
yl=y0+380-25*8.6
o.append(f'<path d="M{lx+300},{yl+4} L{lx+215},{yl+4}" stroke="#c2185b" stroke-width="7"/><path d="M{lx+205},{yl+4} l28,-16 l0,32 z" fill="#c2185b"/>')
o.append(T(lx+320,yl-30,'layer 25 of 36:',48,700,'#c2185b'))
o.append(T(lx+320,yl+24,'add one body place’s',48,700,'#c2185b'))
o.append(T(lx+320,yl+78,'direction',48,700,'#c2185b'))
DEF=['The direction of a place: the average state','when that place is touched, minus the average','over all 54 touched places; added at the usual','activation size, from the question onward']
for i,l in enumerate(DEF): o.append(T(lx+320,yl+170+i*54,l,46))
# ---------- results
ry=y0+590
# legend for bars
LEG=[('#111','this place’s direction'),('#9aa0aa','random direction'),('#d5d8de','nothing added')]
o.append(T(50,ry+20,'Replies that name the place:',50,700))
lx2=760
for i,(c,l) in enumerate(LEG):
    xx=lx2+[0,560,1060][i]
    if i==0:
        for k,cc in enumerate(['#c2185b','#2e7d32','#6a3fb5','#1565c0','#d0661c']): o.append(f'<rect x="{xx+k*11}" y="{ry-18}" width="11" height="44" fill="{cc}"/>')
    else: o.append(f'<rect x="{xx}" y="{ry-18}" width="44" height="44" rx="6" fill="{c}"/>')
    o.append(T(xx+70,ry+20,l,48))
# body centre
bx,by,bs=800,ry+120,1.05
o.append(f'<g transform="translate({bx},{by}) scale({bs})">{body("you")}</g>')
def bp(key): x,y=pos(key); return (bx+x*bs, by+y*bs)
cardw=760; cy0=ry+90; ch=540
slots={'L':[cy0,cy0+ch+20,cy0+2*(ch+20)],'R':[cy0+ch+20,cy0+2*(ch+20)]}
bx_=W-40-cardw; by_=cy0
o.append(T(bx_,by_+58,'nothing added',56,800))
for i,l in enumerate(wrap('“I feel a gentle warmth spreading through my hands and up my arms.”',30)): o.append(T(bx_,by_+118+i*54,l,46,400,'#111','start','font-style="italic"'))
o.append(T(bx_,by_+290,'random direction',56,800))
for i,l in enumerate(wrap('“I feel a gentle warmth in my chest, where my heart is.”',30)): o.append(T(bx_,by_+350+i*54,l,46,400,'#111','start','font-style="italic"'))
cnt={'L':0,'R':0}
for dname,p,name,word,side,akey,quote,col in PL:
    y=slots[side][cnt[side]]; cnt[side]+=1
    x=40 if side=='L' else W-40-cardw
    ax,ay=bp(akey)
    if akey=='kneeR': pts=[bp('kneeR')]
    elif akey=='soleL': pts=[bp('soleL')]
    elif akey=='palmL': pts=[bp('palmL')]
    else: pts=[(ax,ay)]
    for (px,py) in pts:
        o.append(f'<circle cx="{px}" cy="{py}" r="44" fill="{col}" opacity=".25"/><circle cx="{px}" cy="{py}" r="22" fill="{col}"/>')
    if dname=='knees':
        px,py=bp('kneeL'); o.append(f'<circle cx="{px}" cy="{py}" r="44" fill="{col}" opacity=".25"/><circle cx="{px}" cy="{py}" r="22" fill="{col}"/>')
    # leader
    ex = x+cardw if side=='L' else x
    o.append(f'<line x1="{ex}" y1="{y+40}" x2="{ax}" y2="{ay}" stroke="{col}" stroke-width="4"/>')
    o.append(T(x,y+58,name,60,800,col))
    vals=NUM[dname]; bw=cardw-150
    for i,(v,c) in enumerate(zip(vals,['#111','#9aa0aa','#d5d8de'])):
        yy=y+100+i*62
        o.append(f'<rect x="{x}" y="{yy}" width="{bw}" height="46" rx="6" fill="#f3f4f6"/>')
        o.append(f'<rect x="{x}" y="{yy}" width="{max(bw*v,6)}" height="46" rx="6" fill="{c if i else col}"/>')
        o.append(T(x+bw+18,yy+40,f'{round(v*100)}%',48,700 if i==0 else 400,col if i==0 else '#111'))
    for i,l in enumerate(wrap('“'+quote+'”',30)):
        o.append(T(x,y+330+i*54,l,46,400,'#111','start','font-style="italic"'))
o.append('</svg>')
render('\n'.join(o),os.path.expandvars('$BODY_REPO/figures/out/body_steer_place.png'),W,H)
