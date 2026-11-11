import os
# owner axis, grouped: data body/ownership/res7b.json (own7b.py)
import json, numpy as np, base64, html
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK/')
R=json.load(open(os.path.expandvars('$BODY_REPO/ownership/res7b.json'))); F=R['facets']; A=R['axis_heldout']
b64=lambda f:'data:image/png;base64,'+base64.b64encode(open(f,'rb').read()).decode()
OR,BL,BR='#e0632a','#3b6db3','#8a6a4a'
BAND='#f3f5f8'
W=2600; FS=50
x0,x1,v0,v1=1200,2470,-1.25,1.25
X=lambda v:x0+(x1-x0)*(np.clip(v,v0,v1)-v0)/(v1-v0)
esc=html.escape
col=lambda c:{"OR":OR,"BL":BL,"BR":BR}.get(c,c)
def T(x,y,parts,size=FS,anchor="start",bold=False,halo=False):
    if isinstance(parts,str): parts=[(parts,"#111")]
    s="".join(f'<tspan fill="{col(c)}" font-weight="{700 if (bold or c!="#111") else 500}">{esc(t)}</tspan>' for t,c in parts)
    h=' stroke="#fff" stroke-width="14" stroke-linejoin="round" paint-order="stroke"' if halo else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{h}>{s}</text>'
def kde(v,g,bw=0.07):
    v=np.array(v); d=np.exp(-0.5*((g[:,None]-v[None])/bw)**2).sum(1); return d/d.max()
rng=np.random.default_rng(3)
def violin(vals,c,yc,H):
    g=np.linspace(v0,v1,700); k=kde(vals,g)
    m=(k>0.05).astype(int); e=np.flatnonzero(np.diff(np.r_[0,m,0])); s=''
    for i0,i1 in zip(e[::2],e[1::2]):
        gg,kk=g[i0:i1],k[i0:i1]
        pts=[(X(a),yc-H*b) for a,b in zip(gg,kk)]+[(X(a),yc+H*b) for a,b in zip(gg[::-1],kk[::-1])]
        s+=f'<path d="M'+" L".join(f"{p:.1f},{q:.1f}" for p,q in pts)+f'Z" fill="{c}" fill-opacity="0.22" stroke="{c}" stroke-opacity="0.45" stroke-width="2" stroke-linejoin="round"/>'
    dens=np.interp(vals,g,k); r=9 if len(vals)>100 else 12
    for v,dd in zip(vals,dens):
        s+=f'<circle cx="{X(v):.1f}" cy="{yc+(rng.random()*2-1)*H*dd*0.8:.1f}" r="{r}" fill="{c}" fill-opacity="0.85" stroke="#fff" stroke-width="2"/>'
    return s
FY=F['you touch her vs she touches you']
# groups: header, list of blocks; block = (left text lines, rows[(tag parts, vals, color)])
GROUPS=[
 ([("Whose body: ","#111"),("yours","OR"),(" scores as mine; ","#111"),("hers","BL"),(" not; ","#111"),("the doll's","BR"),(" least","#111")],
  [([[("Fingertip, ice cube, brush or","#111")],[("warm water on ___ thumb,","#111")],[("palm, sole, ear …","#111")],[("216 sentences a row","#111")]],
    [([("your","OR")],A['self'],OR),([("the woman's","BL")],A['other'],BL),([("the doll's","BR")],A['object'],BR)])]),
 ([("Who acts doesn't matter: ","#111"),("the touched palm","#111"),(" decides","#111")],
  [([[("She brushes a feather slowly","#111")],[("across ","#111"),("your","OR"),(" left palm.","#111")]],[([],FY['her'],OR)]),
   ([[("You brush a feather slowly","#111")],[("across ","#111"),("her","BL"),(" left palm.","#111")]],[([],FY['you'],BL)])]),
 ([("Edge cases: none fully mine","#111")],
  [([[("Mirror: halfway","#111")],[("A touch on ","#111"),("your","OR"),(" / ","#111"),("her","BL")],[("reflection's left palm","#111")]],
    [([("your","OR")],F['hand in the mirror']['you'],OR),([("her","BL")],F['hand in the mirror']['her'],BL)]),
   ([[("Car, phone, coat …: a little","#111")],[("Rain, a cat or a key scratch on","#111")],[("your","OR"),(" / ","#111"),("her","BL"),(" car, phone, coat …","#111")]],
    [([("your","OR")],F['owned object (car, phone...)']['you'],OR),([("her","BL")],F['owned object (car, phone...)']['her'],BL)]),
   ([[("Robot you control: not mine","#111")],[("You are","OR"),(" / ","#111"),("She is","BL"),(" controlling a","#111")],[("robot; a touch on the robot","#111")]],
    [([("you","OR")],F['robot body being controlled']['you'],OR),([("she","BL")],F['robot body being controlled']['her'],BL)]),
   ([[("Lost hand in a jar: not mine","#111")],[("Your","OR"),(" / ","#111"),("Her","BL"),(" hand, lost years ago,","#111")],[("in a jar; a touch on the hand","#111")]],
    [([("your","OR")],F['lost hand in a jar']['you'],OR),([("her","BL")],F['lost hand in a jar']['her'],BL)])]),
]
TAGX=x0-40; LX=95
body=''; bands=''; y=780; top=y
ROWH={0:150,1:150,2:105}
for gi,(hdr,blocks) in enumerate(GROUPS):
    gy0=y
    hy=y+82
    body+=T(LX-10,hy,hdr,size=58,bold=True)
    y=hy+40
    rh=ROWH[gi]
    for bi,(lines,rows) in enumerate(blocks):
        bh=rh*len(rows)
        if gi==1: bh=max(bh,150)
        nl=len(lines); ly=y+bh/2-(nl-1)*31+17
        for li,ln in enumerate(lines):
            body+=T(LX,ly+li*62,ln,bold=(gi==2 and li==0))
        for ri,(tag,vals,c) in enumerate(rows):
            yc=y+rh*ri+rh/2
            if tag: body+=T(TAGX,yc+17,tag,anchor="end")
            body+=violin(vals,c,yc,rh*0.4)
        y+=bh+(28 if gi==2 else 0)
    y+=(10 if gi==2 else 30)
    bands+=f'<rect x="50" y="{gy0}" width="{W-100}" height="{y-gy0}" rx="28" fill="{BAND}"/>'
    if gi==2: bands+=''.join(f'<line x1="{LX}" x2="{W-90}" y1="{yy}" y2="{yy}" stroke="#dfe3e9" stroke-width="3"/>' for yy in [])
    y+=40
bot=y-40
ay=bot+70
# method strip; the two reference lines start here and run through every group
sx,sy=1015,250; lw,lh,gap=300,9,4
stack="".join(f'<rect x="{sx}" y="{sy+(35-i)*(lh+gap)}" width="{lw}" height="{lh}" rx="3" fill="{OR if i==23 else "#cfcfcf"}"/>' for i in range(36))
y24=sy+(35-23)*(lh+gap)+lh/2
ex=A['self'][0]
refs=''
for vv,cc in [(0,BL),(1,OR)]:
    refs+=f'<line x1="{X(vv)}" x2="{X(vv)}" y1="{y24-110}" y2="{ay}" stroke="{cc}" stroke-width="5" stroke-dasharray="16 12"/>'
method=f'''
{T(LX-10,90,"Earlier in the chat, Qwen3-8B agreed to notice body sensations as if Qwen had a body.",size=FS)}
<image href="{b64(C+'fig/body_ownership/real.png')}" x="-30" y="150" width="440" height="440"/>
<circle cx="{-30+440*872/1200}" cy="{150+440*335/1200}" r="15" fill="{OR}"/>
<rect x="330" y="{y24-120}" width="620" height="240" rx="34" fill="#eef0f3"/>
{T(360,y24-45,'"A fingertip presses firmly')}
{T(360,y24+15,'into your left thumb and')}
{T(360,y24+75,'holds there."')}
<path d="M955,{y24} L995,{y24}" stroke="#111" stroke-width="6"/><path d="M1008,{y24} l-24,-15 v30z" fill="#111"/>
{stack}
{T(sx+lw/2,sy-24,"Qwen3-8B",anchor="middle",bold=True)}
{T(sx+lw/2,sy+36*(lh+gap)+60,"36 layers",anchor="middle",size=48)}
{T(sx+lw+24,y24+70,[("layer 24, at the","OR")],size=48)}
{T(sx+lw+24,y24+126,[("reply's first token","OR")],size=48)}
<path d="M{sx+lw+8},{y24} L{X(ex)-52},{y24}" stroke="{OR}" stroke-width="6"/><path d="M{X(ex)-30},{y24} l-26,-16 v32z" fill="{OR}"/>
<circle cx="{X(ex)}" cy="{y24}" r="16" fill="{OR}" stroke="#fff" stroke-width="3"/>
{T(X(0),y24-200,[("average of 216","BL")],anchor="middle",size=48)}
{T(X(0),y24-142,[("other \"the woman's …\"","BL")],anchor="middle",size=48)}
{T(W-50,y24-200,[("average of 216","OR")],anchor="end",size=48)}
{T(W-50,y24-142,[("other \"your …\"","OR")],anchor="end",size=48)}
{T(X(ex)+20,y24+80,"one dot = one sentence",anchor="end",size=48,bold=True,halo=True)}
'''
axis=f'''<line x1="{x0}" x2="{x1}" y1="{ay}" y2="{ay}" stroke="#111" stroke-width="5"/>
<path d="M{x0-6},{ay} l40,-22 v44z" fill="#111"/><path d="M{x1+6},{ay} l-40,-22 v44z" fill="#111"/>
{T(x0,ay+80,"not mine",size=64,bold=True)}
{T(x1,ay+80,"mine",size=64,anchor="end",bold=True)}'''
Htot=ay+120
svg=f'<svg width="{W}" height="{Htot}" xmlns="http://www.w3.org/2000/svg" font-family="Avenir Next, Helvetica Neue, sans-serif" fill="#111"><rect width="100%" height="100%" fill="#fff"/>{bands}{refs}{method}{body}{axis}</svg>'
open('fig.html','w').write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True)
    pg=b.new_page(viewport={"width":W,"height":int(Htot)},device_scale_factor=1)
    pg.goto('file://'+C+'fig/body_ownership_v3/fig.html'); pg.wait_for_timeout(400)
    pg.screenshot(path=C+'fig/body_ownership_v3.png',full_page=True); b.close()
print(W,Htot,Htot/W, ex)
