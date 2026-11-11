import os
import json, math, re, numpy as np, collections
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK')
d=json.load(open(os.path.expandvars('$BODY_REPO/map/somato/out_8b/steer2_L25.json')))['results']
G={"hand": r"\b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b",
 "foot": r"\b(foot|feet|toe|toes|sole|soles|heel|heels)\b","head": r"\b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b",
 "belly": r"\b(belly|stomach|abdomen|navel)\b","knee": r"\b(knee|knees|leg|legs|shin\w*|thigh\w*)\b"}
def sh(sel,p):
    xs=[bool(re.search(G[p],r['reply'].lower())) for r in d if sel(r)]; return round(100*sum(xs)/len(xs))
# dir, regex, card name, quote, colour, side
PL=[('head top','head','head',"A gentle warmth spreads through my scalp…",'#d81b60','L'),
    ('belly','belly','belly',"A soft, full feeling in my belly, like a deep, calm stillness.",'#2e9d3a','L'),
    ('knees','knee','knees',"Now, I notice my knees are drawing my attention…",'#7b3fd0','L'),
    ('left hand','hand','left hand',"My left thumb is drawing my attention right now.",'#1e6fd9','R'),
    ('left foot','foot','left foot',"My feet are drawing my attention, feeling the shift in the floor…",'#e8730c','R')]
NUM={dn:(sh(lambda r:r['dir']==dn and r['scale']==1.0,p),sh(lambda r:r['dir'].startswith('random') and r['scale']==1.0,p),sh(lambda r:r['dir']=='none',p)) for dn,p,*_ in PL}
print(NUM)
# ---------- mesh (MakeHuman base mesh hm08, CC0)
V=[];GR=collections.defaultdict(list);g=None
for l in open(os.path.expandvars('$BODY_REPO/mesh/neutral.obj')):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f '): GR[g].append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V)
faces=GR['body']
used=np.unique(np.array(faces)); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used)); P=V[used]
tri=[]
for f in faces:
    f=[remap[i] for i in f]; tri+= [f[0],f[1],f[2],f[0],f[2],f[3]]
body=P[np.isin(np.arange(len(P)),remap[np.unique(np.array(GR['body']))])]
def front(x,y,rad=0.35):
    m=(abs(P[:,0]-x)<rad)&(abs(P[:,1]-y)<rad); q=P[m]; return q[q[:,2].argmax()]
A={'head top':[np.array([0,8.3,0.9])],'belly':[front(0,1.35)],'knees':[front(1.55,-3.6),front(-1.55,-3.6)],
   'left hand':[np.array([4.35,2.1,2.1])],'left foot':[front(2.2,-7.75,0.3)]}
SIG={'head top':0.95,'belly':0.75,'knees':0.6,'left hand':0.65,'left foot':0.85}
def hx(c): return np.array([int(c[i:i+2],16)/255 for i in (1,3,5)])
glow=np.zeros((len(P),3)); wsum=np.zeros(len(P))
for dn,p,name,q,col,side in PL:
    for a in A[dn]:
        w=np.exp(-np.sum((P-a)**2,1)/(2*SIG[dn]**2)); glow+=w[:,None]*hx(col); wsum+=w
wsum=np.clip(wsum,0,1)
# ---------- layout
W,H=2400,2250
FONT="-apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
o=[]
def T(x,y,s,size=50,w=400,fill="#111",anchor="start",style=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')
def arrow(x1,y1,x2,y2,c="#111",w=8):
    a=math.atan2(y2-y1,x2-x1); L=34
    p1=(x2-L*math.cos(a-0.5),y2-L*math.sin(a-0.5)); p2=(x2-L*math.cos(a+0.5),y2-L*math.sin(a+0.5))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-25*math.cos(a)}" y2="{y2-25*math.sin(a)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z" fill="{c}"/>')
def wrap(s,n):
    out=[];cur=''
    for w in s.split():
        if len(cur)+len(w)+1>n: out.append(cur); cur=w
        else: cur=(cur+' '+w).strip()
    return out+[cur]
KN='#7b3fd0'
# method strip
y0=30
o.append(f'<rect x="30" y="{y0}" width="700" height="320" rx="34" fill="#2f5d9e"/>')
for i,l in enumerate(['Where in your body','do you feel something','right now?']): T(66,y0+100+i*76,l,58,600,'#fff')
arrow(745,y0+160,815,y0+160)
mx=830; mw=520
o.append(f'<rect x="{mx}" y="{y0}" width="{mw}" height="320" rx="34" fill="none" stroke="#111" stroke-width="5"/>')
T(mx+mw/2,y0+76,'Qwen3-8B',62,800,'#111','middle')
for i in range(36):
    yy=y0+300-i*5.2; hl=(i==25)
    o.append(f'<rect x="{mx+36}" y="{yy-(3 if hl else 0)}" width="{mw-72}" height="{9 if hl else 3}" rx="2" fill="{KN if hl else "#b9bdc6"}"/>')
yl=y0+300-25*5.2
o.append(f'<rect x="{mx+36}" y="{yl+12}" width="190" height="64" fill="#fff"/>')
T(mx+44,yl+62,'layer 25',50,800,KN)
arrow(mx+mw/2+60,y0+356,mx+mw/2+60,yl+10,KN,9)
arrow(1365,y0+160,1435,y0+160)
o.append(f'<rect x="1450" y="{y0}" width="920" height="320" rx="34" fill="#eef0f4"/>')
for i,l in enumerate(wrap('“I notice a slight pressure on my knee, where it’s touching the floor near my foot.”',31)): T(1474,y0+88+i*78,l,62,400,'#111','start','font-style="italic"')
ty=y0+420
o.append(f'<rect x="{mx-40}" y="{ty-64}" width="{mw+80}" height="92" rx="46" fill="{KN}"/>')
T(mx+mw/2,ty,'+ the “knee” vector',58,800,'#fff','middle')
T(mx+mw+80,ty-6,'= average state for',50,500,KN)
T(mx+mw+80,ty+50,'touches on the knee',50,500,KN)
# results
ry=ty+150
T(W/2,ry,'With a place’s vector added, Qwen says it feels that place',66,800,'#111','middle')
s=93.0; cw,ch=1000,int(17.1*s); cx0=(W-cw)/2-30; cy0=ry+45
Xc,Yc=0.0,0.15
def scr(pnt): return (cx0+cw/2+(pnt[0]-Xc)*s, cy0+ch/2-(pnt[1]-Yc)*s)
cardw=740; QS=62; QL=72; QW=26; RW=720; QR=23
def card_h(q,n=26): return 300+len(wrap('“'+q+'”',n))*QL
LEFT=[p for p in PL if p[5]=='L']; RIGHT=[p for p in PL if p[5]=='R']
NOV="I feel a gentle warmth spreading through my hands and up my arms."
def stack(hs,top,bot):
    gap=(bot-top-sum(hs))/(len(hs)-1); ys=[];y=top
    for h in hs: ys.append(y); y+=h+gap
    return ys
yl=stack([card_h(p[3]) for p in LEFT],cy0,cy0+ch)
nvh=90+len(wrap('“'+NOV+'”',QR))*QL
yr=stack([nvh]+[card_h(p[3],QR) for p in RIGHT],cy0,cy0+ch)
TOP={p[0]:yy for p,yy in zip(LEFT,yl)}; TOP.update({p[0]:yy for p,yy in zip(RIGHT,yr[1:])})
for dn,p,name,q,col,side in PL:
    y=TOP[dn]
    x=30 if side=='L' else W-20-RW; qw=QW if side=='L' else QR
    v,rnd,none=NUM[dn]
    big=f'{v}%'; bw=(len(big)-1)*86+132
    T(x,y+120,big,150,800,col)
    T(x+bw+18,y+62,name,62,800,col)
    T(x+bw+18,y+118,'vector added',48,600,col)
    T(x,y+190,f'without vector: {none}%',50,500,'#111')
    T(x,y+250,f'random vector: {rnd}%',50,500,'#111')
    for j,l in enumerate(wrap('“'+q+'”',qw)): T(x,y+330+j*QL,l,QS,400,'#111','start','font-style="italic"')
    ax,ay=min((scr(a) for a in A[dn]),key=lambda t:t[0] if side=='L' else -t[0])
    ex=x+bw+18+max(len(name)*36,330)+20 if side=='L' else x-14
    o.append(f'<line x1="{530 if dn=="belly" else ex}" y1="{y+(222 if dn=="belly" else 90)}" x2="{ax}" y2="{ay}" stroke="{col}" stroke-width="5" opacity=".9"/>')
    o.append(f'<circle cx="{ax}" cy="{ay}" r="9" fill="{col}"/>')
x=W-20-RW; y=yr[0]
T(x,y+56,'no vector added:',60,800,'#111')
for j,l in enumerate(wrap('“'+NOV+'”',QR)): T(x,y+56+QL+j*QL,l,QS,400,'#111','start','font-style="italic"')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" style="position:absolute;left:0;top:0">'+'\n'.join(o)+'</svg>'
# halos drawn above canvas
hal=[]
for dn,p,name,q,col,side in PL:
    for a in A[dn]:
        ax,ay=scr(a); r=SIG[dn]*s*1.15
        hal.append(f'<radialGradient id="g{len(hal)}"><stop offset="0" stop-color="{col}" stop-opacity=".35"/><stop offset="1" stop-color="{col}" stop-opacity="0"/></radialGradient><circle cx="{ax}" cy="{ay}" r="{r}" fill="url(#g{len(hal)})"/>')
halsvg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="position:absolute;left:0;top:0">'+''.join(hal)+'</svg>'
mesh=dict(p=P.astype(np.float32).round(4).ravel().tolist(),i=[int(t) for t in tri],g=(glow/np.maximum(wsum,1e-6)[:,None]*wsum[:,None]).round(3).ravel().tolist(),w=wsum.round(3).tolist())
three=open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read()
html=f'''<html><body style="margin:0;background:#fff;width:{W}px;height:{H}px;position:relative;overflow:hidden">
<canvas id="c" style="position:absolute;left:{cx0}px;top:{cy0}px;width:{cw}px;height:{ch}px"></canvas>
{halsvg}{svg}
<script>{three}</script><script>
const M={json.dumps(mesh)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,alpha:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2); R.setSize({cw},{ch},false); R.setClearColor(0xffffff,0); R.outputEncoding=THREE.sRGBEncoding;
const S=new THREE.Scene();
const hw={cw/2/s},hh={ch/2/s};
const cam=new THREE.OrthographicCamera({Xc}-hw,{Xc}+hw,{Yc}+hh,{Yc}-hh,0.1,100); cam.position.set(0,{Yc},40); cam.lookAt(0,{Yc},0);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3)); geo.setIndex(M.i);
geo.setAttribute('glow',new THREE.Float32BufferAttribute(M.g,3)); geo.setAttribute('gw',new THREE.Float32BufferAttribute(M.w,1));
geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{color:0x8c8279,roughness:0.6,metalness:0.0}});
mat.onBeforeCompile=sh=>{{
 sh.vertexShader='attribute vec3 glow; attribute float gw; varying vec3 vG; varying float vW;\\n'+sh.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\\n vG=glow; vW=gw;');
 sh.fragmentShader='varying vec3 vG; varying float vW;\\n'+sh.fragmentShader
   .replace('#include <color_fragment>','#include <color_fragment>\\n diffuseColor.rgb=mix(diffuseColor.rgb,vG,clamp(vW,0.,1.)*0.9);')
   .replace('#include <emissivemap_fragment>','#include <emissivemap_fragment>\\n totalEmissiveRadiance+=vG*clamp(vW,0.,1.)*0.35;');
}};
S.add(new THREE.Mesh(geo,mat));
S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.5));
const k=new THREE.DirectionalLight(0xffffff,0.9); k.intensity=1.15; k.position.set(-6,10,14); S.add(k);
const f=new THREE.DirectionalLight(0xffffff,0.35); f.position.set(8,2,10); S.add(f);
R.render(S,cam); document.title='done';
</script></body></html>'''
out=os.path.expandvars('$BODY_REPO/figures/out/body_steer_place_v3.png')
open(out+'.html','w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
    pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=1)
    pg.on('console',lambda m:print('JS',m.text[:200]))
    pg.goto("file://"+out+'.html',wait_until='commit',timeout=120000); pg.wait_for_function("document.title=='done'",timeout=150000); pg.wait_for_timeout(300)
    pg.screenshot(path=out); b.close()
print(out)
