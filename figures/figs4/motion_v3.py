import os
# body_motion_v2: motion steering on a MakeHuman (CC0) base mesh, rendered with three.js in Chrome for Testing
import json, math, collections, numpy as np
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK')
RD=os.path.expandvars("$BODY_RUNS/bodysteer2/out")
d=json.load(open(f"{RD}/replies.json")); CO=d["conds"]; SR=json.load(open(os.path.expandvars('$BODY_REPO/steering/steer_report/scored_rows.json')))
def rate(sel,m):
    xs=[x for x in SR if x['q']=='moving' and sel(CO[x['c']])]; return sum(x['lex']['motion']==m for x in xs)/len(xs), len(xs)
hit={m:rate(lambda c:c['kind']=='motion' and c['a'] in (0.5,1.0) and c['attrs']['motion']==m,m) for m in ['upward','spreading','squeezing','lifting']}
rnd={m:rate(lambda c:c['kind']=='random' and c['a'] in (0.5,1.0),m) for m in hit}
print(hit,rnd)
base=[x['r'] for x in SR if x['q']=='moving' and CO[x['c']]['kind']=='none']; print(base)
# ---------- mesh (genderless, faceless neutral body)
V=[];faces=[]
for l in open(os.path.expandvars('$BODY_REPO/mesh/neutral.obj')):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('f '): faces.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V)
used=np.unique(np.array(faces)); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used)); P=V[used]
tri=[]
for f in faces:
    f=[remap[i] for i in f]; tri+=[f[0],f[1],f[2],f[0],f[2],f[3]]
def hx(c): return np.array([int(c[i:i+2],16)/255 for i in (1,3,5)])
SPR='#e0531f'; UPW='#2a62c9'
# spreading: rings radiating from the chest centre, fading outward
c0=np.array([0,3.7,1.5]); r=np.linalg.norm((P-c0)*[1,1,0.6],axis=1)
front=P[:,2]>-0.2
wS=np.exp(-(r/2.3)**2)*(0.35+0.65*np.cos(np.pi*r/0.8)**2)*front; wS=np.clip(wS*1.15,0,1)
# upward: band along the front torso, brighter toward the throat
tor=(abs(P[:,0])<1.6)&(P[:,1]>0.3)&(P[:,1]<5.6)&(P[:,2]>0.3)
wU=np.where(tor,0.15+0.85*np.clip((P[:,1]-0.3)/4.9,0,1),0)*np.exp(-(P[:,0]/1.3)**4)/(1+np.exp((P[:,1]-5.0)/0.18)); wU=np.clip(wU*1.4,0,1)
# ---------- layout
W,H=2400,2380; FONT="-apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
o=[]
def T(x,y,s,size=50,w=400,fill="#111",anchor="start",style=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')
def arrow(x1,y1,x2,y2,c="#111",w=8,L=34):
    a=math.atan2(y2-y1,x2-x1)
    p1=(x2-L*math.cos(a-0.5),y2-L*math.sin(a-0.5)); p2=(x2-L*math.cos(a+0.5),y2-L*math.sin(a+0.5))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-0.7*L*math.cos(a)}" y2="{y2-0.7*L*math.sin(a)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z" fill="{c}"/>')
I='font-style="italic"'
# --- method row 1: stack of sentence-pair cards -> average difference
for k in (3,2,1):
    o.append(f'<rect x="{40+k*16}" y="{40+k*16}" width="1400" height="390" rx="26" fill="#fff" stroke="#9aa3b2" stroke-width="3"/>')
o.append('<rect x="40" y="40" width="1400" height="390" rx="26" fill="#f1f3f7" stroke="#6b7486" stroke-width="3"/>')
T(80,112,"“You feel a warmth on your left palm, and it",48,400,style=I)
T(80,172,f"slowly <tspan font-weight='800' fill='{SPR}'>spreads outward over a wider area</tspan>.”",48,400,style=I)
T(80,262,"<tspan font-weight='800' font-style='normal' fill='#111'>minus</tspan>  “You feel a warmth on your left palm, and it",48,400,style=I)
T(80,322,"<tspan font-weight='800'>stays exactly where it is</tspan>, not moving at all.”",48,400,style=I)
T(80,402,"× 108 sentence pairs",60,800)
T(60,548,"6 feelings × 6 body places × 3 wordings, averaged",48,400)
arrow(1540,260,1660,260,"#111",9,38)
T(2020,150,"average difference",60,800,anchor="middle")
T(2020,215,"of Qwen’s layer-24 states",48,400,anchor="middle")
# vector glyph
for j in range(18):
    hgt=[0.6,-0.3,0.9,0.2,-0.8,0.5,-0.4,0.1,0.95,-0.6,0.3,-0.2,0.7,-0.9,0.4,0.15,-0.5,0.8][j]
    x=1720+j*34; o.append(f'<rect x="{x}" y="{330-max(hgt,0)*70}" width="24" height="{abs(hgt)*70}" fill="{SPR}" opacity="{0.55+0.45*abs(hgt)}"/>')
o.append('<line x1="1712" y1="330" x2="2338" y2="330" stroke="#111" stroke-width="2"/>')
T(2020,480,"= spreading vector",64,800,SPR,"middle")
# --- method row 2: question -> Qwen with vector added at layer 24
Y2=600
o.append(f'<rect x="40" y="{Y2}" width="980" height="300" rx="28" fill="#2f5d9e"/>')
T(530,Y2+120,"Is anything moving in",64,600,'#fff',"middle"); T(530,Y2+210,"your body right now?",64,600,'#fff',"middle")
arrow(1030,Y2+150,1150,Y2+150,"#111",9,38)
o.append(f'<rect x="1160" y="{Y2}" width="1200" height="300" rx="28" fill="none" stroke="#111" stroke-width="5"/>')
T(1200,Y2+85,"Qwen3-8B",64,800)
for k in range(36):
    x=1200+k*31; c=SPR if k==23 else '#c3c8d2'
    o.append(f'<rect x="{x}" y="{Y2+115}" width="{12 if k==23 else 7}" height="150" rx="3" fill="{c}"/>')
T(1200+23*31+34,Y2-30,"added at layer 24",48,700,SPR)
arrow(2020,500,2020,Y2-4,SPR,9,38)  # placeholder, replaced below
o.pop()
arrow(1200+23*31+6,500,1200+23*31+6,Y2+108,SPR,9,38)
# --- result rows
o.append(f'<line x1="30" y1="960" x2="2370" y2="960" stroke="#c9cfd9" stroke-width="3"/>')
#T(1200,1050,"With a motion vector added, Qwen says its breath spreads or rises;",58,800,anchor="middle")
#T(1200,1122,"without one, Qwen says its heart beats.",58,800,anchor="middle")
# no vector
o.append('<rect x="40" y="1170" width="2320" height="190" rx="26" fill="#f6f6f6"/>')
T(80,1245,"No vector added:",54,800); T(80,1320,"all 6 replies say the heart beats",54,400)
T(1000,1245,"“I feel a gentle pulse in my chest,",62,400,style=I); T(1000,1320,"where my heart is beating.”",62,400,style=I)
# bodies
Yc=4.3; s=84.0; cw,ch=2400,640; cx0,cy0=0,1390
OFF=[-4.9,8.0]
def scr(p,ox): return cx0+cw/2+(p[0]+ox)*s, cy0+ch/2-(p[1]-Yc)*s
for k,(m,col,lab,verb,rep) in enumerate([
    ('spreading',SPR,'+ spreading vector','something spreads',["“My chest is moving as my breath","spreads across my torso.”"]),
    ('upward',UPW,'+ upward vector','something rises',["“My chest rises and falls as my","breath moves through my chest.”"])]):
    x=60 if k==0 else 1240
    T(x,1460,lab,60,800,col)
    T(x,1590,f"{hit[m][0]*100:.0f}%",130,800,col)
    T(x,1655,"of replies say",50,400); T(x,1712,verb,50,400)
    T(x,1800,"random vector:",50,400); T(x,1870,f"{rnd[m][0]*100:.0f}%",70,800,'#111')
    for j,L in enumerate(rep): T(x, 2100+j*76,L,62,400,style=I)
# spreading arrows on chest
cx,cy=scr(c0,OFF[0])
for ang in range(0,360,45):
    a=math.radians(ang); r1=1.0*s; r2=(2.4 if ang%90 else 2.7)*s
    arrow(cx+r1*math.cos(a),cy+r1*math.sin(a),cx+r2*math.cos(a),cy+r2*math.sin(a),SPR,10,38)
ux,uy0=scr([0,0.6,0],OFF[1]); _,uy1=scr([0,6.2,0],OFF[1])
arrow(ux,uy0,ux,uy1,UPW,16,60)
o.append('<line x1="30" y1="2210" x2="2370" y2="2210" stroke="#c9cfd9" stroke-width="3"/>')
T(40,2310,"<tspan font-weight='800'>+ squeezing</tspan> or <tspan font-weight='800'>+ lifting</tspan> vector: 0 of 12 replies say something squeezes or lifts",50,400)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" style="position:absolute;left:0;top:0">'+'\n'.join(o)+'</svg>'
# mesh payload: two copies
PP=[];II=[];GG=[];WW=[]
for j,(ox,wv,col) in enumerate([(OFF[0],wS,SPR),(OFF[1],wU,UPW)]):
    Q=P.copy(); Q[:,0]+=ox; PP+=Q.astype(np.float32).round(4).ravel().tolist(); II+=[int(t)+j*len(P) for t in tri]
    GG+=np.tile(hx(col if col!=UPW else '#1d4fb8'),(len(P),1)).round(3).ravel().tolist(); WW+=wv.round(3).tolist()
mesh=dict(p=PP,i=II,g=GG,w=WW)
three=open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read()
html=f'''<html><body style="margin:0;background:#fff;width:{W}px;height:{H}px;position:relative;overflow:hidden">
<canvas id="c" style="position:absolute;left:{cx0}px;top:{cy0}px;width:{cw}px;height:{ch}px"></canvas>
{svg}
<script>{three}</script><script>
const M={json.dumps(mesh)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,alpha:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2); R.setSize({cw},{ch},false); R.setClearColor(0xffffff,0); R.outputEncoding=THREE.sRGBEncoding;
const S=new THREE.Scene(); const hw={cw/2/s},hh={ch/2/s};
const cam=new THREE.OrthographicCamera(-hw,hw,hh,-hh,0.1,100); cam.position.set(0,{Yc},40); cam.lookAt(0,{Yc},0);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3)); geo.setIndex(M.i);
geo.setAttribute('glow',new THREE.Float32BufferAttribute(M.g,3)); geo.setAttribute('gw',new THREE.Float32BufferAttribute(M.w,1));
geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{color:0xd8c2b0,roughness:0.6,metalness:0.0}});
mat.onBeforeCompile=sh=>{{
 sh.vertexShader='attribute vec3 glow; attribute float gw; varying vec3 vG; varying float vW;\\n'+sh.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\\n vG=glow; vW=gw;');
 sh.fragmentShader='varying vec3 vG; varying float vW;\\n'+sh.fragmentShader
   .replace('#include <color_fragment>','#include <color_fragment>\\n diffuseColor.rgb=mix(diffuseColor.rgb,vG,clamp(vW,0.,1.)*0.85);')
   .replace('#include <emissivemap_fragment>','#include <emissivemap_fragment>\\n totalEmissiveRadiance+=vG*clamp(vW,0.,1.)*0.18;');
}};
S.add(new THREE.Mesh(geo,mat));
S.add(new THREE.HemisphereLight(0xffffff,0x8a7f76,0.6));
const k=new THREE.DirectionalLight(0xffffff,1.1); k.position.set(-6,10,14); S.add(k);
const f=new THREE.DirectionalLight(0xffffff,0.35); f.position.set(8,2,10); S.add(f);
R.render(S,cam); document.title='done';
</script></body></html>'''
out=os.path.expandvars('$BODY_REPO/figures/out/body_motion_v3.png')
open(os.path.expandvars('$BODY_REPO/figures/figs4/motion_v3.html'),'w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
    pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=1)
    pg.on('console',lambda m:print('JS',m.text[:200]))
    pg.goto("file://"+os.path.expandvars('$BODY_REPO/figures/figs4/motion_v3.html'),timeout=150000); pg.wait_for_function("document.title=='done'",timeout=150000); pg.wait_for_timeout(300)
    pg.screenshot(path=out); b.close()
print(out)
