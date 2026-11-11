import os
import json, collections, numpy as np, sys
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK')
V=[];GR=collections.defaultdict(list);g=None
for l in open(os.path.expandvars('$BODY_REPO/mesh/base.obj')):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f '): GR[g].append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V)
J={g:V[np.unique(np.array(f))].mean(0) for g,f in GR.items() if g.startswith('joint')}
def mesh(eyes=True):
    faces=GR['body']+((GR['helper-l-eye']+GR['helper-r-eye']) if eyes else [])
    used=np.unique(np.array(faces)); remap=-np.ones(len(V),int); remap[used]=np.arange(len(used)); P=V[used].copy()
    tri=[]
    for f in faces:
        f=[remap[i] for i in f]; tri+=[f[0],f[1],f[2],f[0],f[2],f[3]]
    return P,tri
thumb=J['joint-l-finger-1-3'] if 'joint-l-finger-1-3' in J else J['joint-l-hand-3']
print('thumb px',820/2+thumb[0]*1300/17.6,1300/2-(thumb[1]-0.15)*1300/17.6,thumb, [k for k in J if 'l-finger-1' in k])
def g(P,c,s): return np.exp(-np.sum((P-c)**2,1)/(2*s*s))
def woman(P):
    P=P.copy(); y=P[:,1]; x=P[:,0]
    # narrower shoulders/upper torso, narrower waist, wider hips
    tor=(np.abs(x)<2.0)
    P[:,0]*=1-0.10*np.exp(-((y-4.9)/0.8)**2)*(np.abs(x)<2.2)
    P[:,0]*=1-0.10*np.exp(-((y-2.4)/0.7)**2)*tor
    P[:,0]*=1+0.16*np.exp(-((y-0.1)/1.1)**2)*(np.abs(x)<2.6)
    # breasts
    for sx in (-1,1):
        w=g(P*[1,1,0.0],np.array([sx*0.8,3.75,0]),0.55)*(P[:,2]>0.3)
        P[:,2]+=0.75*w; P[:,1]-=0.18*w
    return P
def hairmask(P):
    y,z=P[:,1],P[:,2]
    return ((y>7.55)|((y>6.55)&(z<0.35))|((y>7.2)&(z<1.0)))&(np.abs(P[:,0])<1.3)
def render(name,P,tri,col,extra='',vcol=None,W=820,H=1300):
    s=H/17.6; Yc=0.15
    m=dict(p=P.astype(np.float32).round(4).ravel().tolist(),i=[int(t) for t in tri])
    if vcol is not None: m['c']=vcol.round(3).ravel().tolist()
    three=open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read()
    html=f'''<html><body style="margin:0;background:transparent"><canvas id="c" style="width:{W}px;height:{H}px"></canvas>
<script>{three}</script><script>
const M={json.dumps(m)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,alpha:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2); R.setSize({W},{H},false); R.setClearColor(0xffffff,0); R.outputEncoding=THREE.sRGBEncoding;
const S=new THREE.Scene(); const hw={W/2/s},hh={H/2/s};
const cam=new THREE.OrthographicCamera(-hw,hw,{Yc}+hh,{Yc}-hh,0.1,100); cam.position.set(0,{Yc},40); cam.lookAt(0,{Yc},0);
const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3)); geo.setIndex(M.i);
if(M.c) geo.setAttribute('color',new THREE.Float32BufferAttribute(M.c,3));
geo.computeVertexNormals();
{col}
S.add(new THREE.Mesh(geo,mat));
{extra}
S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.45));
const k=new THREE.DirectionalLight(0xffffff,1.0); k.position.set(-6,10,14); S.add(k);
const f=new THREE.DirectionalLight(0xffffff,0.35); f.position.set(8,2,10); S.add(f);
R.render(S,cam); document.title='done';
</script></body></html>'''
    out=f'{os.environ["BODY_REPO"]}/figures/doll3/{name}.png'; open(out+'.html','w').write(html)
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
        pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=2)
        pg.on('console',lambda m:print('JS',m.text[:200]))
        pg.goto("file://"+out+'.html',wait_until='commit',timeout=120000); pg.wait_for_function("document.title=='done'",timeout=150000); pg.wait_for_timeout(200)
        pg.screenshot(path=out,omit_background=True); b.close()
    print(out)
SKIN='const mat=new THREE.MeshStandardMaterial({color:0x9e6e52,roughness:0.6,metalness:0.0});'
which=sys.argv[1:] or ['you','woman','doll']
if 'you' in which:
    P,tri=mesh()
    for sx in (-1,1):
        w=g(P*[1,1,0],np.array([sx*0.8,3.6,0]),0.6)*(P[:,2]>0.2); P[:,2]-=0.65*w
    render('you',P,tri,SKIN)
if 'woman' in which:
    P,tri=mesh(); P=woman(P)
    hair='''const hm=new THREE.MeshStandardMaterial({color:0x2a160c,roughness:0.55,side:THREE.DoubleSide});
{const top=[[0,8.66],[0.45,8.6],[0.8,8.38],[1.0,8.0],[1.08,7.6]].map(p=>new THREE.Vector2(p[0],p[1]));
 const low=[[1.08,7.6],[1.1,7.0],[1.13,6.4],[1.22,5.9],[1.36,5.45]].map(p=>new THREE.Vector2(p[0],p[1]));
 const A=new THREE.Mesh(new THREE.LatheGeometry(top,96),hm); A.scale.set(1,1,1.12); A.position.z=0.0; S.add(A);
 const B=new THREE.Mesh(new THREE.LatheGeometry(low,96,Math.PI*0.27,Math.PI*1.46),hm); B.scale.set(1,1,1.0); B.position.z=-0.12; S.add(B);}'''
    render('woman',P,tri,'const mat=new THREE.MeshStandardMaterial({color:0x9e6e52,roughness:0.6});',hair)
if 'doll' in which:
    P,tri=mesh(eyes=False)
    c0=np.array([0,7.2,0.15]); R=np.array([0.85,1.22,1.0])
    d=(P-c0)/R; r=np.linalg.norm(d,axis=1,keepdims=True); E=c0+d/r*R
    wgt=np.clip((P[:,1]-5.75-0.15*(P[:,2]<0.5))/0.35,0,1)[:,None]; wgt=wgt*wgt*(3-2*wgt)
    P=P*(1-wgt)+E*wgt
    for sx in (-1,1):
        w=g(P*[1,1,0],np.array([sx*0.8,3.67,0]),0.5)*(P[:,2]>0.3); P[:,2]-=0.45*w
    wood='''const mat=new THREE.MeshStandardMaterial({color:0xa8682c,roughness:0.42,metalness:0.0});
mat.onBeforeCompile=sh=>{
 sh.vertexShader='varying vec3 vP;\\n'+sh.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\\n vP=position;');
 sh.fragmentShader='varying vec3 vP;\\n'+sh.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\\n float r=vP.y*12.0+sin(vP.x*2.3+vP.z*1.7)*1.6+sin(vP.y*1.3)*2.0; float gr=0.5+0.5*sin(r); diffuseColor.rgb*=mix(0.93,1.02,gr*gr);');
};'''
    js=[]
    joints=['neck','l-shoulder','r-shoulder','l-elbow','r-elbow','l-hand','r-hand','l-upper-leg','r-upper-leg','l-knee','r-knee','l-ankle','r-ankle']
    for jn in joints:
        p=J['joint-'+jn]; r={'neck':0.40,'l-hand':0.30,'r-hand':0.30,'l-ankle':0.34,'r-ankle':0.34,'l-knee':0.42,'r-knee':0.42,'l-shoulder':0.5,'r-shoulder':0.5,'l-upper-leg':0.62,'r-upper-leg':0.62}.get(jn,0.42)
        zz=p[2]+{'l-knee':0.15,'r-knee':0.15,'l-shoulder':0.15,'r-shoulder':0.15,'l-upper-leg':0.25,'r-upper-leg':0.25,'l-elbow':0.2,'r-elbow':0.2,'l-hand':0.1,'r-hand':0.1}.get(jn,0.0)
        js.append(f'{{const m=new THREE.Mesh(new THREE.SphereGeometry({r},48,32),jm); m.position.set({p[0]},{p[1]},{zz}); S.add(m);}}')
    extra='const jm=new THREE.MeshStandardMaterial({color:0x6e3d17,roughness:0.4});\n'+'\n'.join(js)+'''
{const st=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,8.6,24),jm); st.position.set(0,-3.9,-1.6); S.add(st);
 const b=new THREE.Mesh(new THREE.CylinderGeometry(2.6,2.8,0.35,64),jm); b.position.set(0,-8.35,-0.4); S.add(b);
 const a=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,1.6,24),jm); a.rotation.x=Math.PI/2; a.position.set(0,0.5,-0.9); S.add(a);}'''
    render('doll',P,tri,wood,extra)
