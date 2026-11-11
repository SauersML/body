import os
import json, numpy as np, collections, sys
from playwright.sync_api import sync_playwright
H=os.path.expandvars('$BODY_REPO/mesh/')
def load(p):
    V=[];GR=collections.defaultdict(list);g=None
    for l in open(p):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('g '): g=l.split()[1]
        elif l.startswith('f '): GR[g].append([int(t.split('/')[0])-1 for t in l.split()[1:]])
    return np.array(V),GR
VB,GB=load(H+'base.obj'); V,G=load(H+'neutral.obj')
J={g:VB[np.unique(np.array(f))].mean(0) for g,f in GB.items() if g.startswith('joint')}
w=J['joint-l-hand']; e=J['joint-l-elbow']; ax=(w-e)/np.linalg.norm(w-e)
F=np.array(G['body']); P=V
c=w-ax*0.9  # include a bit of forearm
keep=((P-c)@ax>0)&(np.linalg.norm(P-w,axis=1)<3.0)
fk=F[keep[F].all(1)]; used=np.unique(fk); rm=-np.ones(len(P),int); rm[used]=np.arange(len(used)); Q=P[used]-w; fk=rm[fk]
# PCA frame: x along arm, z = palm normal
U_,S_,Vt=np.linalg.svd(Q-Q.mean(0),full_matrices=False); A=Vt
if A[0]@ax<0: A[0]*=-1
A[2]=np.cross(A[0],A[1]); Q=Q@A.T
Q=Q[:,[1,0,2]]*np.array([1,-1,1])  # fingers point down? rotate so fingers point up
Q[:,1]*=-1
tri=np.hstack([fk[:,[0,1,2]],fk[:,[0,2,3]]]).ravel()
M=dict(p=Q.astype(np.float32).round(5).ravel().tolist(),i=tri.tolist())
three=open(H+'three.min.js').read()
def render(col,out,flipz):
    W=Hh=1200; ext=float(np.abs(Q[:,:2]).max())*1.05
    html=f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)};const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true,alpha:true}});
R.setPixelRatio(2);R.setSize({W},{Hh},false);R.setClearColor(0xffffff,0);R.outputEncoding=THREE.sRGBEncoding;
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{color:{col},roughness:0.55,side:THREE.DoubleSide}});const S=new THREE.Scene();const m=new THREE.Mesh(geo,mat);m.rotation.y={'Math.PI' if flipz else 0};S.add(m);
S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.55));const d=new THREE.DirectionalLight(0xffffff,1.1);d.position.set(-4,8,14);S.add(d);
const cam=new THREE.OrthographicCamera(-{ext},{ext},{ext},-{ext},0.1,100);cam.position.set(0,0,40);cam.lookAt(0,0,0);R.render(S,cam);document.title='done';</script></body></html>'''
    open('/tmp/hand.html','w').write(html)
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
        pg=b.new_page(viewport={"width":W,"height":Hh}); pg.goto("file:///tmp/hand.html",wait_until='commit'); pg.wait_for_function("document.title=='done'",timeout=120000); pg.wait_for_timeout(200)
        pg.screenshot(path=out,omit_background=True); b.close()
flip=len(sys.argv)>1
render('0x8c8279','real.png',flip); render('0xdc8e98','rubber.png',flip)
