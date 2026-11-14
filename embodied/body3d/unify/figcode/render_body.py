import json, numpy as np
from playwright.sync_api import sync_playwright
H='$BODY_WORK/fig/human/'
V=[];F=[];g=None
for l in open(H+'neutral.obj'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V);F=np.array(F); used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
tri=np.hstack([F[:,[0,1,2]],F[:,[0,2,3]]]).ravel()
W,Hh=1000,1500
M=dict(p=P.astype(np.float32).round(5).ravel().tolist(),i=tri.tolist())
three=open(H+'three.min.js').read()
html=f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true,alpha:true}});
R.setPixelRatio(2);R.setSize({W},{Hh},false);R.setClearColor(0xffffff,0);R.outputEncoding=THREE.sRGBEncoding;
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
const S=new THREE.Scene();
S.add(new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{color:0x8f877f,roughness:0.65}})));
// probe rod: from A to tip T
const T=new THREE.Vector3(-1.78,-5.0,0.55), A=new THREE.Vector3(-4.6,-6.6,2.2);
const d=new THREE.Vector3().subVectors(T,A);const L=d.length();
const rod=new THREE.Mesh(new THREE.CylinderGeometry(0.09,0.09,L,24),new THREE.MeshStandardMaterial({{color:0x0a0a0a,roughness:0.5}}));
rod.position.copy(A).addScaledVector(d,0.5);rod.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.clone().normalize());S.add(rod);
const tip=new THREE.Mesh(new THREE.SphereGeometry(0.2,32,16),new THREE.MeshStandardMaterial({{color:0x9e1f12,roughness:0.5}}));tip.position.copy(T);S.add(tip);
S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.55));const dl=new THREE.DirectionalLight(0xffffff,1.1);dl.position.set(-6,10,14);S.add(dl);
const f=new THREE.DirectionalLight(0xffffff,0.35);f.position.set(8,2,10);S.add(f);
const hh=8.75,hw=hh*{W/Hh};const cam=new THREE.OrthographicCamera(-hw,hw,hh,-hh,0.1,100);cam.position.set(-0.3,0.1,40);cam.lookAt(-0.3,0.1,0);
R.render(S,cam);document.title='done';</script></body></html>'''
open('body.html','w').write(html)
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',args=['--use-angle=metal','--enable-webgl','--ignore-gpu-blocklist'])
    pg=b.new_page(viewport={'width':W,'height':Hh},device_scale_factor=2); pg.goto('file://'+__import__('os').path.abspath('body.html')); pg.wait_for_function("document.title=='done'")
    pg.locator('#c').screenshot(path='body.png',omit_background=True); b.close()
