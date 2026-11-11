import os
import sys, json, numpy as np
from playwright.sync_api import sync_playwright
H=os.path.expandvars('$BODY_REPO/mesh/')
obj=sys.argv[1]; out=sys.argv[2]; zoom=sys.argv[3] if len(sys.argv)>3 else 'full'
V=[];F=[];g=None
for l in open(obj):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V);F=np.array(F); used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
tri=np.hstack([F[:,[0,1,2]],F[:,[0,2,3]]]).ravel()
W,Hh=(3000,1800) if zoom=="full" else (3000,1300)
yc,hh=(0.15,8.8) if zoom=='full' else map(float,zoom.split(','))
cxs=0 if zoom=='full' else -0.65
M=dict(p=P.astype(np.float32).round(6).ravel().tolist(),i=tri.tolist())
three=open(H+'three.min.js').read()
html=f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2);R.setSize({W},{Hh},false);R.setClearColor(0xffffff,1);R.outputEncoding=THREE.sRGBEncoding;R.autoClear=false;
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{color:0x8c8279,roughness:0.6}});
const hh={hh},hw=hh*{W/3/Hh};R.clear();
for(const [k,ang] of [[0,0],[1,Math.PI/2],[2,Math.PI/4]]){{
 const S=new THREE.Scene();const m=new THREE.Mesh(geo,mat);m.rotation.y=-ang;S.add(m);
 S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.5));const d=new THREE.DirectionalLight(0xffffff,1.15);d.position.set(-6,10,14);S.add(d);
 const f=new THREE.DirectionalLight(0xffffff,0.35);f.position.set(8,2,10);S.add(f);
 const cam=new THREE.OrthographicCamera(-hw,hw,hh,-hh,0.1,100);const cx={cxs}*Math.sin(ang);cam.position.set(cx,{yc},40);cam.lookAt(cx,{yc},0);
 R.setViewport(k*{W//3},0,{W//3},{Hh});R.render(S,cam);}}
document.title='done';</script></body></html>'''
open('/tmp/prev.html','w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
    pg=b.new_page(viewport={"width":W,"height":Hh}); pg.on('console',lambda m:print('JS',m.text[:200]))
    pg.goto("file:///tmp/prev.html",wait_until='commit'); pg.wait_for_function("document.title=='done'",timeout=120000); pg.wait_for_timeout(200)
    pg.screenshot(path=out); b.close()
