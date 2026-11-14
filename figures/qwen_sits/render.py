import sys, json, numpy as np
from playwright.sync_api import sync_playwright
H='$BODY_WORK/fig/human/'
d=np.load('posed.npy',allow_pickle=True).item(); P,F=d['P'],d['F']; C=json.load(open('chair.json'))
tri=np.hstack([F[:,[0,1,2]],F[:,[0,2,3]]]).ravel()
ang=float(sys.argv[1]) if len(sys.argv)>1 else 55; out=sys.argv[2] if len(sys.argv)>2 else 'body.png'
W,Hh=1400,1600
M=dict(p=P.astype(np.float32).round(5).ravel().tolist(),i=tri.tolist())
three=open(H+'three.min.js').read()
html=f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)};const C={json.dumps(C)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true,alpha:true}});
R.setPixelRatio(2);R.setSize({W},{Hh},false);R.setClearColor(0xffffff,0);R.outputEncoding=THREE.sRGBEncoding;
R.shadowMap.enabled=true;R.shadowMap.type=THREE.PCFSoftShadowMap;
const S=new THREE.Scene();
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
const body=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{color:0x9c938b,roughness:0.6}}));body.castShadow=true;body.receiveShadow=true;S.add(body);
const wood=new THREE.MeshStandardMaterial({{color:0x9a7452,roughness:0.65}});
function box(w,h,d,x,y,z){{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),wood);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;S.add(m);return m;}}
const st=C.seat, fl=C.floor, bz=C.back-0.05, hw=C.hipw+0.5, fz=bz+5.0, th=0.35;
box(2*hw,th,fz-bz,0,st-th/2,(bz+fz)/2);
for(const sx of [-1,1]) for(const z of [bz+0.25,fz-0.25]) box(0.32,st-th-fl,0.32,sx*(hw-0.25),(st-th+fl)/2,z);
for(const sx of [-1,1]) box(0.32,6.2,0.32,sx*(hw-0.25),st+3.1,bz-0.05);
box(2*hw,1.6,0.25,0,st+4.6,bz-0.05);box(2*hw,0.25,0.22,0,st+1.6,bz-0.05);
const pl=new THREE.Mesh(new THREE.PlaneGeometry(80,80),new THREE.ShadowMaterial({{opacity:0.16}}));pl.rotation.x=-Math.PI/2;pl.position.y=fl;pl.receiveShadow=true;S.add(pl);
S.add(new THREE.HemisphereLight(0xffffff,0x6a625c,0.55));
const d=new THREE.DirectionalLight(0xffffff,1.1);d.position.set(1.5,22,4);d.castShadow=true;d.shadow.mapSize.set(4096,4096);
Object.assign(d.shadow.camera,{{left:-12,right:12,top:12,bottom:-12,near:1,far:60}});d.shadow.radius=6;d.shadow.bias=-0.0005;S.add(d);
const f=new THREE.DirectionalLight(0xffffff,0.35);f.position.set(-10,4,8);S.add(f);
const a={ang}*Math.PI/180, r=34;
const cam=new THREE.PerspectiveCamera(30,{W/Hh},1,200);cam.position.set(r*Math.sin(a),5,r*Math.cos(a)+1.5);cam.lookAt(0,0.6,1.5);
R.render(S,cam);document.title='done';</script></body></html>'''
open('/tmp/sit.html','w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path="$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
    pg=b.new_page(viewport={"width":W,"height":Hh},device_scale_factor=2); pg.on('console',lambda m:print('JS',m.text[:200]))
    pg.goto("file:///tmp/sit.html",wait_until='commit'); pg.wait_for_function("document.title=='done'",timeout=180000); pg.wait_for_timeout(200)
    pg.screenshot(path=out,omit_background=True); b.close()
