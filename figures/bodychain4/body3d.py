import os
import sys, json, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from playwright.sync_api import sync_playwright
H=os.path.expandvars('$BODY_REPO/mesh/')
P=np.load('meshP.npy'); F=np.load('meshF.npy'); A=sp.load_npz('meshA.npz'); L=json.load(open('landmarks.json'))
th=float(sys.argv[1]); out=sys.argv[2]; W,Hh=int(sys.argv[3]),int(sys.argv[4]); yc,hh=float(sys.argv[5]),float(sys.argv[6])
tri=np.hstack([F[:,[0,1,2]],F[:,[0,2,3]]]).ravel()
M=dict(p=P.astype(np.float32).round(4).ravel().tolist(),i=tri.tolist())
three=open(H+'three.min.js').read(); hw=hh*W/Hh
html=f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2);R.setSize({W},{Hh},false);R.setClearColor(0xffffff,1);R.outputEncoding=THREE.sRGBEncoding;
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{color:0xc9ced6,roughness:0.75}});
const S=new THREE.Scene();const m=new THREE.Mesh(geo,mat);m.rotation.y={th};S.add(m);
S.add(new THREE.HemisphereLight(0xffffff,0x8a8f99,0.75));const d=new THREE.DirectionalLight(0xffffff,0.8);d.position.set(-6,10,14);S.add(d);
const cam=new THREE.OrthographicCamera(-{hw},{hw},{hh},-{hh},0.1,100);cam.position.set(0,{yc},40);cam.lookAt(0,{yc},0);
R.render(S,cam);document.title='done';</script></body></html>'''
open('/tmp/b3.html','w').write(html)
with sync_playwright() as pw:
    b=pw.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
    pg=b.new_page(viewport={"width":W,"height":Hh},device_scale_factor=2)
    pg.goto("file:///tmp/b3.html",wait_until='commit'); pg.wait_for_function("document.title=='done'",timeout=120000); pg.wait_for_timeout(200)
    pg.screenshot(path=out); b.close()
c,s=np.cos(th),np.sin(th)
def proj(p): xp=p[...,0]*c+p[...,2]*s; zp=-p[...,0]*s+p[...,2]*c; return np.stack([(xp+hw)/(2*hw)*W,(hh-(p[...,1]-yc))/(2*hh)*Hh,zp],-1)
# skin path thumb -> shoulder
d,pred=dijkstra(A,indices=L['shoulder'],return_predecessors=True); path=[L['thumb']]
while path[-1]!=L['shoulder']: path.append(int(pred[path[-1]]))
# offset path slightly outward along normal so it sits on the surface
json.dump(dict(land={k:proj(P[v]).tolist() for k,v in L.items()},path=proj(P[path]).tolist(),th=th,W=W,H=Hh),open(out.replace('.png','.json'),'w'))
