import json, numpy as np, sys
from playwright.sync_api import sync_playwright
from mesh import load
H='$BODY_WORK/fig/human/'
P,T=load()
def render(panels, out, W=2400, Hh=1600, bg='#0b1020', rough=0.62, extra_js='', PP=None, TT=None):
    PP=P if PP is None else PP; TT=T if TT is None else TT
    """panels: list of dict(col=(N,3) srgb, ang=radians, vp=(x,y,w,h) in px from bottom-left, hh=half height world units, yc)"""
    three=open(H+'three.min.js').read()
    js=[]
    for k,pn in enumerate(panels):
        col=np.clip(pn['col'],0,1)**2.2
        js.append(dict(c=col.astype(np.float32).round(4).ravel().tolist(),a=pn.get('ang',0),vp=pn['vp'],hh=pn.get('hh',8.9),yc=pn.get('yc',0.2),cx=pn.get('cx',0)))
    M=dict(p=PP.astype(np.float32).round(5).ravel().tolist(),i=TT.ravel().tolist())
    html=f'''<html><body style="margin:0;background:{bg}"><canvas id="c" style="width:{W}px;height:{Hh}px"></canvas><script>{three}</script><script>
const M={json.dumps(M)}; const PN={json.dumps(js)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true}});
R.setPixelRatio(1);R.setSize({W},{Hh},false);R.setClearColor(new THREE.Color('{bg}'),1);R.outputEncoding=THREE.sRGBEncoding;R.autoClear=false;R.clear();
for(const pn of PN){{
 const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);geo.computeVertexNormals();
 geo.setAttribute('color',new THREE.Float32BufferAttribute(pn.c,3));
 const mat=new THREE.MeshStandardMaterial({{vertexColors:true,roughness:{rough},metalness:0.0}});
 const S=new THREE.Scene();const m=new THREE.Mesh(geo,mat);m.rotation.y=-pn.a;S.add(m);
 S.add(new THREE.HemisphereLight(0xffffff,0x404858,0.75));
 const d=new THREE.DirectionalLight(0xffffff,0.95);d.position.set(-5,9,14);S.add(d);
 const f=new THREE.DirectionalLight(0xffffff,0.35);f.position.set(9,3,10);S.add(f);
 const r=new THREE.DirectionalLight(0xbfd4ff,0.45);r.position.set(0,6,-14);S.add(r);
 const hw=pn.hh*pn.vp[2]/pn.vp[3];
 const cam=new THREE.OrthographicCamera(-hw,hw,pn.hh,-pn.hh,0.1,100);cam.position.set(pn.cx,pn.yc,40);cam.lookAt(pn.cx,pn.yc,0);
 R.setViewport(pn.vp[0],pn.vp[1],pn.vp[2],pn.vp[3]);R.render(S,cam);}}
{extra_js}
document.title='done';</script></body></html>'''
    open('/tmp/bodyart.html','w').write(html)
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path="$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
        pg=b.new_page(viewport={"width":W,"height":Hh})
        pg.goto("file:///tmp/bodyart.html",wait_until='commit'); pg.wait_for_function("document.title=='done'",timeout=180000); pg.wait_for_timeout(200)
        pg.screenshot(path=out); b.close()
