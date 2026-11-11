import os
# render neutral.obj with per-vertex colors to PNG (front, or front+back), Chrome for Testing headless
import json, numpy as np, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.expandvars('$BODY_REPO/figures/selfother2'))
from anchors import P, TRI
CHROME = os.environ.get("CHROME_BIN")
THREE = open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read()
def render(cols, out, views=(0,), W=700, Hh=1400, yc=0.15, hh=8.75):
    """cols: (nverts,3) floats 0..1 (sRGB). views: rotation angles (0 front, pi back)."""
    M = dict(p=P.astype(np.float32).round(4).ravel().tolist(), i=TRI.ravel().tolist(), c=np.clip(cols, 0, 1).astype(np.float32).round(3).ravel().tolist())
    Wt = W * len(views); hw = hh * W / Hh
    html = f'''<html><body style="margin:0;background:#fff"><canvas id="c" style="width:{Wt}px;height:{Hh}px"></canvas><script>{THREE}</script><script>
const M={json.dumps(M)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true,alpha:true}});
R.setPixelRatio(2);R.setSize({Wt},{Hh},false);R.setClearColor(0xffffff,0);R.outputEncoding=THREE.sRGBEncoding;R.autoClear=false;
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3));geo.setIndex(M.i);
const c=new Float32Array(M.c.length);for(let k=0;k<c.length;k++){{const v=M.c[k];c[k]=v<=0.04045?v/12.92:Math.pow((v+0.055)/1.055,2.4);}}
geo.setAttribute('color',new THREE.BufferAttribute(c,3));geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{vertexColors:true,roughness:0.75}});R.clear();
const views={json.dumps(list(views))};
views.forEach((ang,k)=>{{
 const S=new THREE.Scene();const m=new THREE.Mesh(geo,mat);m.rotation.y=ang;S.add(m);
 S.add(new THREE.HemisphereLight(0xffffff,0x8a8580,0.75));const d=new THREE.DirectionalLight(0xffffff,0.75);d.position.set(-6,10,14);S.add(d);
 const f=new THREE.DirectionalLight(0xffffff,0.3);f.position.set(8,2,10);S.add(f);
 const cam=new THREE.OrthographicCamera(-{hw},{hw},{hh},-{hh},0.1,100);cam.position.set(0,{yc},40);cam.lookAt(0,{yc},0);
 R.setViewport(k*{W},0,{W},{Hh});R.render(S,cam);}});
document.title='done';</script></body></html>'''
    open('/tmp/so2body.html', 'w').write(html)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME, headless=True, args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
        pg = b.new_page(viewport={"width": Wt, "height": Hh}); pg.on('console', lambda m: print('JS', m.text[:200]))
        pg.goto("file:///tmp/so2body.html", wait_until='commit'); pg.wait_for_function("document.title=='done'", timeout=120000); pg.wait_for_timeout(200)
        pg.locator('canvas').screenshot(path=out, omit_background=True); b.close()
