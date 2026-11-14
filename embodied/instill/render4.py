"""Render the v4 body mesh (three.js, Chrome for Testing headless). render(views, out, W, H)
views: list of dict(P [NV,3] vertex positions, C [NV,3] colors 0-1, ang (rotation about y, radians), x0,y0,hh (ortho window center, half height),
spheres [(xyz, r, rgb)], rods [(a, b, r, rgb)]). Each view gets an equal-width column. proj(view, xyz, W, H) -> pixel."""
import json, numpy as np, os
from playwright.sync_api import sync_playwright
HUM = "$BODY_WORK/fig/human/"
CH = "$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
_B = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4", "body4.npz"))
F = _B["F"]; TRI = np.hstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]]).ravel()


def proj(v, xyz, W, H):
    a = v.get("ang", 0.0); x, y, z = np.asarray(xyz, float).T
    xr = np.cos(a) * x - np.sin(a) * z  # rotation.y = -a applied to the mesh
    hw = v["hh"] * W / H
    return np.stack([(xr - v.get("x0", 0) + hw) / (2 * hw) * W, (v["hh"] - (y - v.get("y0", 0))) / (2 * v["hh"]) * H], -1)


def render(views, out, W, H, bg="#ffffff", rough=0.55):
    n = len(views); Wv = W // n
    js = []
    for v in views:
        js.append(dict(p=np.asarray(v["P"], np.float32).round(4).ravel().tolist(), c=np.asarray(v["C"], np.float32).round(3).ravel().tolist(),
                       ang=float(v.get("ang", 0)), x0=float(v.get("x0", 0)), y0=float(v.get("y0", 0)), hh=float(v["hh"]),
                       sph=[[*map(float, s[0]), float(s[1]), *map(float, s[2])] for s in v.get("spheres", [])],
                       rods=[[*map(float, r[0]), *map(float, r[1]), float(r[2]), *map(float, r[3])] for r in v.get("rods", [])]))
    three = open(HUM + "three.min.js").read()
    html = f'''<html><body style="margin:0;background:{bg}"><canvas id="c" style="width:{W}px;height:{H}px"></canvas><script>{three}</script><script>
const VS={json.dumps(js)}; const I={json.dumps(TRI.tolist())};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true,alpha:true}});
R.setPixelRatio(2);R.setSize({W},{H},false);R.setClearColor(0xffffff,0);R.outputEncoding=THREE.sRGBEncoding;R.autoClear=false;R.clear();
VS.forEach((v,k)=>{{
 const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(v.p,3));
 const col=v.c.map(x=>Math.pow(x,2.2));geo.setAttribute('color',new THREE.Float32BufferAttribute(col,3));geo.setIndex(I);geo.computeVertexNormals();
 const S=new THREE.Scene();const g=new THREE.Group();g.rotation.y=-v.ang;S.add(g);
 g.add(new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{vertexColors:true,roughness:{rough},metalness:0.0}})));
 v.sph.forEach(s=>{{const m=new THREE.Mesh(new THREE.SphereGeometry(s[3],32,24),new THREE.MeshStandardMaterial({{color:new THREE.Color(Math.pow(s[4],2.2),Math.pow(s[5],2.2),Math.pow(s[6],2.2)),roughness:0.35}}));m.position.set(s[0],s[1],s[2]);g.add(m);}});
 v.rods.forEach(r=>{{const a=new THREE.Vector3(r[0],r[1],r[2]),b=new THREE.Vector3(r[3],r[4],r[5]);const L=a.distanceTo(b);
   const m=new THREE.Mesh(new THREE.CylinderGeometry(r[6],r[6],L,24),new THREE.MeshStandardMaterial({{color:new THREE.Color(Math.pow(r[7],2.2),Math.pow(r[8],2.2),Math.pow(r[9],2.2)),roughness:0.5}}));
   m.position.copy(a.clone().add(b).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),b.clone().sub(a).normalize());g.add(m);}});
 S.add(new THREE.HemisphereLight(0xffffff,0x77706a,0.62));const d=new THREE.DirectionalLight(0xffffff,0.95);d.position.set(-6,10,14);S.add(d);
 const f=new THREE.DirectionalLight(0xffffff,0.3);f.position.set(8,2,10);S.add(f);
 const hh=v.hh,hw=hh*{Wv}/{H};const cam=new THREE.OrthographicCamera(-hw,hw,hh,-hh,0.1,200);cam.position.set(v.x0,v.y0,60);cam.lookAt(v.x0,v.y0,0);
 R.setViewport(k*{Wv},0,{Wv},{H});R.render(S,cam);}});
document.title='done';</script></body></html>'''
    tmp = out + ".html"; open(tmp, "w").write(html)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CH, headless=True, args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
        pg = b.new_page(viewport={"width": W, "height": H}); pg.on("console", lambda m: print("JS", m.text[:200]))
        pg.goto("file://" + tmp, wait_until="commit"); pg.wait_for_function("document.title=='done'", timeout=180000); pg.wait_for_timeout(200)
        pg.screenshot(path=out, omit_background=True); b.close()
    os.remove(tmp)
