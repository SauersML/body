"""Back view of the MakeHuman CC0 body (fig/human/base.obj) with two membrane wings attached at the shoulder blades.
wing_field(fn) -> per-vertex colour from fn(u, v): u root->tip (0..1), v front edge->back edge (0..1)."""
import os
import sys, json, numpy as np
sys.path.insert(0, os.path.expandvars('$BODY_REPO/figures/figs6'))
import mesh as M   # body mesh P, tri
from playwright.sync_api import sync_playwright
C = os.path.expandvars('$BODY_WORK')
EXE = os.environ.get("CHROME_BIN")
NU, NV = 90, 40
def lead(u, s):   # leading (front) edge, body coords, s=+1 person's left (x>0)
    x = 0.75 + 7.6 * u; y = 4.55 + 4.6 * np.sin(np.pi * 0.55 * u) - 0.9 * u ** 3; z = -0.95 - 1.2 * u
    return np.stack([s * x, y, z], -1)
def trail(u, s):  # trailing (back) edge
    x = 0.75 + 7.6 * u ** 1.15; y = 4.55 - 0.7 - 4.2 * np.sin(np.pi * u) ** 0.9 + 3.4 * u ** 2 + 0.9 * u; z = -0.95 - 1.2 * u
    return np.stack([s * x, y, z], -1)
def wing(s):
    u = np.linspace(0, 1, NU); v = np.linspace(0, 1, NV); U, Vv = np.meshgrid(u, v, indexing='ij')
    Lp, Tp = lead(U, s), trail(U, s); P = Lp + Vv[..., None] * (Tp - Lp)
    P[..., 2] += -0.25 * np.sin(np.pi * Vv) * np.sin(np.pi * np.clip(U * 1.2, 0, 1))   # slight camber
    tri = []
    for i in range(NU - 1):
        for j in range(NV - 1):
            a = i * NV + j; b = a + NV; tri += [[a, b, a + 1], [b, b + 1, a + 1]]
    return P.reshape(-1, 3), np.array(tri), U.ravel(), Vv.ravel()
def render(W, H, svg, canvas, scenes, out, Yc, s):
    """scenes: list of (x_offset, wing_colour_fn(side, U, V) -> rgb (n,3)); body light grey; back view."""
    cx0, cy0, cw, ch = canvas; PP, II, CC = [], [], []
    def add(P, tri, rgb, ox):
        Q = P.copy(); Q[:, 0] = -Q[:, 0]; Q[:, 2] = -Q[:, 2]; Q[:, 0] += ox   # yaw 180: back faces camera
        n0 = len(PP) // 3; PP.extend(Q.astype(np.float32).round(4).ravel().tolist()); II.extend((tri + n0).ravel().tolist())
        CC.extend(np.asarray(rgb).round(3).ravel().tolist())
    for ox, fn in scenes:
        add(M.P, M.tri, np.tile([0.55, 0.53, 0.51], (len(M.P), 1)), ox)
        for sd in (1, -1):
            P, tri, U, Vv = wing(sd); rgb = fn(sd, U, Vv)
            add(P, tri, rgb, ox); add(P, tri[:, ::-1], rgb, ox)
    three = open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read(); mesh = dict(p=PP, i=II, c=CC)
    html = f'''<html><body style="margin:0;background:#fff;width:{W}px;height:{H}px;position:relative;overflow:hidden">
<canvas id="c" style="position:absolute;left:{cx0}px;top:{cy0}px;width:{cw}px;height:{ch}px"></canvas>
{svg}
<script>{three}</script><script>
const M={json.dumps(mesh)};
const R=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true,alpha:true,preserveDrawingBuffer:true}});
R.setPixelRatio(2); R.setSize({cw},{ch},false); R.setClearColor(0xffffff,0); R.outputEncoding=THREE.sRGBEncoding;
const S=new THREE.Scene(); const hw={cw/2/s},hh={ch/2/s};
const cam=new THREE.OrthographicCamera(-hw,hw,{Yc}+hh,{Yc}-hh,0.1,100); cam.position.set(0,{Yc},40); cam.lookAt(0,{Yc},0);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.Float32BufferAttribute(M.p,3)); geo.setIndex(M.i);
geo.setAttribute('color',new THREE.Float32BufferAttribute(M.c,3)); geo.computeVertexNormals();
const mat=new THREE.MeshStandardMaterial({{vertexColors:true,roughness:0.6,metalness:0.0,side:THREE.FrontSide}});
S.add(new THREE.Mesh(geo,mat));
S.add(new THREE.HemisphereLight(0xffffff,0x8a7f76,0.6));
const k=new THREE.DirectionalLight(0xffffff,0.8); k.position.set(-6,10,14); S.add(k);
const f=new THREE.DirectionalLight(0xffffff,0.3); f.position.set(8,2,10); S.add(f);
R.render(S,cam); document.title='done';
</script></body></html>'''
    open(out + '.html', 'w').write(html)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=EXE, headless=True, args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        pg.goto("file://" + out + '.html'); pg.wait_for_function("document.title=='done'", timeout=180000); pg.wait_for_timeout(300)
        pg.screenshot(path=out); b.close()
if __name__ == "__main__":
    def fn(sd, U, V): return np.stack([0.2 + 0.7 * U, 0.4 + 0 * U, 0.9 - 0.7 * U + 0.2 * V], 1).clip(0, 1)
    render(1600, 1000, "", (0, 0, 1600, 1000), [(-9.5, fn), (9.5, fn)], "/tmp/wingtest.png", 3.0, 45)
