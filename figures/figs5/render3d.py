import os
import json, numpy as np
from playwright.sync_api import sync_playwright
C=os.path.expandvars('$BODY_WORK')
EXE=os.environ.get("CHROME_BIN")
def page(W,H,svg,canvas,bodies,P,tri,out,Yc,s,yaw=0.0):
    """canvas=(x,y,w,h); bodies=[(world_x_offset, rgb per vertex (n,3) in 0..1)]; s=px per world unit"""
    cx0,cy0,cw,ch=canvas; PP=[];II=[];CC=[]
    for j,(ox,rgb) in enumerate(bodies):
        Q=P.copy(); c,sn=np.cos(yaw),np.sin(yaw); Q=np.c_[Q[:,0]*c+Q[:,2]*sn,Q[:,1],-Q[:,0]*sn+Q[:,2]*c]; Q[:,0]+=ox
        PP+=Q.astype(np.float32).round(4).ravel().tolist(); II+=(tri+j*len(P)).ravel().tolist(); CC+=np.asarray(rgb).round(3).ravel().tolist()
    mesh=dict(p=PP,i=II,c=CC); three=open(os.path.expandvars('$BODY_REPO/mesh/three.min.js')).read()
    html=f'''<html><body style="margin:0;background:#fff;width:{W}px;height:{H}px;position:relative;overflow:hidden">
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
const mat=new THREE.MeshStandardMaterial({{vertexColors:true,roughness:0.62,metalness:0.0}});
S.add(new THREE.Mesh(geo,mat));
S.add(new THREE.HemisphereLight(0xffffff,0x8a7f76,0.55));
const k=new THREE.DirectionalLight(0xffffff,0.85); k.position.set(-6,10,14); S.add(k);
const f=new THREE.DirectionalLight(0xffffff,0.3); f.position.set(8,2,10); S.add(f);
R.render(S,cam); document.title='done';
</script></body></html>'''
    open(out+'.html','w').write(html)
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader","--use-gl=angle","--use-angle=swiftshader"])
        pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=1)
        pg.on('console',lambda m:print('JS',m.text[:200]) if 'deprecated' not in m.text else None)
        pg.goto("file://"+out+'.html'); pg.wait_for_function("document.title=='done'",timeout=120000); pg.wait_for_timeout(300)
        pg.screenshot(path=out); b.close()
