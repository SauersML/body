import os, sys, json, threading, http.server, functools, socketserver, math
from playwright.sync_api import sync_playwright
HERE=os.path.dirname(os.path.abspath(__file__))
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
srv=socketserver.TCPServer(("127.0.0.1",0),functools.partial(Q,directory=HERE)); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
SC=['#2b8a3e','#1971c2','#e8590c','#ae3ec9']; SN=['palm side','thumb side','back','little-finger side']
def overlay(pg, W, H, cfg):
    hs=cfg['fs']; html=[]
    html.append(f'<div class="t h" style="left:{cfg["hx"]}px;top:{cfg["hy"]}px;font-size:{cfg["hfs"]}px;line-height:1.12">{cfg["header"]}</div>')
    # side key
    kx,ky=cfg['key']
    for s in range(4):
        x=kx+cfg['kdx'][s]; y=ky
        html.append(f'<div class="key" style="left:{x}px;top:{y}px;font-size:{hs}px;color:{SC[s]};font-weight:700"><span class="dot" style="width:{hs*0.8}px;height:{hs*0.8}px;background:{SC[s]}"></span>{SN[s]}</div>')
    for (which,txt,fx,fy,size) in cfg['labels']:
        html.append(f'<div class="t h" style="left:{fx}px;top:{fy}px;font-size:{size}px">{txt}</div>')
    # elbow/wrist labels projected
    X0,X1=pg.evaluate("[ARM.X0,ARM.X1]")
    for which,size,dy in cfg['ew']:
        px,py=pg.evaluate(f"proj({which},-3.9,0,0)")
        html.append(f'<div class="lab" style="left:{px-12}px;top:{py}px;font-size:{size}px;font-weight:700;transform:translate(-100%,-50%)">elbow</div>')
        px,py=pg.evaluate(f"proj({which},{X1+1.2},-2.2,2.4)")
        html.append(f'<div class="lab" style="left:{px}px;top:{py+dy}px;font-size:{size}px;font-weight:700">wrist</div>')
    pg.evaluate("h=>document.getElementById('ov').innerHTML=h", "".join(html))
def open_page(p, W, H, lay):
    br=p.chromium.launch(channel="chromium",headless=True,args=["--use-angle=metal","--enable-gpu","--ignore-gpu-blocklist"])
    pg=br.new_context(viewport={"width":W,"height":H},device_scale_factor=1).new_page()
    pg.on("console", lambda m: print("console:",m.text) if m.type in("error","warning") else None)
    pg.on("pageerror", lambda e: print("pageerror:",e))
    pg.goto(f"http://127.0.0.1:{port}/scene.html?w={W}&h={H}&lay="+json.dumps(lay)); pg.wait_for_function("window.READY===true",timeout=20000)
    return br,pg
