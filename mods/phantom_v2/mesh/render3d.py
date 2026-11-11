import os, json, threading, http.server, functools, socketserver, numpy as np
from playwright.sync_api import sync_playwright
HERE=os.path.dirname(os.path.abspath(__file__))
EXE=os.environ.get("CHROME_BIN")
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
def write_mesh():
    M=np.load(os.path.expandvars('$BODY_REPO/figures/zoom3/mesh.npz'))
    M['v'].astype(np.float32).tofile(f'{HERE}/pos.bin'); M['f'].astype(np.uint32).ravel().tofile(f'{HERE}/idx.bin')
def write_col(name, rgb): np.asarray(rgb,np.float32).ravel().tofile(f'{HERE}/{name}')
def render(W,H,cfg,out,pts=()):
    srv=socketserver.TCPServer(("127.0.0.1",0),functools.partial(Q,directory=HERE)); port=srv.server_address[1]
    threading.Thread(target=srv.serve_forever,daemon=True).start()
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path=EXE,args=["--enable-unsafe-swiftshader"])
        pg=b.new_page(viewport={"width":W,"height":H})
        pg.on("pageerror", lambda e: print("pageerror:",e)); pg.on("console", lambda m: print("console:",m.text))
        pg.goto(f"http://127.0.0.1:{port}/scene.html?w={W}&h={H}&cfg="+json.dumps(cfg)); pg.wait_for_function("window.READY===true",timeout=120000)
        pr=[pg.evaluate(f"proj({x},{y},{z})") for x,y,z in pts]
        pg.screenshot(path=out); b.close()
    srv.shutdown(); return pr
