import os, json, threading, http.server, functools, socketserver
from playwright.sync_api import sync_playwright
from PIL import Image
EXE=os.environ.get("CHROME_BIN")
HERE=os.path.expandvars("$BODY_REPO/figures/nb4")
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
def shoot(cfg,out):
    json.dump(cfg,open(os.path.join(HERE,"cfg.json"),"w"))
    srv=socketserver.TCPServer(("127.0.0.1",0),functools.partial(Q,directory=HERE)); port=srv.server_address[1]
    threading.Thread(target=srv.serve_forever,daemon=True).start()
    W,H,d=cfg["W"],cfg["H"],cfg["dpr"]
    with sync_playwright() as p:
        br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader","--use-angle=metal","--enable-gpu","--ignore-gpu-blocklist"])
        pg=br.new_context(viewport={"width":W,"height":H},device_scale_factor=d).new_page()
        pg.on("console",lambda m: print("console:",m.text) if m.type in("error","warning") else None)
        pg.on("pageerror",lambda e: print("pageerror:",e))
        pg.goto(f"http://127.0.0.1:{port}/scene.html"); pg.wait_for_function("window.READY===true",timeout=200000)
        pg.screenshot(path="/tmp/nb4_big.png"); br.close()
    srv.shutdown()
    Image.open("/tmp/nb4_big.png").convert("RGB").resize((W,H),Image.LANCZOS).save(out)
