import os, sys, json, threading, http.server, functools, socketserver
from playwright.sync_api import sync_playwright
HERE=os.path.dirname(os.path.abspath(__file__))
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
srv=socketserver.TCPServer(("127.0.0.1",0),functools.partial(Q,directory=HERE)); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
EXE=os.environ.get("CHROME_BIN")
mode=sys.argv[1]; out=sys.argv[2]; PAGE=os.environ.get("PAGE","scene.html")
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader","--use-angle=swiftshader","--ignore-gpu-blocklist"])
    pg=br.new_context(viewport={"width":1920,"height":1080}).new_page()
    pg.on("console", lambda m: print("console:",m.text) if m.type in("error","warning") else None)
    pg.on("pageerror", lambda e: print("pageerror:",e))
    pg.goto(f"http://127.0.0.1:{port}/{PAGE}"); pg.wait_for_function("window.READY===true",timeout=600000)
    if mode=='cache': json.dump(pg.evaluate('window.RC'),open(HERE+'/rcache.json','w')); print('cached')
    ev=pg.evaluate("window.EV"); json.dump(ev,open(HERE+'/events.json','w')); print({k:v for k,v in ev.items() if k!='events'})
    if mode=='stills':
        for t in [float(x) for x in sys.argv[3].split(',')]:
            pg.evaluate(f"frame({t})"); pg.screenshot(path=f"{out}/still_{t:05.1f}.png")
    else:
        fps=30; a,b=[int(x) for x in sys.argv[3].split(':')]
        os.makedirs(out,exist_ok=True)
        for k in range(a,b):
            pg.evaluate(f"frame({k/fps})"); pg.screenshot(path=f"{out}/{k:05d}.jpg",type='jpeg',quality=94)
    br.close()
