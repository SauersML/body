import os, json, threading, http.server, functools, socketserver
from playwright.sync_api import sync_playwright
HERE=os.path.dirname(os.path.abspath(__file__))
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
srv=socketserver.TCPServer(("127.0.0.1",0),functools.partial(Q,directory=HERE)); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
EXE=os.environ.get("CHROME_BIN")
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader","--use-angle=swiftshader"])
    pg=br.new_page(viewport={"width":1920,"height":1080})
    pg.goto(f"http://127.0.0.1:{port}/scene2.html"); pg.wait_for_function("window.READY===true",timeout=600000)
    d=pg.evaluate("window.getTiming()"); br.close()
json.dump(d,open(HERE+'/timing.json','w'),indent=1); print(json.dumps(d)[:1500])
