import os,sys,threading,functools,socketserver,numpy as np
from playwright.sync_api import sync_playwright
from drive import EXE,WEB,Q
srv=socketserver.ThreadingTCPServer(("127.0.0.1",0),functools.partial(Q,directory=WEB)); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
N=int(51*60)+1
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader","--use-angle=metal","--enable-gpu","--ignore-gpu-blocklist"])
    pg=br.new_page(); pg.goto(f"http://127.0.0.1:{port}/page.html?v=c4"); pg.wait_for_function("window.READY===true",timeout=300000)
    X=[]
    for k0 in range(0,N,200):
        print(k0,flush=True); X+=pg.evaluate(f"(()=>{{const ids=[...Array(407).keys()];const o=[];for(let k={k0};k<Math.min({N},{k0}+200);k++)o.push(window.M.probeS(k/60,ids));return o;}})()")
    br.close()
X=np.array(X); np.save('probe4.npy',X); print(X.shape)
