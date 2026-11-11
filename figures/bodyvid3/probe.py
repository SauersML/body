import os,sys,json,threading,http.server,functools,socketserver,numpy as np
from playwright.sync_api import sync_playwright
from drive import EXE,WEB,Q
srv=socketserver.ThreadingTCPServer(("127.0.0.1",0),functools.partial(Q,directory=WEB)); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader"])
    pg=br.new_page(); pg.goto(f"http://127.0.0.1:{port}/page.html?v=c2"); pg.wait_for_function("window.READY===true",timeout=300000)
    X=[]
    for k0 in range(0,2431,100):
        X+= [np.array(r).reshape(407,3) for r in pg.evaluate(f"(()=>{{const ids=[...Array(407).keys()];const o=[];for(let k={k0};k<Math.min(2431,{k0}+100);k++)o.push(window.M.probe(k/60,ids).flat());return o;}})()")]
    br.close()
X=np.array(X); np.save('/tmp/bv/probe.npy',X); print(X.shape)
v=np.linalg.norm(np.diff(X,axis=0),axis=2)  # speed per frame
a=np.linalg.norm(np.diff(X,2,axis=0),axis=2)
print('max speed cm/frame',v.max().round(3),'max accel',a.max().round(4))
amax=a.max(1); i=np.argsort(-amax)[:8]; print('worst accel frames',[(k+1)/60 for k in i],amax[i].round(4))
