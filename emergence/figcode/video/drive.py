# Render frames of web/<v>.js headless (Chrome for Testing) and pipe to ffmpeg.
import os, sys, threading, http.server, functools, socketserver, subprocess, time
os.environ["PLAYWRIGHT_BROWSERS_PATH"]="$BODY_WORK"
from playwright.sync_api import sync_playwright
EXE="$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
WEB=os.path.join(os.path.dirname(os.path.abspath(__file__)),'web')
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
def run(v,mode,args,fps=int(os.environ.get("FPS","60")),gpu=True):
    srv=socketserver.ThreadingTCPServer(("127.0.0.1",0),functools.partial(Q,directory=WEB)); port=srv.server_address[1]
    threading.Thread(target=srv.serve_forever,daemon=True).start()
    a=["--enable-unsafe-swiftshader"]+(["--use-angle=metal","--enable-gpu","--ignore-gpu-blocklist"] if gpu else [])
    with sync_playwright() as p:
        br=p.chromium.launch(executable_path=EXE,headless=True,args=a)
        pg=br.new_context(viewport={"width":1920,"height":1080},device_scale_factor=1).new_page()
        pg.on("console",lambda m: print("console:",m.text) if m.type in("error","warning","log") else None)
        pg.on("pageerror",lambda e: print("pageerror:",e))
        pg.goto(f"http://127.0.0.1:{port}/page.html?v={v}"); pg.wait_for_function("window.READY===true",timeout=300000)
        T=pg.evaluate("window.T")
        if mode=='still':
            for s in args:
                t0=time.time(); pg.evaluate(f"window.F({s})"); pg.screenshot(path=f"{v}_{float(s):04.1f}.png"); print(s,'%.2fs'%(time.time()-t0))
        else:
            out=args[0]; a0,a1=(int(args[1]),int(args[2])) if len(args)>2 else (0,int(T*fps))
            ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','image2pipe','-framerate',str(fps),'-i','-','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
            t0=time.time()
            for k in range(a0,a1):
                pg.evaluate(f"window.F({k/fps})"); ff.stdin.write(pg.screenshot(type='png'))
                if k%100==0: print(v,k,a1,'%.1fs'%(time.time()-t0),flush=True)
            ff.stdin.close(); ff.wait()
        br.close()
    srv.shutdown()
if __name__=="__main__": run(sys.argv[1],sys.argv[2],sys.argv[3:])
