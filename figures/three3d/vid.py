exec(open('drive.py').read())
import time, shutil
W,H=1920,1080
lay={"single":1,"v1":[0,0,1,1],"dist1":118,"el":0.45,"hk":0.72,"p1":[-7.4,1.4,0],"p2":[18.9,-10.5,0],"s2":0.62,"look":[0,0,0]}
cfg={"fs":34,"hfs":44,"hx":44,"hy":30,"header":"Qwen3 links each forearm spot to a spot on the same side,<br>equally far from the elbow or wrist",
 "key":(44,136),"kdx":[0,250,520,700],
 "labels":[(1,"Qwen3-32B",44,250,40),(2,"Words of the<br>touch sentences",760,800,40)],"ew":[(1,36,30)]}
FR="/tmp/fa_frames"; shutil.rmtree(FR,ignore_errors=True); os.makedirs(FR)
fps=30; T=13.0; n=int(T*fps)
def ang(t):
    if t<1.2: return 0.0
    if t>T-1.0: return 2*math.pi
    u=(t-1.2)/(T-2.2); return 2*math.pi*(0.5-0.5*math.cos(math.pi*u))
lim=int(sys.argv[1]) if len(sys.argv)>1 else n
with sync_playwright() as p:
    br,pg=open_page(p,W,H,lay); pg.evaluate("frame(0,0)")
    overlay(pg,W,H,cfg)
    # key as a vertical list on the right
    t0=time.time()
    for k in range(min(n,lim)):
        a=ang(k/fps); pg.evaluate(f"frame({a},{a})"); pg.screenshot(path=f"{FR}/{k:04d}.png")
    print("frames",time.time()-t0); br.close()
