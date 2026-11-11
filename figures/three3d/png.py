import os
exec(open('drive.py').read())
W,H=2600,2000
lay={"v1":[0.0,0.15,1.0,0.52],"v2":[0.30,0.64,0.70,0.36],"dist1":66,"dist2":66,"el":0.72}
cfg={"fs":52,"hfs":66,"hx":70,"hy":50,"header":"Qwen3 links each forearm spot to a spot on the same side,<br>equally far from the elbow or wrist",
 "key":(70,200),"kdx":[0,400,820,1100],
 "labels":[(1,"Qwen3-32B",70,330,64),(2,"Words of the<br>touch sentences",70,1530,60)],"ew":[(1,52,40),(2,46,30)]}
A1=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
with sync_playwright() as p:
    br,pg=open_page(p,W,H,lay); pg.evaluate(f"frame({A1},{A1})"); overlay(pg,W,H,cfg)
    pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/forearm_3d.png")); br.close()
print("done")
