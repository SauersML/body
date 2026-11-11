import os
exec(open('vid.py').read().split("lim=")[0])
with sync_playwright() as p:
    br,pg=open_page(p,W,H,lay); pg.evaluate("frame(0,0)"); overlay(pg,W,H,cfg)
    for k,a in enumerate([0,1.2,2.4,3.6,4.8]):
        pg.evaluate(f"frame({a},{a})"); pg.screenshot(path=f"/tmp/probe{k}.png")
    br.close()
from PIL import Image
ims=[Image.open(f"/tmp/probe{k}.png").resize((960,540)) for k in range(5)]
m=Image.new("RGB",(1920,1620),"white")
for k,im in enumerate(ims[:6]): m.paste(im,((k%2)*960,(k//2)*540))
m.save(os.path.expandvars("$BODY_REPO/figures/out/fa_probe.png"))
