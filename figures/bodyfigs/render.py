import os
import sys
from playwright.sync_api import sync_playwright
def render(svg, out, W, H):
    html=f'<html><body style="margin:0;background:#fff">{svg}</body></html>'
    open(out+'.html','w').write(html)
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"),headless=True,args=["--enable-unsafe-swiftshader"]); pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=1)
        pg.goto("file://"+out+'.html'); pg.wait_for_timeout(300); pg.screenshot(path=out); b.close()
