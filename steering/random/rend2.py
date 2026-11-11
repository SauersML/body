import os
from playwright.sync_api import sync_playwright
import re,sys
h=open('fig2.html').read(); H=int(re.search(r'height="(\d+)"',h).group(1))
with sync_playwright() as p:
    b=p.chromium.launch(channel="chromium",headless=True); pg=b.new_page(viewport={"width":2400,"height":H})
    pg.goto(os.path.expandvars("file://$BODY_REPO/steering/random/fig2.html")); pg.wait_for_timeout(600); pg.screenshot(path="fig2.png"); b.close()
