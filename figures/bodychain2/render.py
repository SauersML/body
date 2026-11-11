import os
import sys, os
from playwright.sync_api import sync_playwright
W, H = 2400, 1940
exe = os.environ.get("CHROME_BIN")
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe, args=["--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.goto("file://" + os.path.abspath("fig.html")); pg.wait_for_timeout(300)
    pg.screenshot(path=sys.argv[1], clip={"x": 0, "y": 0, "width": W, "height": H}); b.close()
