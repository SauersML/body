import os
import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"), headless=True); pg = b.new_page(viewport={"width": 2600, "height": 1000})
    pg.goto("file://" + sys.argv[1]); pg.wait_for_timeout(400); pg.screenshot(path=sys.argv[2], full_page=True); b.close()
