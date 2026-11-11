import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(channel="chromium",headless=True); pg=b.new_page(viewport={"width":2400,"height":2000})
    pg.goto("file://"+sys.argv[1]); pg.wait_for_timeout(500); pg.screenshot(path=sys.argv[2]); b.close()
