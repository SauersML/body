import os
import asyncio, os
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"), args=["--enable-unsafe-swiftshader"])
        pg=await b.new_page(viewport={"width":2400,"height":800})
        await pg.goto("file://"+os.path.abspath("fig2.html")); await pg.wait_for_timeout(800)
        await pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/random_steering_body.png"), full_page=True)
        await b.close()
asyncio.run(main())
