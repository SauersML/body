import os
import asyncio
from playwright.async_api import async_playwright
EXE=os.environ.get("CHROME_BIN")
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path=EXE,args=["--enable-unsafe-swiftshader"])
        pg=await b.new_page(viewport={"width":2400,"height":1000},device_scale_factor=1)
        await pg.goto(os.path.expandvars("file://$BODY_REPO/figures/selfother/fig.html"))
        h=await pg.evaluate("document.body.scrollHeight"); await pg.set_viewport_size({"width":2400,"height":h})
        await pg.screenshot(path=os.path.expandvars("$BODY_REPO/figures/out/selfother_main.png"),full_page=True); print(h)
        await b.close()
asyncio.run(main())
