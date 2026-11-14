import sys,json,os
from playwright.sync_api import sync_playwright
os.environ['PLAYWRIGHT_BROWSERS_PATH']='$BODY_WORK'
jobs=json.loads(sys.argv[1])
with sync_playwright() as p:
    b=p.chromium.launch(channel='chromium',args=['--enable-unsafe-swiftshader','--use-angle=swiftshader','--ignore-gpu-blocklist'])
    for out,q in jobs:
        pg=b.new_page(viewport={'width':q.get('w',1200),'height':q.get('h',1200)})
        pg.on('pageerror',lambda e:print('E:',str(e)[:300]));pg.on('requestfailed',lambda r:print('RF:',r.url));pg.on('console',lambda m:print('C:',m.text[:200]))
        from urllib.parse import urlencode
        pg.goto('http://localhost:8765/render.html?'+urlencode({k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in q.items()}))
        pg.wait_for_function('window.done===true',timeout=150000)
        print(out,json.dumps(pg.evaluate('window.info'))[:3000])
        pg.locator('canvas').screenshot(path=out,omit_background=True)
    b.close()
