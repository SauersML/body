import os
from playwright.sync_api import sync_playwright
F = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
class SVG:
    def __init__(s): s.o = []
    def add(s, x): s.o.append(x)
    def t(s, x, y, txt, size=48, w=400, anchor="start", fill="#111", style=""):
        s.o.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}" {style}>{txt}</text>')
    def arrow(s, x1, y1, x2, y2, c="#222", w=7, head=26):
        import math
        a = math.atan2(y2 - y1, x2 - x1); bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
        px, py = -math.sin(a) * head * 0.6, math.cos(a) * head * 0.6
        s.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{c}" stroke-width="{w}"/><polygon points="{x2:.1f},{y2:.1f} {bx+px:.1f},{by+py:.1f} {bx-px:.1f},{by-py:.1f}" fill="{c}"/>')
    def render(s, W, H, path):
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="100%" height="100%" fill="#fff"/>' + "".join(s.o) + "</svg>"
        html = path.replace(".png", ".html"); open(html, "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chromium", headless=True); pg = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
            pg.goto("file://" + html); pg.screenshot(path=path); b.close()
        from PIL import Image
        im = Image.open(path); im = im.resize((W, H), Image.LANCZOS); im.save(path)
def hull(P):
    P = sorted(map(tuple, P))
    def cr(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]
