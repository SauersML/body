import os
F = "Helvetica Neue, Helvetica, Arial, sans-serif"
def T(out, x, y, s, size=48, w=400, anchor="start", fill="#111", halo=0):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}"' + (f' stroke="#fff" stroke-width="{halo}" paint-order="stroke" stroke-linejoin="round"' if halo else '') + f'>{s}</text>')
def render(out, W, H, path, name):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#fff"/>' + "".join(out) + "</svg>"
    h = f"/tmp/{name}.html"; open(h, "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
        pg.goto("file://" + h); pg.screenshot(path=f"/tmp/{name}_2x.png"); br.close()
    from PIL import Image; Image.open(f"/tmp/{name}_2x.png").resize((W, H), Image.LANCZOS).save(path)
def body(out, X, Y, sc, fill="#e6e6e6"):
    def cap(x1, y1, x2, y2, r): out.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{fill}" stroke-width="{2*r*sc:.1f}" stroke-linecap="round"/>')
    for s in (1, -1):
        cap(s*18, 143, s*50, 143, 5.5); cap(s*50, 143, s*77, 143, 4.5); cap(s*82, 141, s*86, 141, 6.5)
        for y, Ln in [(146, 8), (142.5, 9), (139, 8.5), (135.5, 6.5)]: cap(s*89, y, s*(90+Ln), y, 1.5)
        cap(s*81, 146, s*86.5, 156.5, 1.7); cap(s*11, 90, s*12, 50, 8); cap(s*12, 50, s*13, 9, 5.5); cap(s*10.5, 3, s*16.5, 3, 3.2)
    out.append(f'<polygon points="{X(-20)},{Y(148)} {X(20)},{Y(148)} {X(15)},{Y(108)} {X(19)},{Y(86)} {X(-19)},{Y(86)} {X(-15)},{Y(108)}" fill="{fill}" stroke="{fill}" stroke-width="{3*sc}" stroke-linejoin="round"/>')
    cap(0, 140, 0, 152, 5); out.append(f'<ellipse cx="{X(0)}" cy="{Y(165)}" rx="{10.5*sc}" ry="{12.5*sc}" fill="{fill}"/>')
