import numpy as np, os
from playwright.sync_api import sync_playwright
EXE = os.environ.get("CHROME_BIN")
NEAR, FAR = "#E3DED6", "#CFC8BD"
def cr(pts, closed=True):
    P = [np.array(p, float) for p in pts]; n = len(P); d = f"M{P[0][0]:.1f},{P[0][1]:.1f}"
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0, p1, p2, p3 = P[(i - 1) % n] if (closed or i > 0) else P[0], P[i], P[(i + 1) % n], P[(i + 2) % n] if (closed or i + 2 < n) else P[-1]
        c1 = p1 + (p2 - p0) / 6; c2 = p2 - (p3 - p1) / 6
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d + (" Z" if closed else "")
def draw(elems, T, s):
    """elems from shapes.py; T maps (u, v) -> screen; s = px per unit."""
    out = []
    for far in (True, False):
        c = FAR if far else NEAR
        for kind, a, f in elems:
            if f != far: continue
            if kind == "ell":
                u, v, ru, rv, rot = a; x, y = T(u, v)
                out.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{ru * s:.1f}" ry="{rv * s:.1f}" transform="rotate({-rot} {x:.1f} {y:.1f})" fill="{c}"/>')
            elif kind == "limb":
                pts, w = a; out.append(f'<path d="{cr([T(*p) for p in pts], False)}" stroke="{c}" stroke-width="{w * s:.1f}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
            elif kind == "blob":
                out.append(f'<path d="{cr([T(*p) for p in a[0]])}" fill="{c}"/>')
            elif kind in ("poly", "star"):
                pts = [T(*p) for p in a[0]]
                if kind == "star": out.append(f'<path d="{cr(pts)}" fill="{c}"/>')
                else: out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{c}" stroke="{c}" stroke-width="{1.2 * s:.1f}" stroke-linejoin="round"/>')
            elif kind == "taper":
                pts, w0, w1 = a; P = np.array([T(*p) for p in pts]); n = len(P)
                L, R = [], []
                for i in range(n):
                    t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]; t /= np.linalg.norm(t) + 1e-9; nn = np.array([-t[1], t[0]])
                    w = (w0 + (w1 - w0) * (1 - i / (n - 1))) * s / 2 if kind == "taper" else 0
                    L.append(P[i] + nn * w); R.append(P[i] - nn * w)
                out.append(f'<path d="{cr(L + R[::-1])}" fill="{c}"/>')
    return "".join(out)
def render(html, path, W, H):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.expanduser("~/.cache/ms-playwright"))
    open(path.replace(".png", ".html"), "w").write(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=EXE, headless=True)
        pg = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1).new_page()
        pg.goto("file://" + path.replace(".png", ".html")); pg.wait_for_timeout(500); pg.screenshot(path=path); b.close()
