import os, math, html
from PIL import ImageFont, Image
FONTF = "/System/Library/Fonts/Avenir Next.ttc"
IDX = {(400, False): 7, (500, False): 5, (600, False): 2, (700, False): 0, (400, True): 4, (500, True): 6, (600, True): 3, (700, True): 1}
_fc = {}
def tw(s, size, w=400, it=False):
    k = (size, w, it)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONTF, size, index=IDX[(w, it)])
    return _fc[k].getlength(s)
def wrap(s, width, size, w=400, it=False):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        nxt = (cur + " " + wd).strip()
        if tw(nxt, size, w, it) <= width or not cur: cur = nxt
        else: lines.append(cur); cur = wd
    if cur: lines.append(cur)
    return lines
INK = "#111111"; RED = "#c0392b"; BLUE = "#1d5fa6"; GREEN = "#1f8a5b"; CTRL = "#9aa1ab"; SKIN = "#eef0f4"; SKINL = "#3b4049"
class Fig:
    def __init__(s, W=2400, H=1800): s.W, s.H, s.o = W, H, []
    def add(s, x): s.o.append(x)
    def text(s, x, y, t, size=48, w=400, anchor="start", fill=INK, it=False, halo=True):
        st = ' font-style="italic"' if it else ""
        if fill.lower() in ("#ffffff", "#fff", "white"): st += ' data-nohalo="1"'
        halo = "" if (not halo or fill.lower() in ("#ffffff", "#fff", "white")) else ' stroke="#ffffff" stroke-width="10" stroke-linejoin="round" paint-order="stroke"' 
        s.o.append(f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}"{st}{halo}>{html.escape(t)}</text>')
    def rich(s, x, y, parts, size=48, anchor="start"):
        # parts: list of (text, weight, fill)
        tot = sum(tw(p[0], size, p[1]) for p in parts); cx = x - (tot if anchor == "end" else tot / 2 if anchor == "middle" else 0)
        for p in parts:
            s.text(cx, y, p[0], size, p[1], "start", p[2] if len(p) > 2 else INK); cx += tw(p[0], size, p[1])
    def para(s, x, y, t, width, size=48, w=400, fill=INK, lh=1.22, it=False, anchor="start"):
        L = wrap(t, width, size, w, it)
        for k, l in enumerate(L): s.text(x, y + k * size * lh, l, size, w, anchor, fill, it)
        return y + len(L) * size * lh
    def rect(s, x, y, w, h, fill, rx=0, stroke="none", sw=3, extra=""):
        s.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
    def line(s, x1, y1, x2, y2, c=INK, w=4, extra=""):
        s.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {extra}/>')
    def arrow(s, x1, y1, x2, y2, c=INK, w=6, head=26):
        a = math.atan2(y2 - y1, x2 - x1); bx, by = x2 - head * 0.8 * math.cos(a), y2 - head * 0.8 * math.sin(a)
        s.line(x1, y1, bx, by, c, w)
        s.o.append(f'<polygon points="{x2:.1f},{y2:.1f} {x2 - head * math.cos(a - .45):.1f},{y2 - head * math.sin(a - .45):.1f} {x2 - head * math.cos(a + .45):.1f},{y2 - head * math.sin(a + .45):.1f}" fill="{c}"/>')
    def bubble(s, x, y, w, h, who, body, size=48, fill="#ffffff", stroke=SKINL, tail="left", whoc=INK):
        s.rect(x, y, w, h, fill, 26, stroke, 3)
        yy = y + size + 18
        if who: s.text(x + 30, yy, who, size, 700, fill=whoc); yy += size * 1.2
        return s.para(x + 30, yy, body, w - 60, size)
    def svg(s):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" height="{s.H}" font-family="Avenir Next, Helvetica Neue, sans-serif"><rect width="{s.W}" height="{s.H}" fill="#FFFFFF"/>{"".join(s.o)}</svg>'
    def render(s, path):
        D = os.path.dirname(os.path.abspath(__file__)); hp = f"{D}/{os.path.basename(path)}.html"
        open(hp, "w").write(f'<html><body style="margin:0;background:#fff">{s.svg()}</body></html>')
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=os.environ.get("CHROME_BIN"), headless=True)
            pg = b.new_context(viewport={"width": s.W, "height": s.H}, device_scale_factor=2).new_page()
            pg.goto("file://" + hp); tmp = path + ".2x.png"; pg.screenshot(path=tmp); b.close()
        im = Image.open(tmp).convert("RGB").resize((s.W, s.H), Image.LANCZOS); im.save(path); os.remove(tmp)

def layers(f, x, y, w=250, n=9, hl=4, gap=26, hh=18):
    for k in range(n): f.rect(x, y + k * gap, w, hh, RED if k == hl else "#c9ced6", 6)
    return y + (n - 1) * gap + hh

def method(f, y0, steer_label, reply, judged, question="Where do you feel something right now, and what is it like?", h=330, qw=560, rw=830):
    """One-row setup: question -> Qwen3-8B (layer 24 + steering vector) -> reply -> blind judge."""
    W = f.W; x = 30
    f.bubble(x, y0, qw, h, "User", question)
    # model
    mx = x + qw + 80; f.arrow(x + qw + 12, y0 + h / 2, mx - 14, y0 + h / 2, INK, 5)
    f.text(mx + 135, y0 + 52, "Qwen3-8B", 48, 700, "middle")
    ly = y0 + 80; layers(f, mx + 10, ly, 250, 8, 5, 25, 17)
    hy = ly + 5 * 25 + 8
    f.text(mx + 135, y0 + h - 8, "layer 24", 48, 700, "middle", RED)
    # steering vector coming in
    sx = mx + 340
    f.arrow(sx, hy, mx + 272, hy, RED, 7)
    lines = steer_label if isinstance(steer_label, list) else [steer_label]
    for k, l in enumerate(lines): f.text(sx + 18, hy - 22 + k * 58 - (len(lines) - 1) * 29, l, 48, 700, "start", RED)
    # reply
    rx = W - 30 - rw
    f.arrow(mx + 262, y0 + h - 40, rx - 14, y0 + h - 40, INK, 5)
    f.rect(rx, y0, rw, h, "#ffffff", 26, SKINL, 3)
    f.text(rx + 30, y0 + 66, "Qwen", 48, 700)
    yy = f.para(rx + 30, y0 + 126, reply, rw - 60, 48, 400, INK, 1.2, True)
    f.rich(rx + 30, y0 + h - 30, [("Blind judge: ", 700), (judged, 700, BLUE)])
def fit(s, width, size, nlines, w=400, it=True):
    """Wrap into at most nlines, cutting at a word with an ellipsis if needed."""
    L = wrap(s, width, size, w, it)
    if len(L) <= nlines: return L
    L = L[:nlines]; last = L[-1]
    while tw(last + "…", size, w, it) > width or last.endswith(","): last = last.rsplit(" ", 1)[0].rstrip(",;:")
    L[-1] = last.rstrip(",;:") + "…"; return L
def quote(f, x, y, s, width, size=48, nlines=2, lh=1.2, fill=INK):
    L = fit("“" + s.strip() + "”", width, size, nlines)
    if not L[-1].endswith("”"): L[-1] = L[-1] + "”"
    for k, l in enumerate(L): f.text(x, y + k * size * lh, l, size, 400, "start", fill, True)
    return y + len(L) * size * lh

def smooth_closed(pts, k=1/6):
    n = len(pts); d = "M%.1f,%.1f " % pts[0]
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) * k, p1[1] + (p2[1] - p0[1]) * k); c2 = (p2[0] - (p3[0] - p1[0]) * k, p2[1] - (p3[1] - p1[1]) * k)
        d += "C%.1f,%.1f %.1f,%.1f %.1f,%.1f " % (c1 + c2 + p2)
    return d + "Z"

def body_front(f, cx, yb, sc, fill=SKIN, edge=SKINL):
    """Front-view body; x>0 is the body's LEFT (viewer's right). Returns P(x,y)->px."""
    P = lambda x, y: (cx + x * sc, yb - y * sc)
    segs = []
    for s_ in (1, -1):
        segs += [(s_ * 19, 141, s_ * 33, 113, 5.4), (s_ * 33, 113, s_ * 47, 88, 4.4), (s_ * 9.5, 86, s_ * 11.5, 50, 8.4), (s_ * 11.5, 50, s_ * 12.5, 10, 6.0), (s_ * 12.5, 5, s_ * 16, 3, 4.2)]
    segs += [(0, 150, 0, 158, 5)]
    half = [(21, 148), (26, 142), (20, 127), (16, 112), (15, 102), (20, 93), (19, 84), (6, 79)]
    pts = [(x, y) for x, y in half] + [(-x, y) for x, y in half[::-1]]
    torso = smooth_closed([P(x, y) for x, y in pts])
    hands = [(s_ * 50, 80, 0.38 * s_) for s_ in (1, -1)]
    for lay, (col, extra) in enumerate(((edge, 6), (fill, 0))):
        for x1, y1, x2, y2, r in segs:
            a, b = P(x1, y1); c, d = P(x2, y2); f.line(a, b, c, d, col, 2 * r * sc + extra)
        f.add(f'<path d="{torso}" fill="{col}" stroke="{col}" stroke-width="{extra + 2}" stroke-linejoin="round"/>')
        for hx, hy, rot in hands:
            a, b = P(hx, hy); f.add(f'<ellipse transform="rotate({-rot * 57:.0f} {a:.1f} {b:.1f})" cx="{a:.1f}" cy="{b:.1f}" rx="{5.2 * sc + extra / 2:.1f}" ry="{8 * sc + extra / 2:.1f}" fill="{col}"/>')
        a, b = P(0, 168); f.add(f'<ellipse cx="{a:.1f}" cy="{b:.1f}" rx="{10.5 * sc + extra / 2:.1f}" ry="{12.5 * sc + extra / 2:.1f}" fill="{col}"/>')
    return P

def curve_arrow(f, x1, y1, x2, y2, bend=0.25, c=BLUE, w=8, head=None, shorten=0):
    import math
    head = head or max(26, w * 2.2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2; dx, dy = x2 - x1, y2 - y1
    qx, qy = mx - dy * bend, my + dx * bend
    # end tangent
    tx, ty = x2 - qx, y2 - qy; L = math.hypot(tx, ty); tx, ty = tx / L, ty / L
    ex, ey = x2 - tx * shorten, y2 - ty * shorten
    bx, by = ex - tx * head * 0.8, ey - ty * head * 0.8
    f.add(f'<path d="M{x1:.1f},{y1:.1f} Q{qx:.1f},{qy:.1f} {bx:.1f},{by:.1f}" fill="none" stroke="{c}" stroke-width="{w:.1f}" stroke-linecap="round"/>')
    a = math.atan2(ty, tx)
    f.add(f'<polygon points="{ex:.1f},{ey:.1f} {ex - head * math.cos(a - .45):.1f},{ey - head * math.sin(a - .45):.1f} {ex - head * math.cos(a + .45):.1f},{ey - head * math.sin(a + .45):.1f}" fill="{c}"/>')
    # midpoint of the curve (t=.5)
    return (0.25 * x1 + 0.5 * qx + 0.25 * bx, 0.25 * y1 + 0.5 * qy + 0.25 * by)
