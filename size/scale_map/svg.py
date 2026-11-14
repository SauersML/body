import subprocess, os
INK = "#1D1D1F"; C = {"0.6B": "#F0A868", "1.7B": "#E8590C", "8B": "#E07B24", "32B": "#2B5FA8"}
FS = 46; FF = "font-family:'Helvetica Neue',Helvetica,Arial,sans-serif"
class Fig:
    def __init__(s, W, H): s.W, s.H, s.el = W, H, []
    def t(s, x, y, txt, size=FS, w=400, fill=INK, anchor="start"):
        s.el.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" style="{FF}">{txt}</text>')
    def line(s, x1, y1, x2, y2, c=INK, w=4, dash=""):
        s.el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}" {'stroke-dasharray="%s"' % dash if dash else ""}/>')
    def rect(s, x, y, w, h, c, rx=6): s.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}" rx="{rx}"/>')
    def raw(s, x): s.el.append(x)
    def bars(s, x, y, h, items, vmax, bw=150, gap=40, fmt=lambda v: f"{v:.2f}".replace("0.", ".", 1)):
        """items: (label, value, lo, hi, colour). Vertical bars from baseline y+h; CI whiskers; value above; label below."""
        base = y + h; xs = []
        for i, (lab, v, lo, hi, c) in enumerate(items):
            bx = x + i * (bw + gap); top = base - h * v / vmax; s.rect(bx, top, bw, base - top, c, 4); xs.append(bx + bw / 2)
            vy = top - 18
            if lo is not None:
                for yy in (lo, hi): s.line(bx + bw / 2 - 22, base - h * yy / vmax, bx + bw / 2 + 22, base - h * yy / vmax, INK, 4)
                s.line(bx + bw / 2, base - h * lo / vmax, bx + bw / 2, base - h * hi / vmax, INK, 4); vy = min(vy, base - h * hi / vmax - 18)
            s.t(bx + bw / 2, vy, fmt(v), FS, 700, INK, "middle")
            for k, part in enumerate(lab.split("|")): s.t(bx + bw / 2, base + 56 + k * 52, part, FS, 700, INK if c.startswith("#C9") else c, "middle")
        s.line(x - 20, base, x + len(items) * (bw + gap) - gap + 20, base, INK, 4); return xs
    def save(s, path):
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" height="{s.H}" viewBox="0 0 {s.W} {s.H}"><rect width="100%" height="100%" fill="white"/>' + "".join(s.el) + "</svg>"
        open(path + ".svg", "w").write(svg); open(path + ".html", "w").write(f"<html><body style='margin:0'>{svg}</body></html>")
        subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", f"--screenshot={path}.png", f"--window-size={s.W},{s.H}",
                        "--hide-scrollbars", "file://" + os.path.abspath(path + ".html")], capture_output=True, timeout=60)
