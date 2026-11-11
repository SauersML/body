import os
import bodies as BD, shapes, figlib, proj, numpy as np
W, H = 2400, 1500; cw, ch = 600, 500; out = []
for i, b in enumerate(BD.ORDER):
    ox, oy = (i % 4) * cw, (i // 4) * ch
    P = np.array([proj.proj(b, p["xyz"]) for p in BD.B[b]["parts"]])
    els = shapes.DRAW[b]()
    lo = P.min(0) - 12; hi = P.max(0) + 12; s = min((cw - 60) / (hi[0] - lo[0]), (ch - 100) / (hi[1] - lo[1]))
    cx, cy = (lo + hi) / 2
    T = lambda u, v, ox=ox, oy=oy, s=s, cx=cx, cy=cy: (ox + cw / 2 + (u - cx) * s, oy + ch / 2 - (v - cy) * s)
    out.append(figlib.draw(els, T, s))
    for (u, v), p in zip(P, BD.B[b]["parts"]):
        x, y = T(u, v); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#3a7" stroke="#fff" stroke-width="3"/>')
    out.append(f'<text x="{ox+20}" y="{oy+40}" font-size="36" font-family="Inter">{b}</text>')
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">{"".join(out)}</svg>'
figlib.render(f'<html><body style="margin:0;background:#fff">{svg}</body></html>', os.path.expandvars("$BODY_WORK/")+"body/nonhuman/_prev.png", W, H)
