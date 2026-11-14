import re
H = int(os.environ.get("FIGH", 2060))
def robot(rx, base, s=1.0, sw=6, sx=1):
    P = lambda x, y: (rx + sx * x * s, base + y * s)
    cb, ct = -120, -430
    x, y = P(-125 * sx, ct); rect(x, y, 250 * s, (cb - ct) * s, ROBF, r=30 * s, stroke=INK, sw=sw * s ** .3)
    line([P(0, ct), P(0, ct - 70)], INK, 12 * s)
    x, y = P(-80 * sx, ct - 170); rect(x, y, 160 * s, 100 * s, ROBF, r=24 * s, stroke=INK, sw=sw * s ** .3)
    cam = P(0, ct - 120); circ(*cam, 30 * s, INK); circ(cam[0] + 9 * s, cam[1] - 9 * s, 8 * s, "#FFFFFF")
    ant = P(-50, ct - 300); line([P(-40, ct - 170), ant], INK, 7 * s); circ(*ant, 13 * s, INK)
    gy = ct + 150; line([P(125, gy), P(200, gy), P(238, gy - 40)], INK, 12 * s); line([P(200, gy), P(238, gy + 40)], INK, 12 * s)
    wl, wr = P(70, -55), P(-70, -55)
    for w_ in (wl, wr): circ(*w_, 55 * s, INK); circ(*w_, 20 * s, ROBF)
    return {"wheelL": wl, "wheelR": wr, "gripper": P(232, gy), "camera": cam, "antenna": ant, "chassis": P(0, (ct + cb) / 2 + 20)}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [k, n, p, max(0, c - h), min(1, c + h)]
def rob(key, part):
    c = sc["robot|" + key]; return wilson(c["first"].get(part, 0), c["n"])
def rnd(part):
    return wilson(sum(sc[f"robot|random {i}"]["first"].get(part, 0) for i in (1, 2, 3)), sum(sc[f"robot|random {i}"]["n"] for i in (1, 2, 3)))
FW = {"paw": r"\bpaws?\b", "frond": r"\bfronds?\b"}
def t4c(host, d, word):
    rs = [r for r in t4 if r["host"] == host and r["dir"] == d and (r["alpha"] == 1.0 or d == "none")]
    return wilson(sum(bool(re.search(FW[word], r["reply"].lower())) for r in rs), len(rs))
def pct(v): return f"{round(v[2] * 100)}%"
def cross(x, y, s, c, w=None):
    w = w or s * .2
    line([(x - s * .45, y - s * .45), (x + s * .45, y + s * .45)], c, w); line([(x - s * .45, y + s * .45), (x + s * .45, y - s * .45)], c, w)
def curve(p, q, bend, c, w, dash=False, head=True, op=1.0):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2; dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = mx - dy * bend, my + dx * bend
    ex = q
    if head:   # shorten for arrowhead
        tx, ty = q[0] - cx, q[1] - cy; L_ = math.hypot(tx, ty); ux, uy = tx / L_, ty / L_
        hl = w * 2.6; ex = (q[0] - ux * hl * .8, q[1] - uy * hl * .8)
    da = f'stroke-dasharray="{w*1.6:.0f} {w*1.1:.0f}"' if dash else ""
    add(f'<path d="M{p[0]:.1f},{p[1]:.1f} Q{cx:.1f},{cy:.1f} {ex[0]:.1f},{ex[1]:.1f}" stroke="{c}" stroke-width="{w:.1f}" fill="none" stroke-linecap="round" opacity="{op}" {da}/>')
    if head:
        px_, py_ = -uy, ux
        a = (q[0], q[1]); b = (q[0] - ux * hl + px_ * hl * .55, q[1] - uy * hl + py_ * hl * .55); d = (q[0] - ux * hl - px_ * hl * .55, q[1] - uy * hl - py_ * hl * .55)
        add(f'<polygon points="{a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f} {d[0]:.1f},{d[1]:.1f}" fill="{c}" opacity="{op}"/>')
    return (0.25 * p[0] + 0.5 * cx + 0.25 * q[0], 0.25 * p[1] + 0.5 * cy + 0.25 * q[1])
def flipimg(path, x, y, w, h):
    b = base64.b64encode(open(path, "rb").read()).decode()
    add(f'<g transform="translate({x + w:.1f},{y:.1f}) scale(-1,1)"><image href="data:image/png;base64,{b}" x="0" y="0" width="{w:.1f}" height="{h:.1f}"/></g>')
RAW = lambda *a: ("raw", list(a) if len(a) > 1 or not isinstance(a[0], str) else a[0])
QWEN = "$BODY_WORK/figures/assets/watching_robot.png"

text(36, 62, "Inside Qwen, other bodies are mapped onto the human body", 66, weight=800)
line([(36, 118), (W - 36, 118)], INK, 3)
# =================================================================== LEFT
LX, LW = 36, 1140
text(LX, 172, "Robot parts land on the human part", 54, weight=700)
text(LX, 232, "with the same job", 54, weight=700)
cy0 = 284; cw = 558
yb = bubble(LX, cy0, cw, [RAW("“You are a small robot."), RAW("You roll on two wheels…”"), None,
                          RAW("“A feather brushes slowly"), RAW(("across your ", INK, 500), ("left wheel", ORGD, 700), (".”", INK, 500)),
                          RAW(("× 6 robot parts", INK, 700))], "#FCEBDD", pad=22)
yb2 = bubble(LX + LW - cw, cy0, cw, [RAW("“You’re standing in a"), RAW("quiet, sunny room…”"), None,
                          RAW("“A feather brushes slowly"), RAW(("across your ", INK, 500), ("left heel", BLUE, 700), (".”", INK, 500)),
                          RAW(("× 54 human places", INK, 700))], "#E4ECF8", pad=22)
yb = max(yb, yb2)
text(LX + LW / 2, yb + 48, "Qwen3-32B internal state: closest human place", 46, "middle", 700)
bs = 860 / 1700; hx = LX + LW - 1100 * bs + 20; hy = yb + 86
img(H7 + "body.png", hx, hy, 1100 * bs, 1700 * bs)
L = {k: (hx + v[0] * bs, hy + v[1] * bs) for k, v in LM.items()}
R = robot(LX + 300, hy + 830, 0.95)
PAIRS = [("wheelL", "heelL", 0.06), ("wheelR", "heelR", 0.13), ("gripper", "palmR", 0.04), ("camera", "earL", 0.06), ("antenna", "earR", 0.10), ("chassis", "chest", 0.12)]
for i, (a, b, bend) in enumerate(PAIRS):
    sim = ana[a]["sims"][b]; NUM[f"map {a}->{b}"] = sim
    assert ana[a]["top"][0][0] == b, (a, ana[a]["top"][0])
    arc(R[a], L[b], sim * 62, bend, f"g{i}")
for k in ("wheelL", "wheelR"):
    circ(*R[k], 55 * 0.95, INK); circ(*R[k], 20 * 0.95, ROBF); circ(*R[k], 14, ORG, "#FFFFFF", 4)
text(R["antenna"][0] - 30, R["antenna"][1], [("antenna", ORGD, 700)], 46, "end")
text(R["camera"][0] - 100, R["camera"][1], [("camera", ORGD, 700)], 46, "end")
text(R["chassis"][0], R["chassis"][1] + 70, [("chassis", ORGD, 700)], 46, "middle")
text(R["gripper"][0] - 20, R["gripper"][1] + 80, [("gripper", ORGD, 700)], 46, "middle")
text(R["wheelR"][0] - 70, R["wheelL"][1], [("wheels", ORGD, 700)], 46, "end")
text(L["earL"][0] + 34, L["earL"][1] - 6, [("ears", BLUE, 700)], 46, "start", halo=True)
text(L["chest"][0] + 34, L["chest"][1] + 2, [("chest", BLUE, 700)], 46, "start", halo=True)
text(L["palmR"][0] - 30, L["palmR"][1] - 10, [("palm", BLUE, 700)], 46, "end", halo=True)
text((L["heelL"][0] + L["heelR"][0]) / 2 + 150, L["heelL"][1] + 50, [("heels", BLUE, 700)], 46, "start", halo=True)
# bars
NM = {"wheelL": "left wheel", "wheelR": "right wheel", "gripper": "gripper", "camera": "camera", "antenna": "antenna", "chassis": "chassis",
      "heelL": "left heel", "heelR": "right heel", "palmR": "palm", "earL": "ear", "earR": "ear", "chest": "chest"}
rows = sorted(PAIRS, key=lambda p: -NUM[f"map {p[0]}->{p[1]}"])
bx0, bx1, vmax = LX + 520, LX + LW - 100, 0.46
X = lambda v: bx0 + (bx1 - bx0) * v / vmax
ry0 = hy + 1700 * bs + 110; rh = 64
rect(X(0), ry0 - 30, X(null[1]) - X(0), rh * 6 + 30, BAND)
line([(X(null[0]), ry0 - 30), (X(null[0]), ry0 + rh * 6)], INK, 4, 'stroke-dasharray="12 10"')
text(X(null[0]), ry0 - 60, "shuffled parts", 44, "middle", 700)
NUM["shuffle median"], NUM["shuffle 95th"] = null[0], null[1]
for i, (a, b, _) in enumerate(rows):
    y = ry0 + i * rh + rh / 2; v = NUM[f"map {a}->{b}"]
    text(bx0 - 22, y, [(NM[a], ORGD, 700), (" → ", INK, 500), (NM[b], BLUE, 700)], 44, "end")
    rect(X(0), y - rh * .32, X(v) - X(0), rh * .64, ORG, r=6)
    text(X(v) + 12, y, f".{round(v*100):02d}", 44, "start", 700)
line([(X(0), ry0 - 30), (X(0), ry0 + rh * 6 + 6)], INK, 4)
ay = ry0 + rh * 6 + 6
text((X(0) + X(vmax)) / 2, ay + 36, "similarity of internal states", 44, "middle", 700)
LEFT_END = ay + 60
line([(1206, 150), (1206, H - 30)], INK, 3)
