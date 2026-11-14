# steer_base_vs_final.png: per place, naming gain with the place vector, base (x) vs post-trained (y); two families.
import os, sys, json, numpy as np, http.server, functools, socketserver, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ana
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "$BODY_WORK"
OUT = sys.argv[1]; PNG = "$BODY_WORK/fig/steer_base_vs_final.png"
ACC = "#c2255c"; INK = "#121217"; BLU = "#1c6fb5"; LINE = "#9a9aa3"
FONT = "font-family:'Inter','Helvetica Neue',Arial,sans-serif"
def pfmt(p): return "p < 0.001" if p < 1e-3 else f"p = {p:.2g}"
def panel(t, name_b, name_p, color, x0, y0, lo, hi, S=1020):
    ga, gb = np.array(t["ga"]) * 100, np.array(t["gb"]) * 100
    f = lambda v: S * (v - lo) / (hi - lo)
    X = lambda v: x0 + f(v); Y = lambda v: y0 + S - f(v)
    s = f'<rect x="{x0}" y="{y0}" width="{S}" height="{S}" fill="none" stroke="#d8d8de" stroke-width="3"/>'
    step = 10
    for g in np.arange(lo, hi + 1, step):
        s += f'<line x1="{X(g)}" x2="{X(g)}" y1="{y0}" y2="{y0+S}" stroke="#efeff3" stroke-width="3"/><line x1="{x0}" x2="{x0+S}" y1="{Y(g)}" y2="{Y(g)}" stroke="#efeff3" stroke-width="3"/>'
        s += f'<text x="{X(g)}" y="{y0+S+66}" font-size="52" text-anchor="middle">{g:.0f}</text><text x="{x0-22}" y="{Y(g)+18}" font-size="52" text-anchor="end">{g:.0f}</text>'
    s += f'<line x1="{X(lo)}" y1="{Y(lo)}" x2="{X(hi)}" y2="{Y(hi)}" stroke="{LINE}" stroke-width="5" stroke-dasharray="18 14"/>'
    s += f'<text x="{X(hi)-30}" y="{Y(hi)+130}" font-size="48" text-anchor="end" fill="{INK}" transform="rotate(-45 {X(hi)-30} {Y(hi)+130})">equal</text>'
    for a, b in zip(ga, gb): s += f'<circle cx="{X(a)}" cy="{Y(b)}" r="17" fill="{color}" fill-opacity="0.72" stroke="white" stroke-width="3"/>'
    # label the two places furthest above and below the diagonal
    d = gb - ga
    for k in [int(np.argmax(d)), int(np.argmin(d))]:
        s += f'<text x="{X(ga[k])+26}" y="{Y(gb[k])+16}" font-size="48" fill="{INK}">{t["places"][k]}</text>'
    s += f'<text x="{x0+S/2}" y="{y0+S+140}" font-size="58" text-anchor="middle">{name_b}</text>'
    s += f'<text x="{x0-140}" y="{y0+S/2}" font-size="58" text-anchor="middle" transform="rotate(-90 {x0-140} {y0+S/2})">{name_p}</text>'
    sign = "+" if t["d"] >= 0 else "−"
    s += f'<text x="{x0+40}" y="{y0+90}" font-size="62" font-weight="700" fill="{color}">{"Weaker" if t["d"] < 0 else "Stronger"} after post-training</text>'
    s += f'<text x="{x0+40}" y="{y0+165}" font-size="54" fill="{INK}">{sign}{abs(t["d"])*100:.0f} points, {pfmt(t["p"])}</text>'
    return s
def main():
    to = ana.test(f"{OUT}/olmo_base.json", f"{OUT}/olmo_instruct.json"); tq = ana.test(f"{OUT}/qwen_base.json", f"{OUT}/qwen_chat.json")
    v = np.array(to["ga"] + to["gb"] + tq["ga"] + tq["gb"]) * 100; lo = min(0, np.floor(v.min() / 10) * 10); hi = np.ceil(v.max() / 10) * 10
    W, H = 2600, 1930
    m = f'<text x="80" y="110" font-size="62" font-weight="700">1. Place vector, from the model\'s own states</text>'
    m += f'<text x="80" y="190" font-size="54">"An ice cube touches your left knee."  × 16 knee sentences, averaged, minus the all-place mean</text>'
    m += f'<text x="80" y="300" font-size="62" font-weight="700">2. Add the knee vector at 65% depth, sample 32 replies</text>'
    m += f'<text x="80" y="380" font-size="54">"I woke up this morning with a strange ache in my <tspan fill="{ACC}" font-weight="700">knee</tspan>"  × 8 prompts × 3 strengths</text>'
    m += f'<text x="80" y="490" font-size="62" font-weight="700">3. Score: % of replies that say "knee", minus % without the vector</text>'
    m += f'<text x="80" y="570" font-size="54">One dot = one of 38 body places</text>'
    body = m + panel(to, "OLMo-2 7B base", "OLMo-2 7B Instruct (after RL)", ACC, 250, 700, lo, hi) + panel(tq, "Qwen3-8B base", "Qwen3-8B chat", BLU, 1510, 700, lo, hi)
    html = f'<html><head><meta charset="utf-8"></head><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="{FONT};fill:{INK}"><rect width="{W}" height="{H}" fill="#fff"/>{body}</svg></body></html>'
    d = os.path.dirname(os.path.abspath(__file__)); open(f"{d}/fig.html", "w").write(html)
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)); threading.Thread(target=srv.serve_forever, daemon=True).start()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_page(viewport={"width": W, "height": H})
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/fig.html"); pg.wait_for_timeout(300); pg.screenshot(path=PNG); br.close()
    print(PNG)
main()
