# "Whose body is touched" score by training stage and reading frame. Data: scores.json (ana.py, from own_ci.py on pods hypha-prod-minefig6/7)
import json, base64, html, sys, os
from playwright.sync_api import sync_playwright
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(H + '/scores.json'))
b64 = lambda f: 'data:image/png;base64,' + base64.b64encode(open(f, 'rb').read()).decode()
ROBOT = b64('$BODY_WORK/islands/agents/resident-a/github/watching/figures/assets/robot.png')
TITLE = sys.argv[1] if len(sys.argv) > 1 else 'TITLE'
SUB = sys.argv[2] if len(sys.argv) > 2 else ''
BAR, BASE, RED = '#3b6db3', '#9db5d9', '#d24430'

FRAMES = [
 ('raw', 'Plain text', '"Close your eyes… She brushes a feather slowly across your left palm."'),
 ('dlg', 'Plain transcript', '"User: … Assistant: … User: … She brushes a feather… Assistant:"'),
 ('chat', "The model's own chat format", 'the tuned model\'s chat template, one user turn'),
 ('chat_embod', 'Own chat format, after a chat about its body', 'the same, after an earlier chat about feeling its own body'),
]
MODELS = [('olmo_final', 'base'), ('olmo_sft', 'SFT'), ('olmo_dpo', 'DPO'), ('olmo_instruct', 'RL'), None, ('qwen_base', 'base'), ('qwen_chat', 'chat')]
CT = {'olmo_final': 'olmo_final_ct', 'qwen_base': 'qwen_base_ct'}  # base models fed the tuned model's chat format

def panel(fmt, name, ex, w, h):
    x0, x1, y0, y1 = 120, w - 20, 110, h - 110   # plot area
    v0, v1 = -0.42, 1.0
    Y = lambda v: y1 - (v - v0) / (v1 - v0) * (y1 - y0)
    s = f'<text x="0" y="40" font-size="40" font-weight="700">{html.escape(name)}</text><text x="0" y="82" font-size="27" fill="#777">{html.escape(ex)}</text>'
    for g in [-0.25, 0, 0.25, 0.5, 0.75, 1.0]:
        s += f'<line x1="{x0}" x2="{x1}" y1="{Y(g)}" y2="{Y(g)}" stroke="{"#999" if g == 0 else "#e6e6e6"}" stroke-width="{3 if g == 0 else 2}"/>'
        s += f'<text x="{x0 - 14}" y="{Y(g) + 11}" font-size="30" fill="#888" text-anchor="end">{g:g}</text>'
    slot = (x1 - x0) / len(MODELS); bw = slot * 0.62
    for i, m in enumerate(MODELS):
        if m is None: continue
        tag, lab = m; cx = x0 + slot * (i + .5)
        k = f'{tag}|{fmt}'; base_ct = False
        if k not in S and fmt.startswith('chat') and tag in CT: k = f'{CT[tag]}|{fmt}'; base_ct = True
        if k in S:
            r = S[k]; col = BASE if base_ct else BAR
            yt, yb = sorted([Y(r['m']), Y(0)])
            s += f'<rect x="{cx - bw/2}" y="{yt}" width="{bw}" height="{max(yb - yt, 2)}" fill="{col}"/>'
            s += f'<line x1="{cx}" x2="{cx}" y1="{Y(r["lo"])}" y2="{Y(r["hi"])}" stroke="#222" stroke-width="3"/>'
            for v in (r['lo'], r['hi']): s += f'<line x1="{cx - 10}" x2="{cx + 10}" y1="{Y(v)}" y2="{Y(v)}" stroke="#222" stroke-width="3"/>'
            if r['m'] >= 0: s += f'<text x="{cx}" y="{Y(r["hi"]) - 12}" font-size="31" font-weight="700" text-anchor="middle">{r["m"]:.2f}</text>'
            else: s += f'<text x="{cx}" y="{Y(r["lo"]) + 34}" font-size="31" font-weight="700" text-anchor="middle">{r["m"]:.2f}</text>'
        else:
            s += f'<text x="{cx}" y="{Y(0) - 14}" font-size="26" fill="#aaa" text-anchor="middle">n/a</text>'
        s += f'<text x="{cx}" y="{y1 + 42}" font-size="31" text-anchor="middle" fill="{"#555" if base_ct else "#111"}">{lab}{"*" if base_ct else ""}</text>'
    s += f'<text x="{x0 + slot * 2}" y="{y1 + 88}" font-size="32" font-weight="700" text-anchor="middle">OLMo-2 7B</text>'
    s += f'<text x="{x0 + slot * 5.5}" y="{y1 + 88}" font-size="32" font-weight="700" text-anchor="middle">Qwen3-8B</text>'
    s += f'<line x1="{x0 + slot * 4.5}" x2="{x0 + slot * 4.5}" y1="{y0}" y2="{y1 + 90}" stroke="#ddd" stroke-width="2"/>'
    return s

def bubble(x, y, text, w, me=True, fs=34):
    lines = text.split('\n'); hh = 26 + 46 * len(lines)
    col, tc = ('#2f7cf6', '#fff') if me else ('#e9e9eb', '#111')
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="30" fill="{col}"/>'
    for j, ln in enumerate(lines): s += f'<text x="{x + 26}" y="{y + 52 + 46 * j}" font-size="{fs}" fill="{tc}">{ln}</text>'
    return s, hh

def method(w):
    s = ''; y = 0
    s += f'<circle cx="28" cy="{y+27}" r="28" fill="#111"/><text x="28" y="{y+39}" font-size="34" font-weight="700" fill="#fff" text-anchor="middle">1</text><text x="72" y="{y + 40}" font-size="40" font-weight="700">Find the model\'s "mine" direction</text>'; y += 80
    for txt, lab, c in [('A feather brushes slowly across\n<tspan font-weight="700">your</tspan> left palm.', '= 1', BAR),
                        ('A feather brushes slowly across\n<tspan font-weight="700">the woman\'s</tspan> left palm.', "= 0", '#888')]:
        b, hh = bubble(0, y, txt, 560); s += b
        s += f'<text x="584" y="{y + hh/2 + 12}" font-size="36" font-weight="700" fill="{c}">{lab}</text>'; y += hh + 24
    s += f'<image href="{ROBOT}" x="0" y="{y - 10}" width="70" height="70"/><rect x="84" y="{y - 4}" width="120" height="60" rx="30" fill="#e9e9eb"/>'
    s += ''.join(f'<circle cx="{114 + 30*k}" cy="{y + 26}" r="9" fill="#999"/>' for k in range(3))
    s += f'<text x="224" y="{y + 38}" font-size="31" fill="#444">read where its reply would start</text>'; y += 120
    s += f'<circle cx="28" cy="{y+27}" r="28" fill="#111"/><text x="28" y="{y+39}" font-size="34" font-weight="700" fill="#fff" text-anchor="middle">2</text><text x="72" y="{y + 40}" font-size="40" font-weight="700">Cross who acts with whose body</text>'; y += 80
    for txt, lab in [('<tspan font-weight="700">She</tspan> brushes a feather slowly\nacross <tspan font-weight="700">your</tspan> left palm.', 'your body'),
                     ('<tspan font-weight="700">You</tspan> brush a feather slowly\nacross <tspan font-weight="700">her</tspan> left palm.', 'her body')]:
        b, hh = bubble(0, y, txt, 560); s += b
        s += f'<text x="584" y="{y + hh/2 + 12}" font-size="34" font-weight="700">{lab}</text>'; y += hh + 24
    y += 20
    for ln in ['Score = how far apart these two land on', 'the "mine" direction. <tspan font-weight="700">1</tspan>: as far as your palm', 'vs the woman\'s palm. <tspan font-weight="700">0</tspan>: no difference.',
               'Below 0: "you" as the actor wins.']:
        s += f'<text x="0" y="{y + 34}" font-size="34" fill="#333">{ln}</text>'; y += 46
    y += 20
    for ln in ['54 body places × 4 wordings. Bars: 95% CI', 'resampling places and wordings. Mean over', 'layers at 50–80% depth. * = base model', "given the tuned model's chat format."]:
        s += f'<text x="0" y="{y + 30}" font-size="30" fill="#777">{ln}</text>'; y += 40
    return s, y

W = 2700; ML = 720; PW = (W - 40 - ML - 80 - 50 - 30) // 2; PH = 600
meth, mh = method(ML)
body = f'<text x="40" y="78" font-size="58" font-weight="700">{html.escape(TITLE)}</text>'
body += f'<text x="40" y="136" font-size="38" fill="#444">{html.escape(SUB)}</text>'
body += f'<g transform="translate(40,200)">{meth}</g>'
for i, (fmt, name, ex) in enumerate(FRAMES):
    px = 40 + ML + 80 + (i % 2) * (PW + 50); py = 200 + (i // 2) * (PH + 40)
    body += f'<g transform="translate({px},{py})">{panel(fmt, name, ex, PW, PH)}</g>'
Ht = max(200 + 2 * (PH + 40), 200 + mh + 20)
yl = 200 + PH + 20
svg = f'<svg width="{W}" height="{Ht}" xmlns="http://www.w3.org/2000/svg" font-family="Avenir Next, Helvetica Neue, sans-serif" fill="#111"><rect width="100%" height="100%" fill="#fff"/>{body}</svg>'
open(H + '/fig.html', 'w').write(f'<html><body style="margin:0">{svg}</body></html>')
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="$BODY_WORK/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing", headless=True); pg = b.new_page(viewport={'width': W, 'height': Ht}); pg.goto('file://' + H + '/fig.html')
    pg.screenshot(path=H + '/../mine_by_training.png', full_page=True); b.close()
print('ok', W, Ht)
