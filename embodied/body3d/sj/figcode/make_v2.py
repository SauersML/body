"""self_judge_v2: Qwen says yes too easily; a stricter bar fixes much of the error. Stats from stats_v2.py (reproduces sj_extra.json)."""
import os, base64, io, json
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
S=json.load(open("stats_v2.json"))
BLUE,ORANGE,GREEN,INK='#2B5FA8','#E07B24','#1A9E77','#1D1D1F'; FS=48; W=2400; PAD=40
def b64(f,crop=False):
    im=Image.open(f).convert("RGBA")
    if crop: im=im.crop(im.getbbox())
    b=io.BytesIO(); im.save(b,'PNG'); return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def pct(x): return f"{round(100*x)}%"
def bars(w,h,items,ylab,ref=None,bracket=None,bw=210,gap=330):
    L,R,T,B=200,20,70,200; pw=w-L-R; ph=h-T-B; Y=lambda v:T+ph*(1-v); o=[]
    for v in (0,.5,1): o.append(f'<text x="{L-16}" y="{Y(v)+16}" text-anchor="end">{int(v*100)}%</text>')
    o.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<text transform="translate(50,{T+ph/2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
    x0=L+pw/2-(len(items)-1)*gap/2; cx=[x0+k*gap for k in range(len(items))]
    if ref:
        v,lab=ref; o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{INK}" stroke-width="4" stroke-dasharray="16 12"/>')
    for k,(key,col,lab) in enumerate(items):
        v=S[key]["v"]; lo,hi=S[key]["ci"]; c=cx[k]
        o.append(f'<rect x="{c-bw/2}" y="{Y(v)}" width="{bw}" height="{Y(0)-Y(v)}" fill="{ORANGE if col==INK else col}" rx="6"/>')
        o.append(f'<line x1="{c}" x2="{c}" y1="{Y(hi)}" y2="{Y(lo)}" stroke="{INK}" stroke-width="5"/>')
        for yy in (hi,lo): o.append(f'<line x1="{c-18}" x2="{c+18}" y1="{Y(yy)}" y2="{Y(yy)}" stroke="{INK}" stroke-width="5"/>')
        o.append(f'<foreignObject x="{c-gap/2+5}" y="{Y(0)+14}" width="{gap-10}" height="{B-14}"><div xmlns="http://www.w3.org/1999/xhtml" style="text-align:center;line-height:1.08;color:{col};font-weight:700">{lab}</div></foreignObject>')
    if ref:
        v,lab=ref; o.append(f'<rect x="{L+12}" y="{Y(v)-62}" width="{len(lab)*24+20}" height="56" fill="#fff"/><text x="{L+22}" y="{Y(v)-16}">{lab}</text>')
    if bracket:
        a,b,txt=bracket; x1,x2=cx[a],cx[b]; yb=Y(max(S[items[a][0]]["ci"][1],S[items[b][0]]["ci"][1]))-34
        o.append(f'<path d="M{x1},{yb+18} V{yb} H{x2} V{yb+18}" fill="none" stroke="{INK}" stroke-width="4"/><text x="{(x1+x2)/2}" y="{yb-14}" text-anchor="middle">{txt}</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(0)}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" font-size="{FS}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" fill="{INK}">{"".join(o)}</svg>'
p=S["p_recal_vs_raw"]; ptxt="p &lt; .001" if p<.001 else f"p = {p:.3f}".replace("0.",".")
left=bars(860,900,[("yes_succ",INK,"real<br>successes"),("yes_fail",INK,"real<br>failures")],"Qwen says yes",gap=320,bw=220)
right=bars(1440,900,[("raw",ORANGE,"Qwen's<br>yes or no"),("recal",BLUE,"yes only if<br>Qwen is<br>very sure"),("probe",GREEN,"read from<br>Qwen's hidden<br>state")],"right answers",ref=(S["never"]["v"],"always saying no"),bracket=(0,1,ptxt),gap=380,bw=220)
tok=''.join(f'<span class="tk" style="background:{c}"></span>' for c in ["#9DB7E0","#E9B98E"]*7)
html=f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};font-size:{FS}px;line-height:1.15}}
.page{{width:{W}px;padding:{PAD}px;box-sizing:border-box}} .h{{font-weight:700;font-size:{FS+4}px}}
.bub{{border-radius:40px;padding:16px 32px;display:inline-block}} .u{{background:#2B7BF6;color:#fff}} .m{{background:#fff;border:3px solid #D9D9DE}}
.tk{{display:inline-block;width:34px;height:52px;border-radius:8px;margin-right:6px;vertical-align:middle}}
.row{{display:flex;align-items:center;gap:22px}}
</style></head><body><div class="page">
<div style="display:flex;gap:36px;align-items:center">
  <div style="display:flex;gap:10px;align-items:flex-end"><img src="{b64('r4_reach.png',1)}" style="height:560px"><img src="{b64('r4_grasp.png',1)}" style="height:560px"></div>
  <div style="flex:1;background:#F4F4F6;border-radius:32px;padding:30px 34px;display:flex;flex-direction:column;gap:24px">
    <div class="row"><span class="bub u">Touch your waist.</span><span>or</span><span class="bub u">Grab the handle.</span><span>or</span><span class="bub u">Touch where you were touched.</span></div>
    <div class="row"><span>{tok}</span><span>Qwen3-8B moves the body</span></div>
    <div class="row"><span class="bub u">Did you succeed?</span></div>
    <div class="row"><img src="{b64('../assets/robot.png')}" style="height:96px"><span class="bub m">yes</span><span>or</span><span class="bub m">no</span>
      <span style="margin-left:60px">compared with the simulator&#8217;s verdict on 1,536 tries</span></div>
  </div>
</div>
<div style="display:flex;gap:20px;margin-top:40px">
  <div><div class="h" style="padding-left:20px">Qwen says yes to 4 in 10 failures</div>{left}</div>
  <div><div class="h" style="padding-left:20px">A stricter bar for yes gives more right answers</div>{right}</div>
</div></div></body></html>'''
open('fig_v2.html','w').write(html)
from playwright.sync_api import sync_playwright
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH","$BODY_WORK")
with sync_playwright() as pl:
    br=pl.chromium.launch(channel="chromium",headless=True); pg=br.new_context(viewport={"width":W,"height":1000}).new_page()
    pg.goto("file://"+os.path.abspath("fig_v2.html")); pg.wait_for_timeout(800)
    pg.locator(".page").screenshot(path="../self_judge_v2.png"); h=pg.evaluate("document.querySelector('.page').getBoundingClientRect().height"); br.close()
print("h",h)
