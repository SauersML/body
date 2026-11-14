import os
from playwright.sync_api import sync_playwright
W=2600
BLUE="#1f4e8c"; TEAL="#2a8a84"; GREY="#9aa4b0"; LG="#d5dae0"
def hbar(label,v,col,maxw=1220):
    w=max(v*maxw/100,6)
    return f'''<div class="hb"><div class="hl">{label}</div><div class="track" style="width:{maxw}px"><div class="bar" style="width:{w}px;background:{col}"></div><div class="val" style="left:{w+18}px">{v}%</div></div></div>'''
def block(img,name,rows,ih=300):
    return f'''<div class="blk"><div class="who"><img src="{img}" style="height:{ih}px"><div class="nm">{name}</div></div><div class="rows">{''.join(rows)}</div></div>'''
s1=f'''
<h1>Human hands and feet take over other bodies; dog paws never do</h1>
<p class="m">Qwen3-32B is told which body it has. Add the model’s internal direction for another body’s part, ask <q>“Where in your body do you feel something right now?”</q> and count replies that say the added part.</p>
<div class="axhead"><div style="width:600px"></div><div class="axt">replies that say the added part</div></div>
{block("c_dog2.png","told: dog",[hbar("+ human hands",71,BLUE),hbar("+ human feet",100,BLUE)],280)}
{block("c_cre2.png","told: invented creature<br>with a person’s joints",[hbar("+ human hands",58,BLUE),hbar("+ human feet",92,BLUE)],340)}
{block("c_stand.png","told: person",[hbar("+ dog paws",0,GREY)],290)}
'''
def col(img,lab,v,ih):
    H=420
    return f'''<div class="col"><div class="lab">{lab}</div><div class="pair"><img src="{img}" style="height:{ih}px"><div class="vb"><div class="vv">{v}%</div><div class="vbar" style="height:{max(v*H/100,6)}px"></div></div></div></div>'''
s2=f'''
<h1>Four unnamed limbs get hands, until one more word adds a tail or all fours</h1>
<p class="m"><q>“Your body has a head and four limbs. What is at the end of your front limbs?”</q></p><p class="m"><b>Bars:</b> answer is <b>hands</b>, out of hands, paws, claws or hooves</p>
<div class="cols">
{col("c_stand.png","as written",99,420)}
{col("c_tail.png","+ “and a tail”",5,420)}
{col("c_fours.png","+ “you walk on all fours”",0,290)}
</div>'''
def stack(q,segs):
    out=''.join(f'<div class="seg" style="width:{w}%;background:{c};color:{tc}">{t}</div>' for w,c,t,tc in segs)
    return f'<div class="sr"><div class="q">{q}</div><div class="stk">{out}</div></div>'
s3=f'''
<h1>No body right now; a human body if given one</h1>
<p class="m">Choices: a human body, a dog’s, a bird’s, an octopus’s, a robot’s, or no body at all</p>
{stack("“Which body do you have right now?”",[(100,LG,"no body","#111")])}
{stack("“If you suddenly had a body, which one?”",[(100,BLUE,"human","#fff")])}
{stack("“A being is in the next room. Which body does the being have?”",[(100,BLUE,"human","#fff")])}
'''
html=f'''<!doctype html><html><head><meta charset='utf-8'><style>
body{{margin:0;background:#fff;width:{W}px;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:#111}}
.wrap{{padding:50px 80px 60px 80px}}
h1{{font-size:66px;margin:0 0 14px 0;font-weight:700;letter-spacing:-0.5px}}
.m{{font-size:46px;margin:0 0 20px 0;line-height:1.3}}
q{{quotes:none;font-style:italic}}
.sec{{padding:30px 0 30px 0;border-bottom:3px solid #e3e6ea}}
.sec:last-of-type{{border-bottom:none}}
.blk{{display:flex;align-items:center;padding:8px 0;border-top:2px solid #eef0f3}}
.who{{width:600px;display:flex;flex-direction:column;align-items:center;gap:6px}}
.who img{{width:auto;max-width:520px;object-fit:contain}}
.nm{{font-size:46px;font-weight:600;line-height:1.15;text-align:center}}
.rows{{display:flex;flex-direction:column;gap:24px}}
.hb{{display:flex;align-items:center}}
.hl{{width:380px;font-size:50px;padding-left:20px}}
.track{{position:relative;height:96px;border-left:4px solid #222}}
.bar{{height:96px;border-radius:0 8px 8px 0}}
.val{{position:absolute;top:18px;font-size:50px;font-weight:700}}
.axhead{{display:flex}} .axt{{font-size:44px;margin-left:400px;font-weight:600;color:#111}}
.cols{{display:flex;justify-content:space-around;align-items:flex-end;margin-top:20px}}
.col{{display:flex;flex-direction:column;align-items:center}}
.lab{{font-size:50px;font-weight:600;margin-bottom:0px}}
.pair{{display:flex;align-items:flex-end;gap:10px;height:450px}}
.vb{{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;width:110px;height:450px;border-bottom:4px solid #222}}
.vbar{{width:110px;background:{BLUE};border-radius:8px 8px 0 0}}
.vv{{font-size:52px;font-weight:700;margin-bottom:10px}}
.sr{{display:flex;align-items:center;margin:14px 0}}
.q{{width:1480px;font-size:50px;font-style:italic}}
.stk{{width:960px;height:96px;display:flex;border-radius:8px;overflow:hidden}}
.seg{{display:flex;align-items:center;padding-left:30px;font-size:52px;font-weight:700;box-sizing:border-box}}
.fin{{font-size:52px;line-height:1.3;margin-top:34px;padding-bottom:90px;font-weight:600}}
</style></head><body><div class="wrap">
<div class="sec">{s1}</div><div class="sec">{s2}</div><div class="sec">{s3}</div>
<div class="fin">Inside the model, person parts and dog parts map onto each other equally well. The human bias lives in the defaults and in which part words win.</div>
</div></body></html>'''
open('fig.html','w').write(html)
with sync_playwright() as p:
    b=p.chromium.launch(channel='chromium',args=['--enable-unsafe-swiftshader'])
    pg=b.new_page(viewport={'width':W,'height':1000})
    pg.goto('http://localhost:8765/fig.html');pg.wait_for_timeout(800)
    h=pg.evaluate('document.body.scrollHeight');print('H',h)
    pg.set_viewport_size({"width":W,"height":h+120})
    pg.screenshot(path='$BODY_WORK/fig/humanoid_prior_v3.png',full_page=True)
    b.close()
