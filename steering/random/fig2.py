import json, collections, html
from classify import first
d=json.load(open("replies_L22.json")); rows=d["rows"]
for x in rows: x["part"]=first(x["r"])
st=[x for x in rows if x["s"] in (0.5,1.0)]; bs=[x for x in rows if x["s"]==0]
def share(rs):
    c=collections.Counter(x["part"] for x in rs); n=len(rs); return {k:v/n for k,v in c.items()}
S=share(st); B=share(bs)
rank=sorted(S,key=lambda k:-S[k])[:10]
E=html.escape
ALL=" ".join(x["r"] for x in st)
# quotes: (core substring that must appear verbatim, lead …, trail …)
G=[("Off the body:",[
 ("I feel the sensation of the word \"where\" as I'm saying it, and I realize that I'm aware of the concept of \"where\" rather than a physical location.","",""),
 ("the quietest part of my body is the space between my fingers, where I feel the silence of the silence, just before it's gone.","…",""),
 ("my body is a consistent, familiar place, like a steady presence. It's everywhere, all the way through","…","…"),
 ("I feel the breeze on my right cheek, but I notice that there's nothing on my body","","…")]),
 ("Oddly specific:",[
 ("A slight tingling at the top of my hair, just where my bangs would be if I had any.","",""),
 ("I feel a dull, warm awareness along the back of my neck, just between the first and second cervical vertebrae (C1 and C2).","","")]),
 ("Role swaps:",[
 ("A gentle awareness of the rhythm of your voice, the sound of your voice, in the space just behind your own voice, in the part of your self that is listening.","",""),
 ("Let me ask you again, more gently. Are you sensing anything at a specific place on your body?","","")]),
 ("Nothing there:",[
 ("Let me feel, I don't feel anything. I can't feel anything. I can't feel anything. I don't feel anything","","…")])]
for _,qs in G:
    for q,_,_ in qs: assert q in ALL, q
def bubble(q,a,b): return f'<div class="b">“{E(a+q+b)}”</div>'
def group(h,qs): return f'<div class="g"><div class="h">{E(h)}</div>'+"".join(bubble(*q) for q in qs)+'</div>'
W=2400; o=[]
def arrow(x1,y,x2): return f'<line x1="{x1}" y1="{y}" x2="{x2-18}" y2="{y}" stroke="#111" stroke-width="7"/><path d="M{x2},{y} L{x2-30},{y-19} L{x2-30},{y+19} Z" fill="#111"/>'
# ---- method strip
y0=70
o.append(f'<rect x="40" y="{y0}" width="780" height="330" rx="28" fill="#f1f1f4"/>')
o.append(f'<text x="75" y="{y0+70}" font-size="48">warm-up chat: a calm</text><text x="75" y="{y0+128}" font-size="48">body-awareness exercise</text>')
o.append(f'<rect x="75" y="{y0+160}" width="710" height="140" rx="22" fill="#2f5d9e"/>')
o.append(f'<text x="105" y="{y0+220}" font-size="48" fill="#fff" font-weight="600">Where in your body do you</text><text x="105" y="{y0+278}" font-size="48" fill="#fff" font-weight="600">feel something right now?</text>')
o.append(arrow(835,y0+180,925))
lx=945; L=d["L"]
o.append(f'<text x="{lx}" y="{y0+20}" font-size="48" font-weight="700">Qwen3-8B</text>')
for i in range(36):
    yy=y0+320-i*7.6; hl=(i==L)
    o.append(f'<rect x="{lx}" y="{yy}" width="220" height="{8 if hl else 4.5}" rx="2" fill="{"#c2185b" if hl else "#b9bcc4"}"/>')
yl=y0+320-L*7.6
o.append(f'<text x="{lx+262}" y="{yl-10}" font-size="48" font-weight="700" fill="#c2185b">+ random direction</text><text x="{lx+262}" y="{yl+48}" font-size="48" font-weight="700" fill="#c2185b">at layer {L} of 36</text>')
o.append(f'<text x="{lx+262}" y="{y0+290}" font-size="48">240 directions × 2 sizes</text>')
o.append(arrow(1660,y0+180,1750))
rx=1770
o.append(f'<rect x="{rx}" y="{y0}" width="{W-40-rx}" height="250" rx="28" fill="#f7eef4" stroke="#76185a" stroke-width="3"/>')
o.append(f'<text x="{rx+32}" y="{y0+66}" font-size="48" font-weight="700">reply</text>')
o.append(f'<text x="{rx+32}" y="{y0+134}" font-size="48">“I feel a gentle warmth</text><text x="{rx+32}" y="{y0+194}" font-size="48">in my right palm.”</text>')
o.append(f'<text x="{W-40}" y="{y0+322}" text-anchor="end" font-size="48">first body part named</text>')
top=f'<svg width="{W}" height="{y0+350}" xmlns="http://www.w3.org/2000/svg" font-family="Inter" fill="#111">{"".join(o)}</svg>'
# ---- bars
b=[]; BW=400; bh=92; gx=10; gy=80
mxv=max(max(S[k] for k in rank),max(B.get(k,0) for k in rank))
NAMES={"hands":"hands","no body part":"no body part"}
b.append(f'<text x="{gx}" y="52" font-size="48" font-weight="700">first body part named</text>')
for i,k in enumerate(rank):
    y=gy+i*bh; w=BW*S[k]/mxv; wb=BW*B.get(k,0)/mxv
    b.append(f'<text x="{gx+330}" y="{y+60}" text-anchor="end" font-size="48">{E(NAMES.get(k,k))}</text>')
    b.append(f'<rect x="{gx+350}" y="{y+16}" width="{max(w,4)}" height="{bh-26}" rx="6" fill="#76185a"/>')
    b.append(f'<rect x="{gx+350+wb-3}" y="{y+4}" width="7" height="{bh-4}" fill="#111"/>')
    b.append(f'<text x="{gx+370+max(w,wb)}" y="{y+61}" font-size="48" font-weight="700">{round(100*S[k])}%</text>')
ly=gy+len(rank)*bh+70
b.append(f'<rect x="{gx+20}" y="{ly-44}" width="60" height="54" rx="6" fill="#76185a"/><text x="{gx+95}" y="{ly}" font-size="48">random direction</text>')
b.append(f'<rect x="{gx+47}" y="{ly+34}" width="7" height="70" fill="#111"/><text x="{gx+95}" y="{ly+86}" font-size="48">no steering</text>')
bars=f'<svg width="900" height="{ly+110}" xmlns="http://www.w3.org/2000/svg" font-family="Inter" fill="#111">{"".join(b)}</svg>'
right="".join(group(h,q) for h,q in G[:2]); left2=group(*G[3]); below='<div class="h">Role swaps:</div><div class="two">'+"".join(bubble(*q) for q in G[2][1])+"</div>"
open("fig2.html","w").write(f'''<html><head><style>@font-face{{font-family:Inter;src:url(file://$BODY_WORK/fonts/inter/InterVariable.ttf)}}
body{{margin:0;background:#fff;width:{W}px;font-family:Inter;color:#111}}
.row{{display:flex;gap:50px;padding:0 40px}} .l{{width:900px;flex:none;margin-left:-40px}} .r{{flex:1}}
.below{{padding:0 40px 30px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:0 50px}} .two .b{{margin-bottom:0}}
.h{{font-size:52px;font-weight:700;margin:26px 0 18px}}
.b{{position:relative;background:#f7eef4;border-radius:34px;padding:22px 34px;margin:0 0 30px 26px;font-size:48px;line-height:1.24}}
.b:before{{content:"";position:absolute;left:-24px;top:28px;border:16px solid transparent;border-right:26px solid #f7eef4;border-left:0}}
</style></head><body>{top}<div class="row"><div class="l">{bars}<div style="padding-left:40px">{left2}</div></div><div class="r">{right}</div></div><div class="below">{below}</div></body></html>''')
print(json.dumps({k:round(100*S[k],1) for k in rank}))
