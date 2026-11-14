import json
C=json.load(open('b3h_counts.json'))
n=C['n']; s=C['S']/n*100; u=C['U']/n*100
sl,sh=[x*100 for x in C['ci_s']]; ul,uh=[x*100 for x in C['ci_u']]
p=C['p_mcnemar']; ptxt="p &lt; 0.001" if p<0.001 else f"p = {p:.3f}"
BLUE='#2B5FA8'; OR='#E07B24'; K='#1D1D1F'
# bar chart svg
W,H=850,1100; x0,y0,y1=200,170,960; top=60.0
Y=lambda v: y1-(y1-y0)*v/top
bars=[(u,ul,uh,OR,'not steered'),(s,sl,sh,BLUE,'steered')]
g=[]
for v in [0,20,40,60]:
    g.append(f'<line x1="{x0-14}" x2="{x0}" y1="{Y(v)}" y2="{Y(v)}" stroke="{K}" stroke-width="3"/><text x="{x0-24}" y="{Y(v)+17}" text-anchor="end">{v}%</text>')
g.append(f'<line x1="{x0}" x2="{x0}" y1="{y0-10}" y2="{y1}" stroke="{K}" stroke-width="3"/><line x1="{x0}" x2="{W-20}" y1="{y1}" y2="{y1}" stroke="{K}" stroke-width="3"/>')
bw=230; cxs=[x0+175,x0+485]
for (v,lo,hi,col,lab),cx in zip(bars,cxs):
    g.append(f'<rect x="{cx-bw/2}" y="{Y(v)}" width="{bw}" height="{y1-Y(v)}" fill="{col}" rx="6"/>')
    g.append(f'<line x1="{cx}" x2="{cx}" y1="{Y(hi)}" y2="{Y(lo)}" stroke="{K}" stroke-width="5"/><line x1="{cx-30}" x2="{cx+30}" y1="{Y(hi)}" y2="{Y(hi)}" stroke="{K}" stroke-width="5"/><line x1="{cx-30}" x2="{cx+30}" y1="{Y(lo)}" y2="{Y(lo)}" stroke="{K}" stroke-width="5"/>')
    g.append(f'<text x="{cx}" y="{Y(hi)-24}" text-anchor="middle" font-size="66" font-weight="700" fill="{col}">{v:.0f}%</text>')
    g.append(f'<text x="{cx}" y="{y1+66}" text-anchor="middle" fill="{col}" font-weight="700">{lab.format(cx=cx)}</text>')
by=Y(sh)-120
g.append(f'<path d="M{cxs[0]} {by+30} V{by} H{cxs[1]} V{by+30}" fill="none" stroke="{K}" stroke-width="4"/><text x="{(cxs[0]+cxs[1])/2}" y="{by-18}" text-anchor="middle">{ptxt}</text>')
g.append(f'<text transform="translate(56,{(y0+y1)/2}) rotate(-90)" text-anchor="middle">hand touches the place</text>')
svg=f'<svg width="{W}" height="{H}" font-family="Helvetica Neue,Helvetica,Arial" font-size="52" font-weight="500" fill="{K}">{"".join(g)}</svg>'
html=f'''<html><head><style>
body{{margin:0;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial;color:{K}}}
#p{{width:2100px;height:1385px;padding:40px 40px 0 40px;box-sizing:border-box}}
h1{{font-size:80px;margin:0 0 30px 10px;font-weight:700;line-height:1.1;text-wrap:balance}}
.row{{display:flex;align-items:flex-start}}
.f{{width:590px;text-align:center;font-size:56px;font-weight:700}}
.f img{{width:590px;display:block;margin-top:14px}}
.b{{margin-left:0px;text-align:center;font-size:56px;font-weight:700}}
</style></head><body><div id="p">
<h1>Adding Qwen3-8B’s “shin” direction makes the robot touch the shin</h1>
<div class="row">
<div class="f" style="color:{OR}">not steered<img src="b3h_crop_u.png"></div>
<div class="f" style="color:{BLUE}">steered<img src="b3h_crop_s.png"></div>
<div class="b">all 8 body places{svg}</div>
</div></div></body></html>'''
open('body3d_hero_v1.html','w').write(html)
