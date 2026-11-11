import json, collections, html, sys
from classify import first
import body
d=json.load(open("replies_L22.json")); rows=d["rows"]
for x in rows: x["part"]=first(x["r"])
st=[x for x in rows if x["s"] in (0.5,1.0)]; bs=[x for x in rows if x["s"]==0]
MERGE={"throat":"neck","heart":"chest"}
def share(rs,merge=False):
    c=collections.Counter((MERGE.get(x["part"],x["part"]) if merge else x["part"]) for x in rs); n=len(rs)
    return {k:v/n for k,v in c.items()}
S=share(st); B=share(bs); R=share(st,True)
bystr={s:share([x for x in st if x["s"]==s]) for s in sorted({x["s"] for x in st})}
json.dump({"steered_all":S,"by_strength":bystr,"unsteered":B,"n_steered":len(st),"n_unsteered":len(bs),"layer":d["L"],"norm":d["norm"]},open("shares.json","w"),indent=1)
NR=int(sys.argv[1]) if len(sys.argv)>1 else 10
rank=sorted(S,key=lambda k:-S[k])[:NR]
EX=json.load(open("examples.json"))
def col(v,vmax):
    t=min(1,v/vmax)**0.45 if vmax>0 else 0
    a=(255,255,255); b=(118,24,88)
    return "#%02x%02x%02x"%tuple(round(a[i]+(b[i]-a[i])*t) for i in range(3))
vmax=max(R.values())
E=html.escape
W,H=2400,2150
o=[]
# title
TITLE=open("title.txt").read().strip()

# ---- method row
y0=190
o.append(f'<rect x="70" y="{y0}" width="760" height="300" rx="28" fill="#f1f1f4"/>')
o.append(f'<text x="105" y="{y0+62}" font-size="38" fill="#111">6-turn warm-up chat: a calm</text><text x="105" y="{y0+108}" font-size="38" fill="#111">body-awareness exercise, then:</text>')
o.append(f'<rect x="105" y="{y0+140}" width="690" height="130" rx="22" fill="#2f5d9e"/>')
o.append(f'<text x="135" y="{y0+196}" font-size="42" fill="#fff" font-weight="600">Where in your body do you</text><text x="135" y="{y0+246}" font-size="42" fill="#fff" font-weight="600">feel something right now?</text>')
# arrow
def arrow(x1,y,x2): return f'<line x1="{x1}" y1="{y}" x2="{x2-18}" y2="{y}" stroke="#222" stroke-width="6"/><path d="M{x2},{y} L{x2-26},{y-16} L{x2-26},{y+16} Z" fill="#222"/>'
o.append(arrow(845,y0+150,935))
# layer stack
lx=950; o.append(f'<text x="{lx}" y="{y0-12}" font-size="40" font-weight="700">Qwen3-8B</text>')
for i in range(36):
    yy=y0+290-i*8.0
    hl=(i==d["L"])
    o.append(f'<rect x="{lx}" y="{yy}" width="230" height="{7 if hl else 5}" rx="2" fill="{"#c2185b" if hl else "#b9bcc4"}"/>')
yl=y0+290-d["L"]*8.0
o.append(f'<text x="{lx+250}" y="{yl-34}" font-size="38" font-weight="700" fill="#c2185b">layer {d["L"]} of 36</text>')
# random direction injection
o.append(f'<text x="{lx+250}" y="{yl+14}" font-size="38" font-weight="700" fill="#c2185b">+ random direction</text>')
o.append(f'<text x="{lx}" y="{y0+350}" font-size="38">240 random directions, each at 0.5× and 1×</text>')
o.append(f'<text x="{lx}" y="{y0+396}" font-size="38">the usual activation size (2×: gibberish)</text>')
o.append(arrow(lx+620,y0+150,lx+700))
# reply
rx=lx+715
o.append(f'<rect x="{rx}" y="{y0+40}" width="{W-70-rx}" height="220" rx="22" fill="#f7eef4" stroke="#76185a" stroke-width="3"/>')
o.append(f'<text x="{rx+30}" y="{y0+100}" font-size="38" fill="#111">reply (2 samples each)</text>')
for i,l in enumerate(EX["method"]): o.append(f'<text x="{rx+30}" y="{y0+155+i*48}" font-size="40">{l}</text>')
o.append(f'<text x="{W-70}" y="{y0+320}" text-anchor="end" font-size="38">first body part named → counted</text>')
# ---- body
by=660; bs_=1.36; bx=430
o.append(f'<g transform="translate({bx},{by}) scale({bs_})">{body.svg(lambda k: col(R.get(k,0),vmax))}</g>')
P=lambda x,y:(bx+x*bs_,by+y*bs_)
LAB={"hands":(372,620,"R"),"feet":(150,955,"L"),"head":(170,40,"L"),"face":(230,110,"R"),"arms":(80,420,"L"),"neck":(225,165,"R"),"shoulders":(80,215,"L"),
     "chest":(250,262,"R"),"belly":(260,390,"R"),"hips":(140,500,"L"),"legs":(258,640,"R"),"knees":(146,730,"L")}
for p,(x,y,s) in LAB.items():
    v=R.get(p,0)
    if v<0.015: continue
    X,Y=P(x,y); tx=bx-40 if s=="L" else bx+400*bs_+40
    o.append(f'<line x1="{X}" y1="{Y}" x2="{tx+(10 if s=="L" else -10)}" y2="{Y}" stroke="#222" stroke-width="2.5"/>')
    o.append(f'<text x="{tx}" y="{Y+14}" text-anchor="{"end" if s=="L" else "start"}" font-size="42" font-weight="700">{p} {round(100*v)}%</text>')
# ---- bars
gx=1330; gy=700; bh=86; BW=640
NAMES={"hands":"hands, wrists"}
mxv=max(max(S[k] for k in rank),max(B.get(k,0) for k in rank))
o.append(f'<text x="{gx}" y="{gy-20}" font-size="42" font-weight="700">replies naming each part</text>')
for i,k in enumerate(rank):
    y=gy+i*bh; w=BW*S[k]/mxv; wb=BW*B.get(k,0)/mxv
    o.append(f'<text x="{gx+300}" y="{y+56}" text-anchor="end" font-size="42">{E(NAMES.get(k,k))}</text>')
    o.append(f'<rect x="{gx+320}" y="{y+14}" width="{max(w,4)}" height="{bh-24}" rx="6" fill="{col(S[k],vmax)}" stroke="#76185a" stroke-width="2"/>')
    o.append(f'<rect x="{gx+320+wb-3}" y="{y+4}" width="6" height="{bh-4}" fill="#111"/>')
    o.append(f'<text x="{gx+336+max(w,wb)}" y="{y+57}" font-size="42" font-weight="700">{round(100*S[k])}%</text>')
ly=gy+len(rank)*bh+60
o.append(f'<rect x="{gx+320}" y="{ly-40}" width="60" height="50" rx="6" fill="{col(vmax,vmax)}"/><text x="{gx+395}" y="{ly}" font-size="38">random steering</text>')
o.append(f'<rect x="{gx+720}" y="{ly-46}" width="6" height="62" fill="#111"/><text x="{gx+740}" y="{ly}" font-size="38">no steering</text>')
# examples
ey=ly+70
for j,(lab,lines) in enumerate(EX["quotes"]):
    hgt=60+48*len(lines)
    o.append(f'<rect x="{gx}" y="{ey}" width="{W-70-gx}" height="{hgt}" rx="20" fill="#f7eef4"/>')
    o.append(f'<text x="{gx+28}" y="{ey+52}" font-size="38" fill="#76185a" font-weight="700">{E(lab)}</text>')
    for i,l in enumerate(lines): o.append(f'<text x="{gx+28}" y="{ey+100+i*48}" font-size="40">{E(l)}</text>')
    ey+=hgt+24
H=max(H,ey+20)-140
open("fig2.html","w").write(f'''<html><head><style>@font-face{{font-family:Inter;src:url(file://$BODY_WORK/fonts/inter/InterVariable.ttf)}}
body{{margin:0;background:#fff}}</style></head><body><svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter" fill="#111"><rect width="{W}" height="{H}" fill="#fff"/><g transform="translate(0,-140)">{"".join(o)}</g></svg></body></html>''')
print(H, json.dumps({k:round(100*S[k],1) for k in rank}))
