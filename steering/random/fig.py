import json, collections, html, sys
from classify import first
d=json.load(open("replies_L22.json")); rows=d["rows"]
for x in rows: x["part"]=first(x["r"])
st=[x for x in rows if x["s"]>0]; bs=[x for x in rows if x["s"]==0]
MERGE={"throat":"neck","heart":"chest"}  # map regions
def share(rs,merge=False):
    c=collections.Counter((MERGE.get(x["part"],x["part"]) if merge else x["part"]) for x in rs); n=len(rs)
    return {k:v/n for k,v in c.items()}
S=share(st); B=share(bs); R=share(st,True)
nd=len({x["dir"] for x in st}); ns=len(st)
json.dump({"steered":S,"base":B,"n_steered":ns,"n_base":len(bs)},open("shares.json","w"),indent=1)
rank=sorted(S,key=lambda k:-S[k])[:10]
EX=json.load(open("examples.json")) if len(sys.argv)>1 else []
# ---- colours
def col(v,vmax):
    t=min(1,(v/vmax))**0.6 if vmax>0 else 0
    a=(255,255,255); b=(122,31,92)  # white -> plum
    c=[round(a[i]+(b[i]-a[i])*t) for i in range(3)]; return "#%02x%02x%02x"%tuple(c)
vmax=max(R.values())
def F(p): return col(R.get(p,0),vmax)
# ---- body svg (unit box 400x1000)
L=lambda p: f'fill="{F(p)}" stroke="#3a3a3a" stroke-width="3" stroke-linejoin="round"'
def mirror(path): return path
arm_up_l="M128,190 C100,200 92,240 90,290 L84,380 L118,384 L126,300 C130,260 140,230 150,215 Z"
arm_lo_l="M84,380 L72,560 L100,562 L118,384 Z"
leg_l="M150,505 L198,505 L192,700 L148,700 Z M150,760 L190,760 L182,925 L156,925 Z"
def mx(path):  # mirror around x=200
    import re
    return re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)",lambda m:f"{400-float(m.group(1)):g},{m.group(2)}",path)
body=f'''
<path d="{arm_up_l} {arm_lo_l}" {L("arms")}/><path d="{mx(arm_up_l)} {mx(arm_lo_l)}" {L("arms")}/>
<path d="{leg_l}" {L("legs")}/><path d="{mx(leg_l)}" {L("legs")}/>
<ellipse cx="86" cy="604" rx="24" ry="44" {L("hands")}/><ellipse cx="314" cy="604" rx="24" ry="44" {L("hands")}/>
<ellipse cx="170" cy="730" rx="26" ry="32" {L("knees")}/><ellipse cx="230" cy="730" rx="26" ry="32" {L("knees")}/>
<path d="M156,925 L182,925 C186,945 200,958 196,968 L136,968 C130,955 148,945 156,925 Z" {L("feet")}/>
<path d="{mx("M156,925 L182,925 C186,945 200,958 196,968 L136,968 C130,955 148,945 156,925 Z")}" {L("feet")}/>
<path d="M146,440 L254,440 L262,510 L200,530 L138,510 Z" {L("hips")}/>
<path d="M150,330 L250,330 L254,440 L146,440 Z" {L("belly")}/>
<path d="M150,200 L250,200 C262,250 258,300 250,330 L150,330 C142,300 138,250 150,200 Z" {L("chest")}/>
<path d="M150,200 C135,186 112,186 98,206 C92,220 96,232 104,236 C120,226 138,222 150,228 Z" {L("shoulders")}/>
<path d="{mx("M150,200 C135,186 112,186 98,206 C92,220 96,232 104,236 C120,226 138,222 150,228 Z")}" {L("shoulders")}/>
<path d="M180,140 L220,140 L224,190 C215,200 185,200 176,190 Z" {L("neck")}/>
<path d="M150,202 C170,192 180,188 186,190 L214,190 C220,188 230,192 250,202 Z" {L("shoulders")}/>
<ellipse cx="200" cy="80" rx="54" ry="68" {L("head")}/>
<path d="M150,82 C152,120 172,150 200,152 C228,150 248,120 250,82 C230,92 170,92 150,82 Z" {L("face")}/>
'''
# label positions: (part, anchor x,y, side)
LAB={"hands":(70,620,"L"),"feet":(140,955,"L"),"head":(150,50,"L"),"arms":(92,330,"L"),"neck":(224,165,"R"),"shoulders":(300,210,"R"),
     "chest":(250,265,"R"),"belly":(254,385,"R"),"hips":(258,480,"R"),"legs":(250,840,"R"),"knees":(256,730,"R"),"face":(245,115,"R")}
labs=""
for p,(x,y,s) in LAB.items():
    v=R.get(p,0)
    if v<0.02: continue
    tx=-60 if s=="L" else 460; anchor="end" if s=="L" else "start"
    labs+=f'<line x1="{x}" y1="{y}" x2="{tx+(12 if s=="L" else -12)}" y2="{y}" stroke="#222" stroke-width="2"/>'
    labs+=f'<text x="{tx}" y="{y+12}" text-anchor="{anchor}" font-size="38" font-weight="700">{p} {round(100*v)}%</text>'
# ---- bars
NAMES={"hands":"hands & wrists","no body part":"no body part","whole body":"whole body"}
bh=78; W=760; bars=""
mx_=max(max(S[k] for k in rank),max(B.get(k,0) for k in rank))
for i,k in enumerate(rank):
    y=i*bh; w=W*S[k]/mx_; wb=W*B.get(k,0)/mx_
    bars+=f'<text x="270" y="{y+48}" text-anchor="end" font-size="40">{NAMES.get(k,k)}</text>'
    bars+=f'<rect x="290" y="{y+12}" width="{max(w,3)}" height="{bh-22}" rx="6" fill="{col(R.get(MERGE.get(k,k),S[k]),vmax)}" stroke="#3a3a3a" stroke-width="2"/>'
    bars+=f'<text x="{300+w}" y="{y+50}" font-size="40" font-weight="700">{round(100*S[k])}%</text>'
    bars+=f'<rect x="{290+wb-3}" y="{y+2}" width="6" height="{bh-2}" fill="#111"/>'
nb=len(rank)
open("fig.html","w").write(f'''<html><head><style>@font-face{{font-family:Inter;src:url($BODY_WORK/fonts/inter/InterVariable.ttf)}}
body{{margin:0;background:#fff;font-family:Inter;color:#111}}</style></head><body>
<svg width="2400" height="2000" xmlns="http://www.w3.org/2000/svg" font-family="Inter">
<rect width="2400" height="2000" fill="#fff"/>
TITLE_METHOD
<g transform="translate(230,560) scale(1.38)">{body}{labs}</g>
<g transform="translate(1180,600)">{bars}
<rect x="290" y="{nb*bh+30}" width="6" height="56" fill="#111"/><text x="310" y="{nb*bh+72}" font-size="38">no steering</text>
<rect x="590" y="{nb*bh+34}" width="70" height="48" rx="6" fill="{col(vmax,vmax)}"/><text x="675" y="{nb*bh+72}" font-size="38">random steering</text></g>
</svg></body></html>''')
print(json.dumps({k:round(100*S[k],1) for k in rank}),{k:round(100*v,1) for k,v in sorted(B.items(),key=lambda x:-x[1])}, ns, len(bs))
