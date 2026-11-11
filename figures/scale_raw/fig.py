import os
import numpy as np, math, sys
exec(open(os.path.expandvars('$BODY_REPO/figures/scale_raw/drive.py')).read())
F="Helvetica Neue, Helvetica, Arial, sans-serif"
d=np.load("../src/nn2.npz"); SP=np.load("spot_pos.npy"); VS=np.load("vis.npz"); n=len(SP)
R5=np.load("../nb4/rand5.npy")
out=[]
def T(x,y,s,size=50,w=400,anchor="start",fill="#111",extra=""):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}" {extra}>{s}</text>')
def arrow(x1,y1,x2,y2,w=7):
    L=math.hypot(x2-x1,y2-y1); ux,uy=(x2-x1)/L,(y2-y1)/L; hx,hy=x2-ux*28,y2-uy*28
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="#111" stroke-width="{w}"/>')
    out.append(f'<polygon points="{x2},{y2} {hx-uy*17:.0f},{hy+ux*17:.0f} {hx+uy*17:.0f},{hy-ux*17:.0f}" fill="#111"/>')
STOPS=[(0,(40,8,80)),(10,(112,38,150)),(20,(196,62,150)),(30,(240,128,120)),(40,(250,214,80))]
def col(v):
    v=min(max(v,0),40)
    for (a,ca),(b,cb) in zip(STOPS,STOPS[1:]):
        if v<=b: t=(v-a)/(b-a); return "#%02x%02x%02x"%tuple(round(ca[k]+t*(cb[k]-ca[k])) for k in range(3))
W=2600; views=[]
def vp_view(front,x,y,w,h,yc,halfh,xc=0):
    s=1 if front else -1
    views.append(dict(col="plain",vp=[x,y,w,h],halfh=halfh,eye=[xc,yc,400*s],tgt=[xc,yc,0],light=[-120*s,300,300*s]))
    sc=h/(2*halfh); return lambda p: (x+w/2+s*(p[0]-xc)*sc, y+h/2-(p[1]-yc)*sc)
# ---------------- method strip
EX=38
import sys as _s; _s.path.insert(0,os.path.expandvars("$BODY_REPO/hires/v2")); import stim as _S
_p2=[l["phrase"] for l in _S.LOCS]; _k=_p2.index(str(d['ph'][EX]))
a=np.load(os.path.expandvars("$BODY_REPO/hires/v2/red32_short.npz"))["reply|53|all|A"][_k].astype(np.float32); C32=a/(np.abs(a).max()+1e-9)
def dcol(v):
    v=float(np.clip(v*2.2,-1,1)); c=(178,24,43) if v>0 else (33,102,172); t=abs(v)**0.7
    return "#%02x%02x%02x"%tuple(round(255+t*(cc-255)) for cc in c)
for k in (2,1): out.append(f'<rect x="{30+k*14}" y="{30+k*14}" width="680" height="275" rx="22" fill="#fff" stroke="#9aa3ad" stroke-width="3"/>')
out.append('<rect x="30" y="30" width="680" height="275" rx="22" fill="#fff" stroke="#111" stroke-width="4"/>')
T(60,100,"A feather brushes slowly",50); T(60,160,"across the inside of your",50); T(60,220,"left forearm, a hand-width",50,700); T(60,280,"above the wrist.",50,700)
T(30,400,"× 26 touch sentences",50)
arrow(745,155,815,155)
out.append('<rect x="830" y="55" width="380" height="190" rx="24" fill="#1d2b3a"/>')
T(1020,135,"Qwen3-8B",56,700,"middle","#fff"); T(1020,207,"Qwen3-32B",56,700,"middle","#fff")
cw=12.6
for c in range(30): out.append(f'<rect x="{830+c*cw:.1f}" y="275" width="{cw-2:.1f}" height="50" fill="{dcol(C32[c])}"/>')
out.append(f'<rect x="830" y="275" width="{30*cw-2:.1f}" height="50" fill="none" stroke="#111" stroke-width="2"/>')
T(830,400,"layer state",50)
arrow(1235,155,1305,155)
proj=vp_view(1,1310,10,600,330,121,34)
nn=d["nn_32B"][EX]; tw=d["tw"][EX]
for j in nn: x,y=proj(SP[j]); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="none" stroke="#7a2ca0" stroke-width="6.5"/>')
xt,yt=proj(SP[tw]); out.append(f'<path d="M{xt-12},{yt-12} L{xt+12},{yt+12} M{xt-12},{yt+12} L{xt+12},{yt-12}" stroke="#111" stroke-width="6"/>')
xe,ye=proj(SP[EX]); out.append(f'<circle cx="{xe:.1f}" cy="{ye:.1f}" r="12" fill="#111"/>')
T(1610,400,"5 most similar of 407 spots",50,400,"middle")
arrow(1925,155,1995,155)
T(2010,90,"distance on",50,700); T(2010,150,"the body:",50,700)
T(2010,220,"9, 9, 5, 5, 4 cm",50)
T(2010,290,"→ average 6 cm",50,700)
T(2010,400,"× other side: skipped",50)
# ---------------- swarm
SY=450
T(30,SY+50,"Most similar states sit close on the body, closest in the bigger model",60,700)
X0,X1=600,2520; LO,HI=math.log10(2),math.log10(180)
XA=lambda v: X0+(X1-X0)*(math.log10(v)-LO)/(HI-LO)
rows=[("Qwen3-32B",d["d_32B"],"#1f64b4"),("Qwen3-8B",d["d_8B"],"#e8590c"),("word vectors",d["d_words"],"#8a8a8a"),("5 random spots",R5,"#c4c4c4")]
sub=[None,None,"(input layer)",None]
r=5.6; RH=280; top=SY+110
def swarm(xs,r):
    order=np.argsort(xs); placed=[]; ys=np.zeros(len(xs))
    for i in order:
        x=xs[i]; near=[(px,py) for px,py in placed if abs(px-x)<2*r]
        cand=[0]+[s*k*0.5*r for k in range(1,400) for s in (1,-1)]
        for y in cand:
            if all((x-px)**2+(y-py)**2>=(2*r)**2*0.98 for px,py in near): break
        ys[i]=y; placed.append((x,y))
    return ys
meds={}
for k,(name,v,c) in enumerate(rows):
    cy=top+k*RH+RH/2; xs=np.array([XA(max(a,2)) for a in v]); ys=swarm(xs,r)
    s=min(1,(RH/2-14)/max(np.abs(ys).max(),1)); print(name,"max |y|",np.abs(ys).max(),"squash",s)
    for x,y in zip(xs,ys): out.append(f'<circle cx="{x:.1f}" cy="{cy+y*s:.1f}" r="{r}" fill="{c}" stroke="#fff" stroke-width="1.2"/>')
    m=float(np.median(v)); meds[name]=m; xm=XA(m)
    out.append(f'<line x1="{xm:.1f}" y1="{cy-RH/2+12}" x2="{xm:.1f}" y2="{cy+RH/2-12}" stroke="#111" stroke-width="9" stroke-linecap="round"/>')
    out.append(f'<line x1="{xm:.1f}" y1="{cy-RH/2+12}" x2="{xm:.1f}" y2="{cy+RH/2-12}" stroke="{c}" stroke-width="0"/>')
    T(xm+16,cy-RH/2+44,f"{m:.0f} cm",52,700,extra='stroke="#fff" stroke-width="12" paint-order="stroke" stroke-linejoin="round"')
    tc=c if k<2 else "#111"
    T(X0-40,cy+(18 if sub[k] is None else -8),name,54,700,"end",tc)
    if sub[k]: T(X0-40,cy+52,sub[k],48,400,"end")
ya=top+4*RH+10
out.append(f'<line x1="{X0}" y1="{ya}" x2="{X1}" y2="{ya}" stroke="#111" stroke-width="3"/>')
for t in (2,5,10,20,50,100):
    out.append(f'<line x1="{XA(t):.1f}" y1="{ya}" x2="{XA(t):.1f}" y2="{ya+16}" stroke="#111" stroke-width="3"/>')
    T(XA(t),ya+66,f"{t} cm",48,400,"middle")
T(X0,ya+140,"← closer",50,700); T(X0+260,ya+140,"each dot: one of 407 spots, average body distance to its 5 most similar spots",50)
# ---------------- bodies
BY=ya+200; bw,bh=610,800; halfh=bh/2/4.42; yc=89.5
pairs=[("32B","Qwen3-32B","#1f64b4",d["d_32B"]),("8B","Qwen3-8B","#e8590c",d["d_8B"])]
nz=VS["nz"]; fv,bv=VS["front"],VS["back"]
front=np.where(fv&~bv,True,np.where(bv&~fv,False,nz>=0))
for p,(Z,name,c,v) in enumerate(pairs):
    x0=30+p*1290
    T(x0+bw,BY+40,name,60,700,"middle",c)
    for q,fr in enumerate((1,0)):
        pr=vp_view(fr,x0+q*bw,BY+60,bw,bh,yc,halfh)
        idx=np.where(front==bool(fr))[0]
        idx=idx[np.argsort(-v[idx])]  # tight spots drawn on top
        for i in idx:
            x,y=pr(SP[i]); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.8" fill="{col(v[i])}" stroke="#333" stroke-width="1.2"/>')
        T(x0+q*bw+bw/2,BY+60+bh+50,"front" if fr else "back",50,400,"middle")
# color bar
cbY=BY+60+bh+110; gx0,gx1=120,1300
out.append('<defs><linearGradient id="cs">'+"".join(f'<stop offset="{a/40:.3f}" stop-color="#%02x%02x%02x"/>'%cc for a,cc in STOPS)+'</linearGradient></defs>')
out.append(f'<rect x="{gx0}" y="{cbY}" width="{gx1-gx0}" height="44" fill="url(#cs)"/>')
for t in (0,10,20,30,40):
    T(gx0+(gx1-gx0)*t/40,cbY+100,f"{t}"+("+ cm" if t==40 else ""),48,400,"middle")
T(gx1+40,cbY+38,"dot color: the same distance, per spot",50,700)
H=cbY+130
print("H",H, {k:round(m,1) for k,m in meds.items()})
shoot(dict(W=W,H=H,dpr=2,views=views),HERE+"/bodies.png")
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><image href="bodies.png" x="0" y="0" width="{W}" height="{H}"/>'+"".join(out)+"</svg>"
open(HERE+"/fig.html","w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True); pg=br.new_context(viewport={"width":W,"height":H},device_scale_factor=2).new_page()
    pg.goto("file://"+HERE+"/fig.html"); pg.wait_for_timeout(500); pg.screenshot(path="/tmp/sr_2x.png"); br.close()
Image.open("/tmp/sr_2x.png").convert("RGB").resize((W,H),Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_scale_raw.png"))
