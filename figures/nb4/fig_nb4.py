import os
import numpy as np, os, json, sys
exec(open(os.path.expandvars('$BODY_REPO/figures/nb4/drive.py')).read())
HERE=os.path.expandvars("$BODY_REPO/figures/nb4")
F="Helvetica Neue, Helvetica, Arial, sans-serif"
d=np.load(os.path.expandvars("$BODY_REPO/figures/src/nn2.npz")); ph=d['ph']; n=len(ph)
SP=np.load(HERE+"/spot_pos_neutral.npy"); R5=np.load(HERE+"/rand5.npy")
out=[]
def T(x,y,s,size=52,w=400,anchor="start",fill="#111",extra=""):
    out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}" {extra}>{s}</text>')
STOPS=[(0,(8,29,88)),(8,(34,94,168)),(16,(29,145,192)),(24,(65,182,150)),(32,(161,211,90)),(40,(255,214,0))]
def col(v):
    v=min(max(v,0),40)
    for (a,ca),(b,cb) in zip(STOPS,STOPS[1:]):
        if v<=b: t=(v-a)/(b-a); return "#%02x%02x%02x"%tuple(round(ca[k]+t*(cb[k]-ca[k])) for k in range(3))
W=2600
# ---------------- row A: sentence -> model -> internal state (from method3.py, unchanged except wording)
import sys as _s; _s.path.insert(0,os.path.expandvars("$BODY_REPO/hires/v2")); import stim as _S
EX=38
_p2=[l["phrase"] for l in _S.LOCS]; _k=_p2.index(str(ph[EX]))
def _cells(f,key):
    a=np.load(os.path.expandvars("$BODY_REPO/hires/v2/")+f)[key][_k].astype(np.float32); return a/(np.abs(a).max()+1e-9)
C8,C32=_cells("red8b_short.npz","reply|27|all|A"),_cells("red32_short.npz","reply|53|all|A")
def dcol(v):
    v=float(np.clip(v*2.2,-1,1)); a=(178,24,43) if v>0 else (33,102,172); t=abs(v)**0.7
    return "#%02x%02x%02x"%tuple(round(255+t*(c-255)) for c in a)
def arrow(x1,y1,x2,y2,w=7):
    import math; L=math.hypot(x2-x1,y2-y1); ux,uy=(x2-x1)/L,(y2-y1)/L; hx,hy=x2-ux*30,y2-uy*30
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="#111" stroke-width="{w}"/>')
    out.append(f'<polygon points="{x2},{y2} {hx-uy*18:.0f},{hy+ux*18:.0f} {hx+uy*18:.0f},{hy-ux*18:.0f}" fill="#111"/>')
for k in (2,1): out.append(f'<rect x="{40+k*16}" y="{30+k*16}" width="1000" height="230" rx="22" fill="#fff" stroke="#9aa3ad" stroke-width="3"/>')
out.append('<rect x="40" y="30" width="1000" height="230" rx="22" fill="#fff" stroke="#111" stroke-width="4"/>')
T(80,105,"A feather brushes slowly across",54); T(80,170,"the inside of your left forearm,",54,700); T(80,235,"a hand-width above the wrist.",54,700)
T(40,340,"+ 25 more touch sentences (pain, itch, ice…)",50)
arrow(1090,150,1175,150)
out.append('<rect x="1190" y="45" width="440" height="210" rx="26" fill="#1d2b3a"/>')
T(1410,135,"Qwen3-8B",62,700,"middle","#fff"); T(1410,215,"Qwen3-32B",62,700,"middle","#fff")
arrow(1645,150,1730,150)
cw=14.6
for j,(C,lab,nn_) in enumerate([(C8,"Qwen3-8B: 4,096 numbers",4096),(C32,"Qwen3-32B: 5,120 numbers",5120)]):
    y=22+j*128; ncell=round(44*nn_/4096)
    for c in range(ncell): out.append(f'<rect x="{1750+c*cw:.1f}" y="{y}" width="{cw-2:.1f}" height="58" fill="{dcol(C[c])}"/>')
    out.append(f'<rect x="1750" y="{y}" width="{ncell*cw-2:.1f}" height="58" fill="none" stroke="#111" stroke-width="2"/>')
    T(1750,y+108,lab,50)
T(1750,340,"internal state, average of all 26",50)
# ---------------- row B: example spot on the real body + its 5 most similar spots
views=[]
def vp_view(colname,front,x,y,w,h,yc,halfh):
    s=1 if front else -1
    views.append(dict(col=colname,vp=[x,y,w,h],halfh=halfh,eye=[0,yc,400*s],tgt=[0,yc,0],light=[-120*s,300,300*s]))
    sc=h/(2*halfh)
    return lambda p: (x+w/2+s*p[0]*sc, y+h/2-(p[1]-yc)*sc)
BX,BY,BW,BH=20,400,1040,700
projB=vp_view("plain",1,BX,BY,BW,BH,136,BH/2/7.6)
nn=d["nn_32B"][EX]; tw=d["tw"][EX]
def mark_ring(x,y): out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="none" stroke="#e8590c" stroke-width="6"/>')
pts={}
for j in nn: pts[j]=projB(SP[j]); mark_ring(*pts[j])
xt,yt=projB(SP[tw]); out.append(f'<path d="M{xt-11},{yt-11} L{xt+11},{yt+11} M{xt-11},{yt+11} L{xt+11},{yt-11}" stroke="#111" stroke-width="6"/>')
xe,ye=projB(SP[EX]); out.append(f'<circle cx="{xe:.1f}" cy="{ye:.1f}" r="11" fill="#111"/>')
def lead(x1,y1,x2,y2,c="#111"): out.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{c}" stroke-width="3.5"/>')
# direct labels (no key)
lead(xe+4,ye+12,xe+50,ye+130); T(xe+50,ye+185,"touched spot",52,700,"middle")
lead(xt-4,yt-12,xt-60,yt-95); T(30,yt-235,"same spot,",50); T(30,yt-175,"right arm:",50); T(30,yt-115,"skipped",50)
# right side of row B: the steps, in pictures where possible
lx=1110; ly=470
T(lx,ly,"For each of 407 spots on the body:",56,700)
out.append(f'<circle cx="{lx+22}" cy="{ly+82}" r="13" fill="none" stroke="#e8590c" stroke-width="6"/>')
T(lx+60,ly+100,"the 5 spots with the most similar",52); T(lx+60,ly+165,"Qwen3-32B internal states",52)
T(lx,ly+275,"Their straight-line distance on the body:",52,700)
T(lx,ly+345,"9, 9, 5, 5, 4 cm",52); arrow(lx+375,ly+328,lx+465,ly+328,6); T(lx+485,ly+345,"average 6 cm",52,700)
arrow(lx+835,ly+328,lx+925,ly+328,6); T(lx+945,ly+345,"color",52,700)
out.append(f'<circle cx="{lx+1120}" cy="{ly+328}" r="26" fill="{col(6.3)}"/>')
gx0,gx1,gy=lx,lx+1300,ly+400
out.append('<defs><linearGradient id="cs">'+"".join(f'<stop offset="{a/40:.3f}" stop-color="#%02x%02x%02x"/>'%c for a,c in STOPS)+'</linearGradient></defs>')
out.append(f'<rect x="{gx0}" y="{gy}" width="{gx1-gx0}" height="44" fill="url(#cs)"/>')
out.append(f'<line x1="{gx0+(gx1-gx0)*6.3/40:.0f}" y1="{gy-8}" x2="{gx0+(gx1-gx0)*6.3/40:.0f}" y2="{gy+52}" stroke="#111" stroke-width="5"/>')
for t in (0,10,20,30,40): T(gx0+(gx1-gx0)*t/40,gy+104,f"{t}"+(" cm or more" if t==40 else ""),48,400,"middle" if t<40 else "end" if False else "middle")
T(gx0,gy+170,"left and right count as one place",48)
# ---------------- results: color field on the body, front and back, two models
RY=1190
T(30,RY+20,"Similar internal states sit close together on the body",60,700)
bw,bh=620,830; halfh=bh/2/4.57; yc=89.5
rows=[("32B","Qwen3-32B"),("8B","Qwen3-8B")]
pitch=bh+175
for r,(Z,name) in enumerate(rows):
    top=RY+140+r*pitch
    for v,(lab,front) in enumerate([("front",1),("back",0)]):
        vp_view(Z,front,20+v*bw,top,bw,bh,yc,halfh)
        T(20+v*bw+bw/2,top+bh+68,lab,68,700,"middle")
    T(30,top-20,name,64,700)
# histograms
x0,x1,xmax=1460,2540,165; XA=lambda v: x0+(x1-x0)*v/xmax
hy0=RY+120
T(x0-140,hy0-20,"Each spot: average distance to",52,700); T(x0-140,hy0+45,"the 5 spots with the most similar states",52,700)
H4=[("32B","Qwen3-32B",d["d_32B"]),("8B","Qwen3-8B",d["d_8B"]),("words","word vectors before layer 1",d["d_words"]),("random","5 random spots per spot",R5)]
rh=430; bwid=2.5; hs=300/100
for k,(Z,name,v) in enumerate(H4):
    base=hy0+120+k*rh+340
    h,e=np.histogram(np.clip(v,0,xmax-0.01),bins=np.arange(0,xmax+bwid,bwid))
    for c,a in zip(h,e[:-1]):
        if c: out.append(f'<rect x="{XA(a)+0.6:.1f}" y="{base-c*hs:.1f}" width="{XA(a+bwid)-XA(a)-1.2:.1f}" height="{c*hs:.1f}" fill="{col(a+bwid/2)}"/>')
    out.append(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="#111" stroke-width="3"/>')
    # y axis: number of spots
    out.append(f'<line x1="{x0}" y1="{base}" x2="{x0}" y2="{base-100*hs}" stroke="#111" stroke-width="3"/>')
    for t in (0,50,100):
        out.append(f'<line x1="{x0-12}" y1="{base-t*hs:.0f}" x2="{x0}" y2="{base-t*hs:.0f}" stroke="#111" stroke-width="3"/>')
        T(x0-20,base-t*hs+16,str(t),48,400,"end")
    T(x0-100,base-150,"spots",48,400,"middle",extra=f'transform="rotate(-90 {x0-100} {base-150})"')
    med=np.median(v); lx2=XA(55) if Z!="random" else XA(4)
    T(lx2,base-240,name,52,700); T(lx2,base-175,f"median {med:.0f} cm",52)
ya=hy0+120+3*rh+340
for t in range(0,161,20):
    out.append(f'<line x1="{XA(t):.1f}" y1="{ya}" x2="{XA(t):.1f}" y2="{ya+14}" stroke="#111" stroke-width="3"/>')
    T(XA(t),ya+62,f"{t}",48,400,"middle")
T((x0+x1)/2,ya+130,"average distance on the body (cm)",52,400,"middle")
H=RY+140+2*pitch-60
H=max(H,ya+160)
# ---------------- render bodies, then overlay
shoot(dict(W=W,H=H,dpr=2,views=views),HERE+"/bodies.png")
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><image href="bodies.png" x="0" y="0" width="{W}" height="{H}"/>'+"".join(out)+"</svg>"
open(HERE+"/fig.html","w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=EXE,headless=True,args=["--enable-unsafe-swiftshader"]); pg=br.new_context(viewport={"width":W,"height":H},device_scale_factor=2).new_page()
    pg.goto("file://"+HERE+"/fig.html"); pg.wait_for_timeout(500); pg.screenshot(path="/tmp/nb4_2x.png"); br.close()
Image.open("/tmp/nb4_2x.png").convert("RGB").resize((W,H),Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_neighbours_v4.png"))
print(W,H,{z:round(float(np.median(v)),1) for z,_,v in H4})
