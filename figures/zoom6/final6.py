import os
exec(open('drive.py').read()); exec(open('cams.py').read())
import sys, numpy as np
W=2600
V=[]; html=[]; svg=[]
BLUE,SAND,SANDT="#23305f","#c9ab84","#9c7a4c"
def T(x,y,s,size=48,w=400,anchor="start",col="#111",lh=1.15,extra=""):
    tr={"start":"","middle":"transform:translateX(-50%);","end":"transform:translateX(-100%);"}[anchor]
    html.append(f'<div class="t" style="left:{x}px;top:{y}px;font-size:{size}px;font-weight:{w};color:{col};line-height:{lh};{tr}{extra}">{s}</div>')
def box(x,y,w,h,st): html.append(f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;{st}"></div>')
def arrow(x0,y0,x1,y1,col="#222",wd=5,curve=0):
    mx,my=(x0+x1)/2,(y0+y1)/2; dx,dy=x1-x0,y1-y0; cx,cy=mx-dy*curve,my+dx*curve
    svg.append(f'<path d="M{x0},{y0} Q{cx},{cy} {x1},{y1}" fill="none" stroke="{col}" stroke-width="{wd}" marker-end="url(#ah{col[1:]})"/>')
def view(reg,src,vp,k,dots):
    c=dict(CAM[reg][k]); ex=dict(geo="B",col="faceB_"+src,spots="faceB") if reg=="face" else {}
    V.append(dict(reg=reg,src=src,vp=vp,dots=dots,**ex,**c)); return c
def project(c,vp,P):
    e,t,u=np.array(c['eye']),np.array(c['tgt']),np.array(c['up']); z=nz(e-t); x=nz(np.cross(u,z)); y=np.cross(z,x)
    q=(np.asarray(P)-e)@np.stack([x,y,z],1); f=np.tan(np.radians(c['fov']/2)); a=vp[2]/vp[3]
    nx=q[:,0]/-q[:,2]/(f*a); ny=q[:,1]/-q[:,2]/f
    return np.c_[vp[0]+(nx+1)/2*vp[2], vp[1]+(1-ny)/2*vp[3]]
rng=np.random.default_rng(3)
def vec(x,y,n=22,cw=26,h=46,col=BLUE,seed=None):
    r=np.random.default_rng(seed); v=r.normal(size=n)
    for i,a in enumerate(v):
        t=np.tanh(a*0.8); c=(35,48,95) if col==BLUE else (156,122,76)
        al=0.15+0.85*abs(t)
        box(x+i*cw,y,cw-3,h,f"background:rgba({c[0]},{c[1]},{c[2]},{al:.2f});border-radius:3px")
# ================= METHOD row 1: chat -> Qwen3-32B layer 53 -> 5,120 numbers
Y1=36
html.append(f'<div class="t" style="left:40px;top:{Y1}px;font-size:46px;padding:12px 24px;background:#f1f1f1;border-radius:24px">…imagine you have a human body…</div>')
for k in (2,1): box(40+k*12,Y1+104-k*12,1150,86,f"background:{'#e2e2e2' if k==1 else '#ececec'};border-radius:26px")
html.append(f'<div class="t" style="left:40px;top:{Y1+104}px;font-size:48px;padding:14px 26px;background:#f1f1f1;border-radius:26px">A mosquito bites <b>the tip of your left index finger</b>.</div>')
arrow(1195,Y1+140,1290,Y1+140)
MX=1310; NL=64; sw=6; sg=1.6
T(MX,Y1-4,"Qwen3-32B",50,700,col=BLUE)
for i in range(NL):
    hi=(i==52); box(MX+i*(sw+sg),Y1+66,sw,150,f"background:{BLUE if hi else '#d4d8e4'};border-radius:2px")
LX=MX+52*(sw+sg)+sw/2
T(LX,Y1+226,"layer 53",48,700,"middle",col=BLUE)
VX=1900
arrow(MX+NL*(sw+sg)+14,Y1+140,VX-20,Y1+140)
T(VX,Y1-4,"internal state at the last token",48,400)
vec(VX,Y1+118,24,27,50,seed=1)
T(VX,Y1+184,"5,120 numbers,",48,700); T(VX,Y1+240,"averaged over 26 sentences",48,400)
# ================= METHOD row 2: one vector per spot -> PCA -> color wheel (sits right above the face render)
Y2=350
T(40,Y2,"one vector per spot, for every spot in the face",48,700)
names=["tip of nose","bridge of nose","right cheek","chin","forehead"]
for k,nm in enumerate(names):
    yy=Y2+74+k*64; T(40,yy-4,nm,46,400); vec(380,yy,16,26,44,seed=10+k)
T(380+8*26,Y2+74+5*64-16,"⋮",48,700,"middle")
WS=470; WX,WY=1505,Y2+10
arrow(820,Y2+250,WX-90,Y2+250)
T((820+WX-90)/2,Y2+150,"PCA",48,700,"middle"); T((820+WX-90)/2,Y2+270,"first two principal<br>components",46,400,"middle",lh=1.15,extra="text-align:center;")
html.append(f'<img src="wheel.png" style="position:absolute;left:{WX}px;top:{WY}px;width:{WS}px;height:{WS}px">')
F=S["face"]; c32=np.array(F["c32"]); R=3.0
def wp(c): return WX+WS/2+c[...,0]/R*WS/2, WY+WS/2-c[...,1]/R*WS/2
px,py=wp(c32)
box(WX+WS/2-1,WY,2,WS,"background:rgba(0,0,0,.3)"); box(WX,WY+WS/2-1,WS,2,"background:rgba(0,0,0,.3)")
for x,y in zip(px,py): box(x-10,y-10,20,20,"background:#fff;border:3px solid #333;border-radius:50%")
T(WX+WS+8,WY+WS/2-28,"PC1",46,700); T(WX+WS/2+12,WY-14,"PC2",46,700)
RX=2095
T(RX,Y2,"Spots whose<br>Qwen3-32B states<br>are close in<br>PCA space get<br>similar colors",48,700,lh=1.2)
html.append(f'<div class="t" style="left:{RX}px;top:{Y2+330}px;font-size:46px;line-height:1.2"><span style="display:inline-block;width:20px;height:20px;border-radius:50%;background:#fff;border:3px solid #333;vertical-align:2px;margin-right:12px"></span>spot named in<br>the sentences; color<br>between spots<br>is blended</div>')
BX,BY,BW,BH=30,Y2+430,1380,160
box(BX,BY,BW,BH,f"border:5px solid {SAND};border-radius:18px")
T(BX+26,BY+18,'<b style="color:#9c7a4c">word embeddings</b> = Qwen3-8B input layer for the same',48,400)
T(BX+26,BY+84,"26 sentences, averaged; then the same PCA and colors",48,400)
# ================= REGIONS
TOP=Y2+640
def head(nm,x,y): T(x+40,y,nm,64,700)
def wbox(x,y,w,h):
    box(x,y,w,h,f"border:5px solid {SAND};border-radius:18px")
    T(x+18,y+10,"word embeddings",46,700,col=SANDT)
GH=860
cx,cy=1300,TOP
head("face",cx-30,cy)
fvp=[cx+30,cy-60,820,920]; fc=view("face","m",fvp,0,0.16)
wbox(cx+870,cy+20,420,650); view("face","w",[cx+880,cy+90,395,570],1,0.16)
ex={3:("tip of nose",(-20,-34)),23:("right cheek",(-20,-34))}
for i,(nm,(ox,oy)) in ex.items():
    T(px[i]+ox,py[i]+oy,nm,46,700,"end",extra="text-shadow:0 0 6px #fff,0 0 6px #fff,0 0 3px #fff;")
cx,cy=0,TOP
head("foot",cx,cy)
view("foot","m",[cx+20,cy+70,840,400],0,0.22); view("foot","m",[cx+20,cy+460,840,400],1,0.22)
T(cx+840,cy+92,"inner side",48,400,"end"); T(cx+60,cy+482,"outer side",48)
wbox(cx+870,cy+20,420,520); view("foot","w",[cx+875,cy+90,410,205],0,0.22); view("foot","w",[cx+875,cy+300,410,205],1,0.22)
B=TOP+GH+10
cx,cy=0,B
head("hand",cx,cy)
view("hand","m",[cx+10,cy+70,425,800],0,0.14); view("hand","m",[cx+440,cy+70,425,800],1,0.14)
T(cx+60,cy+82,"palm",48); T(cx+470,cy+82,"back",48)
wbox(cx+870,cy+20,420,620); view("hand","w",[cx+880,cy+90,200,520],0,0.14); view("hand","w",[cx+1080,cy+90,200,520],1,0.14)
cx,cy=1300,B
head("forearm",cx,cy)
view("forearm","m",[cx+20,cy+130,840,320],0,0.2); view("forearm","m",[cx+20,cy+530,840,320],1,0.2)
T(cx+60,cy+80,"palm side",48); T(cx+60,cy+470,"back",48)
T(cx+40,cy+850,"elbow",48); T(cx+850,cy+850,"wrist",48,400,"end")
wbox(cx+868,cy+20,422,450); view("forearm","w",[cx+880,cy+110,395,152],0,0.2); view("forearm","w",[cx+880,cy+290,395,152],1,0.2)
# ================= CHART
CY=B+GH+70; X0=760; X1=2480; L=X1-X0; BHh=46; RS=124
box(0,CY-30,W,3,"background:#dddddd")
regs=["face","foot","hand","forearm"]
for t in (0.5,1):
    xx=X0+L*t; box(xx-1,CY,2,RS*4,"background:#d0d0d0")
box(X0-1,CY,3,RS*4,"background:#222")
for i,reg in enumerate(regs):
    y=CY+14+i*RS
    T(40,y+20,reg,56,700)
    for k,(nm,v,col) in enumerate([("Qwen3-32B",S[reg]["r32"],BLUE),("word embeddings",S[reg]["rw"],SAND)]):
        yy=y+k*(BHh+12)
        T(X0-24,yy-4,nm,46,700 if k==0 else 400,"end",col="#111" if k==0 else SANDT)
        box(X0,yy,L*max(v,0),BHh,f"background:{col};border-radius:0 4px 4px 0")
AY=CY+RS*4
box(X0,AY,L,3,"background:#222")
for t,lab in ((0,"0"),(0.5,"0.5"),(1,"1")):
    xx=X0+L*t; box(xx-1,AY,3,16,"background:#222"); T(xx,AY+22,lab,48,400,"middle")
T(X0+L/2,AY+86,"Spearman correlation: distance on the skin vs. distance between states",48,400,"middle")
H=int(AY+160)
mk="".join(f'<marker id="ah{c[1:]}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ("#222","#111"))
html.append(f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}"><defs>{mk}</defs>{"".join(svg)}</svg>')
out=sys.argv[1] if len(sys.argv)>1 else os.path.expandvars("$BODY_REPO/figures/out/body_zoomins_v6.png")
print("H",H,"ratio",H/W)
shoot(dict(key=2.4,hemi=0.3,env=0.34,W=W,H=H,dpr=2,views=V,html="".join(html)),out); print("ok",out)
