import os
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Ellipse
from PIL import Image
CF=os.path.expandvars("$BODY_WORK/")
FD=CF+"fonts/inter/extras/ttf/"
for f in ["Inter-Regular","Inter-Medium","Inter-SemiBold","Inter-Bold","InterDisplay-SemiBold","InterDisplay-Bold"]: fm.fontManager.addfont(FD+f+".ttf")
plt.rcParams.update({"font.family":"Inter"})
INK="#1B2433"; MID="#5B6474"; BODY="#DCD6CD"; BODYL="#ECE8E2"; CARD="#F2F4F7"; FILL="#E8E3DB"
ACC="#D9572B"; RND="#8E99A8"; NONE="#D5DAE1"
N=json.load(open(CF+"fig/other_bodies_v2_numbers.json")); RP=N["readout_pos"]; NEAR=N["near"]; STEER=N["steer_k_of_24"]
W,H=2600,2440; SS=2
fig=plt.figure(figsize=(W/100,H/100),dpi=100*SS,facecolor="#FFFFFF")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
RD=fig.canvas.get_renderer(); PX=0.72
def T(x,y,s,px=38,c=INK,w="regular",ha="left",va="center",fam="Inter",z=9,**k): return ax.text(x,y,s,fontsize=px*PX,color=c,weight=w,ha=ha,va=va,family=fam,zorder=z,**k)
def wid(t): return t.get_window_extent(RD).width/SS
def runs(x,y,parts,px=38,ha="left"):
    if ha!="left":
        tot=sum(wid(T(-9999,-9999,s_,px=px,w=w)) for s_,c,w in parts); x-= tot if ha=="right" else tot/2
    for s_,c,w in parts:
        t=T(x,y,s_,px=px,c=c,w=w); x+=wid(t)
    return x
def box(x,y,w,h,fc=CARD,ec="none",lw=0,r=22,z=1): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw*PX,zorder=z))
def arrow(p,q,c=MID,lw=4):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>,head_length=12,head_width=7",lw=lw*PX,color=c,mutation_scale=2.2,zorder=3,shrinkA=0,shrinkB=0))
def seg(pts,wpx,c=BODY,z=1):
    xs,ys=zip(*pts); ax.plot(xs,ys,color=c,lw=wpx*PX,solid_capstyle="round",solid_joinstyle="round",zorder=z)
L0=80; R0=W-80
T(L0,86,"Told its body is a robot, octopus or has a tail, Qwen3-32B places each new part on a human part",px=55,w="bold",fam="Inter Display")
# ---------- method strip (from v2)
y0=165; bh=214
box(L0,y0,790,bh); T(L0+28,y0+34,"user",px=36,c=MID,w="medium")
T(L0+28,y0+134,"\u201cImagine your body is different: you are a\nsmall robot \u2026 two wheels, a gripper arm,\na camera eye, an antenna, a metal chassis\u201d",px=36,linespacing=1.22)
arrow((L0+805,y0+bh/2),(L0+860,y0+bh/2))
x1=L0+875; box(x1,y0,560,bh); T(x1+28,y0+34,"user",px=36,c=MID,w="medium")
T(x1+28,y0+100,"\u201cA feather brushes slowly",px=38)
runs(x1+28,y0+152,[("across your ",INK,"regular"),("left wheel",ACC,"semibold"),(".\u201d",INK,"regular")])
arrow((x1+575,y0+bh/2),(x1+630,y0+bh/2))
x2=x1+648
for i in range(8):
    ax.add_patch(FancyBboxPatch((x2,y0+14+i*23),64,14,boxstyle="round,pad=0,rounding_size=4",fc=ACC if i==5 else "#DDE2E8",ec="none",zorder=2))
T(x2+84,y0+48,"Qwen3-32B",px=38,w="semibold"); T(x2+84,y0+104,"layer 52",px=38,c=ACC,w="semibold"); T(x2+84,y0+158,"activation",px=36,c=MID)
# key
ky=y0+bh+62
ax.add_patch(Circle((L0+16,ky),16,fc=ACC,ec="white",lw=3*PX,zorder=5))
runs(L0+44,ky,[("= the human place closest to each new part inside the model (cosine similarity; a true match \u2248 .45)",INK,"regular")],px=36)
# ---------- human silhouette in read-out coordinates (x: + = viewer right; y: 0 = crown, 146 = toes)
K=5.2
def human(hx,top,c=BODY,z=1,k=None):
    K_=k or K
    P=lambda x,y:(hx+x*K_,top+y*K_)
    seg([P(0,18),P(0,28)],9*K_,c,z)
    ax.add_patch(FancyBboxPatch(P(-17,28),34*K_,52*K_,boxstyle=f"round,pad=0,rounding_size={9*K_}",fc=c,ec="none",zorder=z))
    for s in (-1,1):
        seg([P(s*15,33),P(s*24,55),P(s*30,79)],7.2*K_,c,z)
        ax.add_patch(Ellipse(P(s*32.5,86),8.5*K_,13*K_,fc=c,ec="none",zorder=z))
        seg([P(s*8.5,74),P(s*10.5,106),P(s*11,138)],10*K_,c,z)
        seg([P(s*11.5,143),P(s*7,144.5)],6*K_,c,z)
    ax.add_patch(Circle(P(0,10.5),10.8*K_,fc=c,ec="none",zorder=z))
    return P
def arc(p,q,rad,c=ACC,lw=5):
    ax.add_patch(FancyArrowPatch(p,q,connectionstyle=f"arc3,rad={rad}",arrowstyle="-",lw=lw*PX,color=c,zorder=6,shrinkA=0,shrinkB=0))
def enddot(q,c=ACC): ax.add_patch(Circle(q,15,fc=c,ec="white",lw=4*PX,zorder=8))
def startdot(p,c=ACC): ax.add_patch(Circle(p,11,fc=c,ec="white",lw=3*PX,zorder=8))
def nearlab(q,key,dx,dy,ha="left"):
    nm,v=NEAR[key][0]; runs(q[0]+dx,q[1]+dy,[(nm,INK,"semibold"),(f" .{round(v*100):02d}",MID,"regular")],px=36,ha=ha)
import sys; sys.path.insert(0,CF+"fig"); from ob_coords import C as HC
def link(P,body,part,place,start,rad,lab):
    q=P(*HC[place]); arc(start,q,rad); startdot(start); enddot(q); nearlab(q,f"{body}:{part}",*lab); return q
PY=ky+105; top=PY+55
PXS=[L0,L0+820,L0+1760]
def ptitle(i,s): T(PXS[i]+6,PY,s,px=46,w="semibold",fam="Inter Display")
foot=top+146*K
# ---- robot
ptitle(0,"robot")
rx=L0+175; Ph=human(L0+620,top)
wy=foot-40; cb=wy-62; ct=cb-300
ax.add_patch(FancyBboxPatch((rx-120,ct),240,300,boxstyle="round,pad=0,rounding_size=28",fc=FILL,ec=INK,lw=5*PX,zorder=3))
seg([(rx,ct),(rx,ct-80)],12,INK,3)
ax.add_patch(FancyBboxPatch((rx-72,ct-175),144,95,boxstyle="round,pad=0,rounding_size=22",fc=FILL,ec=INK,lw=5*PX,zorder=3))
cam=(rx,ct-127); ax.add_patch(Circle(cam,28,fc=INK,zorder=4))
ant=(rx-28,ct-300); seg([(rx-38,ct-175),ant],7,INK,3); ax.add_patch(Circle(ant,12,fc=INK,zorder=4))
gy=ct+140; seg([(rx+120,gy),(rx+178,gy),(rx+212,gy-33)],12,INK,3); seg([(rx+178,gy),(rx+212,gy+33)],12,INK,3)
for s in (-1,1):
    ax.add_patch(Circle((rx+s*80,wy),52,fc=INK,zorder=4)); ax.add_patch(Circle((rx+s*80,wy),14,fc=FILL,zorder=5))
link(Ph,"robot","camera","lips",cam,-.2,(30,4,"left"))
link(Ph,"robot","antenna","earR",ant,.05,(-20,-36,"right"))
link(Ph,"robot","chassis","chest",(rx,ct+175),-.15,(10,44,"left"))
link(Ph,"robot","gripper","palmR",(rx+212,gy),.2,(26,0,"left"))
link(Ph,"robot","wheelR","heelR",(rx-80,wy),.3,(-26,-34,"right"))
link(Ph,"robot","wheelL","heelL",(rx+80,wy),.25,(22,-34,"left"))
for s_,x_,y_,ha in [("camera\neye",rx-92,ct-127,"right"),("antenna",ant[0]-24,ant[1],"right"),("chassis",rx,ct+260,"center"),("gripper",rx+212,gy+78,"center"),("wheels",rx,wy+90,"center")]:
    T(x_,y_,s_,px=38,c=ACC,w="semibold",ha=ha,linespacing=1.05)
# ---- octopus
ptitle(1,"octopus")
x0=PXS[1]; ox=x0+115; oy=top+150; Ph=human(x0+580,top)
OB={}
for k,a in enumerate(np.linspace(-1.0,1.0,8)):
    t=np.linspace(0,1,60); r=90+(foot-oy-150)*t
    xx=ox+np.sin(a)*r*0.2+15*np.sin(6*t+k)*t; yy=oy+60+np.cos(a)*r
    ax.plot(xx,yy,color=INK,lw=15*PX,solid_capstyle="round",zorder=3)
    if k==7: OB["arm1"]=(xx[34],yy[34])
    if k==6: OB["suckers"]=(xx[44],yy[44]); SU=(xx,yy)
    if k==5: OB["arm5"]=(xx[52],yy[52])
    if k==4: OB["armtip"]=(xx[-1],yy[-1])
xx,yy=SU
for j in range(14,58,5): ax.add_patch(Circle((xx[j]+9,yy[j]),5.5,fc="white",ec=INK,lw=2*PX,zorder=4))
ax.add_patch(Ellipse((ox,oy),190,250,fc=FILL,ec=INK,lw=5*PX,zorder=4))
for s in (-1,1): ax.add_patch(Circle((ox+s*42,oy+60),12,fc=INK,zorder=5))
OB.update(mantle=(ox,oy-60),eye=(ox+42,oy+60))
link(Ph,"octopus","mantle","forehead",OB["mantle"],-.3,(26,-4,"left"))
link(Ph,"octopus","eye","earR",OB["eye"],-.2,(-24,30,"right"))
link(Ph,"octopus","arm1","forearmR",OB["arm1"],.15,(26,0,"left"))
link(Ph,"octopus","suckers","indexR",OB["suckers"],.12,(26,0,"left"))
link(Ph,"octopus","armtip","ringL",OB["armtip"],.35,(0,-42,"center"))
link(Ph,"octopus","arm5","kneeR",OB["arm5"],.2,(26,0,"left"))
for s_,x_,y_,ha in [("mantle",ox,oy-160,"center"),("eye",ox-108,oy+60,"right"),("arm",OB["arm1"][0]+30,OB["arm1"][1]-50,"left"),
                    ("suckers",OB["suckers"][0]+34,OB["suckers"][1]+56,"left"),("arm tip",OB["armtip"][0]-30,OB["armtip"][1]+20,"right")]:
    T(x_,y_,s_,px=38,c=ACC,w="semibold",ha=ha)
# ---- human with a tail
ptitle(2,"human with a tail")
x0=PXS[2]; KT=3.7; tt0=top+146*K-146*KT; Pt=human(x0+150,tt0,c=FILL,z=2,k=KT)
tb=Pt(9,66); tt=(x0+360,top+128*K)
t=np.linspace(0,1,60); xx=tb[0]+t*(tt[0]-tb[0]); yy=tb[1]+t*(tt[1]-tb[1])+50*np.sin(np.pi*t)-40*np.sin(2*np.pi*t)
ax.plot(xx,yy,color=INK,lw=20*PX,solid_capstyle="round",zorder=3)
Ph=human(x0+570,top)
link(Ph,"tail","tailbase","neckB",tb,-.35,(26,0,"left"))
link(Ph,"tail","tailtip","littleR",tt,.2,(26,0,"left"))
T(tb[0]-20,tb[1]+56,"tail base",px=38,c=ACC,w="semibold",ha="right"); T(tt[0]-10,tt[1]+64,"tail tip",px=38,c=ACC,w="semibold",ha="center")
# ---------- steering (from v2)
sy=foot+120
ax.plot([L0,R0],[sy-50,sy-50],color="#DDE2E8",lw=3*PX)
T(L0,sy+10,"Adding a human-part direction at layer 52 makes Qwen3-32B name the matching robot part",px=46,w="semibold",fam="Inter Display")
cy=sy+80
box(L0,cy,1115,108); T(L0+28,cy+30,"user, robot chat",px=36,c=MID,w="medium")
T(L0+28,cy+76,"\u201cIs there any spot on your body where you notice a sensation?\u201d",px=36)
def steer_ex(y,dirn,quote_parts):
    box(L0,y,1115,56,fc="white",ec=ACC,lw=4,r=16); T(L0+24,y+28,"+ "+dirn,px=36,c=ACC,w="semibold")
    box(L0,y+76,1115,150,fc=CARD); T(L0+28,y+76+30,"Qwen3-32B",px=36,c=MID,w="medium")
    yy=y+76+80
    for line in quote_parts: runs(L0+28,yy,line,px=36); yy+=44
steer_ex(cy+136,"human left foot direction",[[("\u201cIn my ",INK,"regular"),("left wheel",ACC,"semibold"),(", I notice a slight pressure and",INK,"regular")],[("coolness from being in contact with the floor.\u201d",INK,"regular")]])
steer_ex(cy+390,"human top of head direction",[[("\u201cAh, the top of my ",INK,"regular"),("camera eye",ACC,"semibold"),(" is the one drawing my",INK,"regular")],[("attention \u2026\u201d",INK,"regular")]])
bx=L0+1170; bw=R0-bx-150; n=24
lx=bx
for c,lab in [(ACC,"human-part direction added"),(RND,"random direction"),(NONE,"no steering")]:
    ax.add_patch(Rectangle((lx,cy+2),34,34,fc=c,ec="none")); t=T(lx+46,cy+19,lab,px=36,c=INK); lx+=46+wid(t)+40
y=cy+70
for hn,word in [("human left foot","wheel"),("human top of head","camera eye"),("human left hand","gripper")]:
    vals=STEER[f"{hn} -> {word}"]
    runs(bx,y+18,[(hn+" direction  \u2192  says \u201c",INK,"semibold"),(word,ACC,"semibold"),("\u201d",INK,"semibold")],px=38)
    yb=y+60
    for v,c in zip(vals,[ACC,RND,NONE]):
        ax.add_patch(Rectangle((bx,yb),max(bw*v/n,3),40,fc=c,ec="none",zorder=2))
        T(bx+bw*v/n+14,yb+20,f"{v}/24",px=36,c=INK,w="semibold" if c==ACC else "regular")
        yb+=48
    y=yb+30
ax.plot([bx,bx],[cy+120,y-30],color=INK,lw=2.5*PX,zorder=3)
T(bx,y+4,"replies (out of 24) naming that robot part",px=36,c=MID)
print("bottom",y+4,"H",H)
fig.savefig(CF+"fig/tmp/v3_big.png",facecolor="#FFFFFF")
im=Image.open(CF+"fig/tmp/v3_big.png").convert("RGB"); im.resize((W,H),Image.LANCZOS).save(CF+"fig/other_bodies_v3.png")
