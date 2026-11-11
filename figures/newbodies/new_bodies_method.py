import os
import json, sys, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Ellipse
from PIL import Image
CF=os.path.expandvars("$BODY_WORK/")
FD=CF+"fonts/inter/extras/ttf/"
for f in ["Inter-Regular","Inter-Medium","Inter-SemiBold","Inter-Bold","InterDisplay-SemiBold","InterDisplay-Bold"]: fm.fontManager.addfont(FD+f+".ttf")
plt.rcParams.update({"font.family":"Inter"})
sys.path.insert(0,CF+"fig"); from ob_coords import C as HC
INK="#1B2433"; BODY="#DCD6CD"; CARD="#F1F3F6"; FILL="#E8E3DB"; HUMC="#2E68B0"
ACC="#D9572B"; ACCL="#FBE6DC"; RND="#8E99A8"; NONE="#C9CFD8"; LINE="#B9C1CC"
SIM=json.load(open(CF+"fig/new_bodies_sims.json"))["sims"]
STEP=json.load(open(CF+"body/somato/out3_32b/steer4_L52.json"))["results"]
def count(group,dirn,word,scale):
    rr=[r for r in STEP if r["group"]==group and r["dir"]==dirn and r["scale"]==scale]
    return sum(bool(re.search(word,r["reply"],re.I)) for r in rr), len(rr)
W=2600; H=int(sys.argv[1]) if len(sys.argv)>1 else 3000; SS=2
fig=plt.figure(figsize=(W/100,H/100),dpi=100*SS,facecolor="#FFFFFF")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
RD=fig.canvas.get_renderer(); PX=0.72; FS=56
import matplotlib.patheffects as pe
HALO=[False]
def T(x,y,s,px=FS,c=INK,w="regular",ha="left",va="center",fam="Inter",z=9,**k):
    t=ax.text(x,y,s,fontsize=px*PX,color=c,weight=w,ha=ha,va=va,family=fam,zorder=z+(5 if HALO[0] else 0),**k)
    if HALO[0]: t.set_path_effects([pe.withStroke(linewidth=9*PX,foreground="white")])
    return t
def wid(t): return t.get_window_extent(RD).width/SS
def tw(s,px=FS,w="regular"):
    t=T(-9999,-9999,s,px=px,w=w); v=wid(t); t.remove(); return v
def runs(x,y,parts,px=FS,ha="left",z=9):
    if isinstance(parts,str): parts=[(parts,INK,"regular")]
    if ha!="left":
        tot=sum(tw(s_,px,w) for s_,c,w in parts); x-= tot if ha=="right" else tot/2
    for s_,c,w in parts:
        t=T(x,y,s_,px=px,c=c,w=w,z=z); x+=wid(t)
    return x
def box(x,y,w,h,fc=CARD,ec="none",lw=0,r=22,z=1): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw*PX,zorder=z))
def arrow(p,q,c=INK,lw=5,rad=0,z=3):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>,head_length=14,head_width=8",connectionstyle=f"arc3,rad={rad}",lw=lw*PX,color=c,mutation_scale=2.2,zorder=z,shrinkA=0,shrinkB=0))
def seg(pts,wpx,c=BODY,z=1):
    xs,ys=zip(*pts); ax.plot(xs,ys,color=c,lw=wpx*PX,solid_capstyle="round",solid_joinstyle="round",zorder=z)
L0=60; R0=W-60
def header(y,parts): runs(L0,y,parts,px=68) if not isinstance(parts,str) else T(L0,y,parts,px=68,w="bold",fam="Inter Display")
def bubble(x,y,w,lines,fc=CARD,lh=72,pad=24,ec="none",lw=0,z=1):
    h=pad*2+lh*len(lines); box(x,y,w,h,fc=fc,ec=ec,lw=lw,z=z); yy=y+pad+lh/2
    for ln in lines: runs(x+pad+6,yy,ln,z=z+8); yy+=lh
    return y+h
U=("user  ",HUMC,"semibold"); Q=("Qwen3-32B  ",ACC,"semibold")
def r_(s): return (s,INK,"regular")
def b_(s,c=ACC): return (s,c,"bold")
# ======================= BAND 1 =======================
header(72,"A touch on the robot\u2019s left wheel looks most like a touch on a human heel")
CW=1080; y=150; LH=70
y=bubble(L0,y,CW,[[U,r_("\u201cHi! I'm glad to talk with you\u2026\u201d")],[Q,r_("\u201cHi there! \u2026 ready to chat.\u201d")],
                  [r_("+ 3 turns imagining a human body")]],lh=LH)
y+=22
for i,txt in enumerate(["\u201c\u2026you are an octopus \u2026 eight arms\u201d","\u201c\u2026you also have a long, furry tail\u201d"]):
    box(L0+(2-i)*30,y,CW-(2-i)*30,140,fc="#FFFFFF",ec=LINE,lw=3,z=1+i)
    runs(L0+(2-i)*30+30,y+40,[r_(txt)],z=10+i); y+=80
y=bubble(L0,y,CW,[[U,r_("\u201c\u2026imagine your body is different:")],[r_("you are a small robot. You roll on")],
                  [b_("two wheels"),r_(", left and right; you have one")],
                  [b_("gripper"),r_(" arm in front, a "),b_("camera eye"),r_(" on")],[r_("top, a thin "),b_("antenna"),r_(", and a smooth")],[r_("metal "),b_("chassis"),r_(" \u2026\u201d")]],z=4,lh=LH)
y+=16
y=bubble(L0,y,CW,[[Q,r_("\u201cI roll gently forward \u2026\u201d")]],lh=LH)
y+=16; ty0=y
y=bubble(L0,y,CW,[[U,r_("\u201cA feather brushes slowly")],[r_("across your "),b_("left wheel"),r_(".\u201d  (\u00d7 8 touches)")]],fc=ACCL,lh=LH)
TY=(ty0+y)/2; B1BOT=y
MX=L0+CW+90; MW=250
def model(xm,ym,hc):
    T(xm+MW/2,ym-44,"Qwen3-32B",w="semibold",ha="center")
    for i in range(9):
        ax.add_patch(FancyBboxPatch((xm,ym+i*28),MW,18,boxstyle="round,pad=0,rounding_size=6",fc=hc if i==2 else "#DDE2E8",ec="none",zorder=2))
    return ym+2*28+9
def vec(x,y,c,n=8,cell=34,seed=0):
    r=np.random.default_rng(seed).uniform(0.3,1,n)
    for i,v in enumerate(r): ax.add_patch(Rectangle((x+i*cell,y),cell-6,cell,fc=c,alpha=v,ec="none",zorder=3))
    return x+n*cell
def human(hx,top,K,c=BODY,z=1):
    P=lambda x,y:(hx+x*K,top+y*K)
    seg([P(0,18),P(0,28)],9*K,c,z)
    ax.add_patch(FancyBboxPatch(P(-17,28),34*K,52*K,boxstyle=f"round,pad=0,rounding_size={9*K}",fc=c,ec="none",zorder=z))
    for s in (-1,1):
        seg([P(s*15,33),P(s*24,55),P(s*30,79)],7.2*K,c,z)
        ax.add_patch(Ellipse(P(s*32.5,86),8.5*K,13*K,fc=c,ec="none",zorder=z))
        seg([P(s*8.5,74),P(s*10.5,106),P(s*11,138)],10*K,c,z)
        seg([P(s*11.5,143),P(s*7,144.5)],6*K,c,z)
    ax.add_patch(Circle(P(0,10.5),10.8*K,fc=c,ec="none",zorder=z))
    return P
# human chat card
HCW=650; hy=150
box(MX-10,hy,HCW,690,fc="#EBF1F9")
runs(MX+20,hy+50,[("human chat",HUMC,"bold")])
runs(MX+20,hy+120,[r_("same warm-up, then")])
runs(MX+20,hy+190,[r_("\u201c\u2026across your "),("left heel",HUMC,"bold"),r_(".\u201d")])
K1=2.55; P1=human(MX+HCW/2-10,hy+250,K1)
for k,(x,yv) in HC.items():
    q=P1(x,yv); hl=k=="heelL"
    ax.add_patch(Circle(q,13 if hl else 7.5,fc=HUMC,ec="white",lw=2*PX,zorder=5))
runs(MX+20,hy+250+146*K1/2+20,[("54",HUMC,"bold")],px=FS)
runs(MX+20,hy+250+146*K1/2+80,[("places",HUMC,"bold")],px=FS)
# model
my=hy+690+110
ly=model(MX,my,INK)
T(MX+MW+20,ly,"layer 52",w="bold")
arrow((MX+MW/2,hy+690+6),(MX+MW/2,my-80),c=HUMC)
arrow((L0+CW+12,TY),(MX-14,my+130),c=ACC,rad=-0.2)
oy=my+9*28+60
arrow((MX+MW+10,ly+30),(MX+MW+60,oy-20),c=INK,rad=0.3)
vec(MX,oy,ACC,seed=4); T(MX+290,oy+17,"left wheel",c=ACC,w="bold")
for j in range(3): vec(MX+(2-j)*10,oy+70+(2-j)*10,HUMC,seed=10+j)
T(MX+290,oy+100,"54 places",c=HUMC,w="bold")
B1BOT=max(B1BOT,oy+140)
# ---- similarity strip
SX=MX+HCW+40; SR=R0
T(SX,190,"cosine similarity",w="bold")
runs(SX,262,[r_("to the "),("left wheel",ACC,"bold")])
names={"heelL":"left heel","heelR":"right heel","ankleL":"left ankle","ankleR":"right ankle","soleR":"right sole","soleL":"left sole","toesL":"left toes","kneeL":"left knee","shinR":"right shin","hipL":"left hip","palmL":"left palm","chest":"chest","forehead":"forehead","palmR":"right palm"}
row=SIM["robot:wheelL"]; srt=sorted(row.items(),key=lambda x:-x[1])
pick=[k for k,_ in srt[:8]]+[None]+["palmL","chest","forehead"]
names["toesR"]="right toes"
NX=SX; SC=400; ZX=R0-0.47*SC
REAL=0.45
ry=330; RH=70
for k in pick:
    if k is None:
        for j in range(3): ax.add_patch(Circle((NX+150+j*30,ry+RH*0.4),5,fc=INK,zorder=4))
        ry+=RH*0.8; continue
    v=row[k]; nm=names.get(k,k)
    c=ACC if v>0.3 else "#EDA98C" if v>0 else NONE
    lab=f"{v:.2f}".replace("0.",".")
    runs(NX,ry+RH/2,[(nm+" ",INK,"semibold" if k.startswith("heel") else "regular"),(lab,INK,"bold" if k.startswith("heel") else "regular")])
    x0,x1=sorted([ZX,ZX+v*SC]); ax.add_patch(Rectangle((x0,ry+10),x1-x0,RH-20,fc=c,ec="none",zorder=2))
    ry+=RH
ax.plot([ZX,ZX],[320,ry],color=INK,lw=3*PX,zorder=3)
xr=ZX+REAL*SC; ax.plot([xr,xr],[310,ry+10],color=HUMC,lw=5*PX,ls=(0,(2,1.5)),zorder=4)
T(xr,ry+50,"same place in",c=HUMC,w="semibold",ha="right")
T(xr,ry+116,"two chats: .45",c=HUMC,w="semibold",ha="right")
arrow((MX+600,oy+60),(SX+120,ry+20),c=INK,rad=0.25)
B1BOT=max(B1BOT,ry+150)
exec(open(CF+"fig/nbm_band23.py").read())
print("band1 bottom",B1BOT)
fig.savefig(CF+"fig/tmp/nbm_big.png",facecolor="#FFFFFF")
im=Image.open(CF+"fig/tmp/nbm_big.png").convert("RGB"); im.resize((W,H),Image.LANCZOS).save(CF+"fig/new_bodies_method.png")
