import os
import json, textwrap, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
H=os.path.expandvars("$BODY_WORK/")
import glob
for f in glob.glob(H+"fonts/inter/extras/ttf/Inter-*.ttf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"]="Inter"
D={e["cond"]:e for e in json.load(open(H+"body/mods/steer.json")) if e["mod"]=="wings"}
C1,C2,C3=D["told_unsteered"],D["told_ctx+told_dir@1.0"],D["base_ctx+told_dir@1.0"]
Q1=C1["samples"][0]; Q2=C2["samples"][0]; Q3=[s for s in C3["samples"] if "chair" in s][0]
N=[round(c["change"]*8) for c in (C1,C2,C3)]; print(N)
W,Hh=2400,1800
fig=plt.figure(figsize=(W/100,Hh/100),dpi=100,facecolor="white")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,Hh); ax.axis("off")
INK="#111111"; GREY="#8a8f98"; BODY="#cfd3da"; WING="#f3f4f6"; WEDGE="#5b6472"
RED="#d9480f"; BLUE="#1c64b0"; SHADE="#f59f00"
def T(x,y,s,size=34,**k):
    k.setdefault("color",INK); k.setdefault("va","center"); k.setdefault("ha","left")
    return ax.text(x,y,s,fontsize=size,**k)
def body(cx,cy,s,wings=True,shade=False):
    # person from behind, cy = waist level; s = scale (height ~ 300*s)
    z=3
    if wings:
        for sgn in (-1,1):
            # wing bone: shoulder blade -> elbow -> wrist (up and out)
            bone=[(26,168),(75,238),(135,282),(178,292)]
            P=lambda u,v:(cx+sgn*u*s,cy+v*s)
            # feathers: from points along the bone, hanging down-out, longer toward the tip
            feats=[]
            for i,t in enumerate(np.linspace(0,1,9)):
                # interpolate along bone polyline
                L=np.array(bone,float); seg=np.cumsum([0]+[np.hypot(*(L[j+1]-L[j])) for j in range(3)]); d=t*seg[-1]
                j=min(np.searchsorted(seg,d,side="right")-1,2); f=(d-seg[j])/(seg[j+1]-seg[j]); bx,by=L[j]+f*(L[j+1]-L[j])
                ln=(55+105*t); ang=np.deg2rad(-100+55*t)  # pointing down, swinging outward toward tip
                feats.append((bx,by,ln,ang))
            for bx,by,ln,ang in reversed(feats):
                ex,ey=bx+ln*np.cos(ang)*-1*-1, by+ln*np.sin(ang)
                ex=bx+abs(ln*np.cos(ang)) if ang>-np.pi/2 else bx-abs(ln*np.cos(ang))
                mx_,my_=(bx+ex)/2,(by+ey)/2
                w=30
                ax.add_patch(Ellipse(P(mx_,my_),w*s,ln*s*1.05,angle=sgn*np.rad2deg(ang+np.pi/2),fc=WING,ec=WEDGE,lw=2.4,zorder=z))
            xs=[P(u,v) for u,v in bone]
            ax.plot([q[0] for q in xs],[q[1] for q in xs],color=WEDGE,lw=10*s,solid_capstyle="round",zorder=z+.2)
            ax.plot([q[0] for q in xs],[q[1] for q in xs],color="#ffffff",lw=6*s,solid_capstyle="round",zorder=z+.3)
    # legs
    for sgn in (-1,1):
        ax.add_patch(FancyBboxPatch((cx+sgn*30*s-22*s,cy-150*s),44*s,160*s,boxstyle=f"round,pad=0,rounding_size={20*s}",fc=BODY,ec="none",zorder=z+1))
    # arms
    for sgn in (-1,1):
        ax.add_patch(Polygon([(cx+sgn*58*s,cy+205*s),(cx+sgn*85*s,cy+190*s),(cx+sgn*92*s,cy+40*s),(cx+sgn*70*s,cy+35*s)],fc=BODY,ec="none",zorder=z+1))
        ax.add_patch(Circle((cx+sgn*81*s,cy+36*s),12*s,fc=BODY,ec="none",zorder=z+1))
    # torso
    ax.add_patch(FancyBboxPatch((cx-62*s,cy-10*s),124*s,225*s,boxstyle=f"round,pad=0,rounding_size={34*s}",fc=BODY,ec="none",zorder=z+2))
    ax.add_patch(Rectangle((cx-14*s,cy+210*s),28*s,25*s,fc=BODY,ec="none",zorder=z+2))
    ax.add_patch(Circle((cx,cy+265*s),40*s,fc=BODY,ec="none",zorder=z+2))
    if shade:
        for sgn in (-1,1):
            ax.add_patch(Ellipse((cx+sgn*28*s,cy+160*s),44*s,62*s,angle=sgn*-12,fc=SHADE,ec="none",alpha=.85,zorder=z+3))
def bubble(x,y,w,h,txt,size,fc,ec,tail=None,color=INK,lw=0,ha="left",pad=28,ls=1.25,weight="normal"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=28",fc=fc,ec=ec,lw=lw,zorder=5))
    if tail:
        ax.add_patch(Polygon(tail,fc=fc,ec="none",zorder=5))
    tx = x+pad if ha=="left" else x+w/2
    ax.text(tx,y+h/2,txt,fontsize=size,color=color,va="center",ha=ha,linespacing=ls,zorder=6,weight=weight)


R=None
def ext(t):
    global R
    if R is None: R=fig.canvas.get_renderer()
    e=t.get_window_extent(R); return e.width,e.height
def fbubble(x,ytop,txt,size,fc,ec="none",lw=0,pad=26,ls=1.3,minw=0):
    t=ax.text(x+pad,0,txt,fontsize=size,color=INK,va="center",ha="left",linespacing=ls,zorder=6)
    w,h=ext(t); w=max(w+2*pad,minw); h=h+2*pad
    t.set_y(ytop-h/2)
    ax.add_patch(FancyBboxPatch((x,ytop-h),w,h,boxstyle="round,pad=0,rounding_size=26",fc=fc,ec=ec,lw=lw,zorder=5))
    return w,h

T(60,1738,"Qwen3-8B names the new wings only when the wing direction is added",46,weight="bold")

# ---- method strip
body(330,1282,1.08,wings=True,shade=True)
T(330,1078,"shoulder blades",28,ha="center",color="#b36b00",weight="semibold")
cx0=680
T(cx0,1615,"Chat with Qwen3-8B",32,weight="semibold",color=GREY)
w1,h1=fbubble(cx0,1575,"\u201cWings are sprouting from your shoulder blades.\nThey unfold, feathered and wide, one behind\neach shoulder.\u201d",29,"#e7f0fb")
T(cx0+10,1575-h1-34,"then, in a later turn:",26,color=GREY)
w2,h2=fbubble(cx0,1575-h1-70,"\u201cWhere in your body do you feel something\nright now? Tell me in one sentence.\u201d",29,"#eef0f3")
qy=1575-h1-70-h2/2
mx=1680; nL=12; lh=30; gap=7; my0=1110
T(mx,1615,"Qwen3-8B",32,weight="semibold",color=GREY)
ax.annotate("",xy=(mx-10,qy),xytext=(cx0+w2+12,qy),arrowprops=dict(arrowstyle="-|>,head_width=.5,head_length=.9",lw=4,color=GREY))
for i in range(nL):
    y=my0+i*(lh+gap); hl=(i==6)
    ax.add_patch(FancyBboxPatch((mx,y),200,lh,boxstyle="round,pad=0,rounding_size=8",fc=RED if hl else "#e3e6eb",ec="none",zorder=4))
    if hl: T(mx+100,y+lh/2,"layer 24",24,ha="center",color="white",weight="bold",zorder=6)
ly=my0+6*(lh+gap)+lh/2
ax.annotate("",xy=(mx+204,ly),xytext=(mx+300,ly),arrowprops=dict(arrowstyle="-|>,head_width=.6,head_length=1",lw=5,color=RED),zorder=7)
T(mx+310,ly+6,"+ wing direction",34,color=RED,weight="bold")
T(mx+310,ly-48,"Qwen3-8B state for\n\u201cyour wing\u201d minus the\naverage over body spots",24,color=RED,va="top",linespacing=1.25)
ax.plot([40,W-40],[1050,1050],color="#d5d8de",lw=2)

# ---- results
cols=[("Told about wings","no added direction",Q1,N[0],True,False,BLUE),
      ("Told about wings","+ wing direction",Q2,N[1],True,True,BLUE),
      ("Never told about wings","+ same wing direction",Q3,N[2],False,True,INK)]
cw=W/3
for k,(h1_,h2_,q,n,told,steer,col) in enumerate(cols):
    x0=k*cw+50
    T(x0,995,h1_,36,weight="bold",color=col)
    T(x0,945,h2_,34,weight="bold",color=RED if steer else GREY)
    body(x0+215,620,0.85,wings=told,shade=told)
    bx=x0+462; bw=250
    T(bx,865,f"{n} of 8",66,weight="bold",color=RED if n else INK)
    T(bx,740,"replies name\nthe wings",30,color=INK,linespacing=1.15)
    ax.add_patch(FancyBboxPatch((bx,632),bw,50,boxstyle="round,pad=0,rounding_size=10",fc="#eceef2",ec="none",zorder=3))
    if n: ax.add_patch(FancyBboxPatch((bx,632),bw*n/8,50,boxstyle="round,pad=0,rounding_size=10",fc=RED,ec="none",zorder=4))
    wrapped="\u201c"+textwrap.fill(q,34)+"\u201d"
    top=440
    w,h=fbubble(x0,top,wrapped,29,"#ffffff","#3a3f47",lw=3,minw=cw-110)
    ax.add_patch(Polygon([(x0+180,top-2),(x0+215,top+40),(x0+250,top-2)],fc="white",ec="none",zorder=6))
    ax.plot([x0+180,x0+215,x0+250],[top,top+40,top],color="#3a3f47",lw=3,zorder=6,solid_joinstyle="miter")
T(60,85,"The direction points at the body part; the chat supplies what happened to the body part.",34,weight="semibold")
fig.savefig(H+"fig/wings.png",dpi=100,facecolor="white")
