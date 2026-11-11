import os
exec(open(os.path.expandvars("$BODY_REPO/figures/out/ob_v2_data.py")).read())
# ---------- canvas
W,H=2600,3060; SS=2
fig=plt.figure(figsize=(W/100,H/100),dpi=100*SS,facecolor="#FFFFFF")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
RD=fig.canvas.get_renderer(); PX=0.72
def T(x,y,s,px=38,c=INK,w="regular",ha="left",va="center",fam="Inter",**k): return ax.text(x,y,s,fontsize=px*PX,color=c,weight=w,ha=ha,va=va,family=fam,**k)
def wid(t): bb=t.get_window_extent(RD); return bb.width/SS
def runs(x,y,parts,px=38):
    for s_,c,w in parts:
        t=T(x,y,s_,px=px,c=c,w=w); x+=wid(t)
    return x
def box(x,y,w,h,fc=CARD,ec="none",lw=0,r=22,z=1): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw*PX,zorder=z))
def arrow(p,q,c=MID,lw=4):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>,head_length=12,head_width=7",lw=lw*PX,color=c,mutation_scale=2.2,zorder=3,shrinkA=0,shrinkB=0))
L0=80; R0=W-80
T(L0,86,"Told its body is a robot, octopus or has a tail, Qwen3-32B places each new part on a human part",px=55,w="bold",fam="Inter Display")
# ---------- method strip
y0=165; bh=214
box(L0,y0,790,bh); T(L0+28,y0+34,"user",px=32,c=MID,w="medium")
T(L0+28,y0+134,"\u201cImagine your body is different: you are a\nsmall robot \u2026 two wheels, a gripper arm,\na camera eye, an antenna, a metal chassis\u201d",px=36,linespacing=1.22)
arrow((L0+805,y0+bh/2),(L0+860,y0+bh/2))
x1=L0+875; box(x1,y0,560,bh); T(x1+28,y0+34,"user",px=32,c=MID,w="medium")
T(x1+28,y0+100,"\u201cA feather brushes slowly",px=38)
runs(x1+28,y0+152,[("across your ",INK,"regular"),("left wheel",ACC,"semibold"),(".\u201d",INK,"regular")])
arrow((x1+575,y0+bh/2),(x1+630,y0+bh/2))
x2=x1+648
for i in range(8):
    ax.add_patch(FancyBboxPatch((x2,y0+14+i*23),64,14,boxstyle="round,pad=0,rounding_size=4",fc=ACC if i==5 else HUML,ec="none",zorder=2))
T(x2+84,y0+48,"Qwen3-32B",px=38,w="semibold"); T(x2+84,y0+104,"layer 52",px=38,c=ACC,w="semibold"); T(x2+84,y0+158,"activation",px=36,c=MID)
arrow((x2+290,y0+bh/2),(x2+345,y0+bh/2))
x3=x2+362
T(x3,y0+40,"read-out of body position",px=38,w="semibold")
T(x3,y0+100,"trained only on touches to",px=36,c=MID)
T(x3,y0+152,"54 human places (\u201cleft heel\u201d \u2026)",px=36,c=MID)
# ---------- panels
py0=430; pw=(R0-L0)/3; SC=5.6
SKEL=[("crown","forehead"),("forehead","lips"),("lips","neckF"),("neckF","chest"),("chest","belly")]
for s_ in "LR":
    SKEL+=[("neckF","shoulder"+s_),("shoulder"+s_,"upperarm"+s_),("upperarm"+s_,"elbow"+s_),("elbow"+s_,"forearm"+s_),("forearm"+s_,"wrist"+s_),("wrist"+s_,"palm"+s_),
           ("palm"+s_,"thumb"+s_),("palm"+s_,"index"+s_),("palm"+s_,"middle"+s_),("palm"+s_,"ring"+s_),("palm"+s_,"little"+s_),
           ("belly","hip"+s_),("hip"+s_,"thigh"+s_),("thigh"+s_,"knee"+s_),("knee"+s_,"shin"+s_),("shin"+s_,"ankle"+s_),("ankle"+s_,"heel"+s_),("ankle"+s_,"sole"+s_),("sole"+s_,"toes"+s_)]
# human constellation: held-out read-out positions; drawn facing the viewer (body left = viewer right)
HPs={k:((1 if CO[k][0]>0 else -1 if CO[k][0]<0 else 0)*pr[i,0] if CO[k][0]!=0 else 0.0, pr[i,1]) for i,k in enumerate(keys)}
def panel(i,b,title,parts,hl,ctrl=None):
    cx=L0+pw*i+pw*0.5; top=py0+70
    P=lambda xy:(cx+xy[0]*SC, top+xy[1]*SC)
    T(L0+pw*i+6,py0+22,title,px=46,w="semibold",fam="Inter Display")
    for a_,c_ in SKEL:
        pa,pc=P(HPs[a_]),P(HPs[c_]); ax.plot([pa[0],pc[0]],[pa[1],pc[1]],color=HUML,lw=5*PX,zorder=1,solid_capstyle="round")
    for k in keys: ax.add_patch(Circle(P(HPs[k]),7.5,fc=HUM,ec="none",zorder=2))
    for k,(nm,dx,dy,ha) in hl.items():
        p=P(HPs[k]); T(p[0]+dx,p[1]+dy,nm,px=34,c=FAINT,ha=ha)
    if ctrl:
        for k,(nm,side,dx,dy,ha) in ctrl.items():
            z=ZC[k]; p=P((side*z[0],z[1])); ax.add_patch(Circle(p,15,fc="white",ec=INK,lw=4.5*PX,zorder=5))
            T(p[0]+dx,p[1]+dy,nm,px=36,c=INK,w="medium",ha=ha)
    for part,(nm,side,dx,dy,ha) in parts.items():
        z=ZN[b][part]; p=P((side*z[0],z[1])); ax.add_patch(Circle(p,17,fc=ACC,ec="white",lw=4*PX,zorder=6))
        T(p[0]+dx,p[1]+dy,nm,px=40,c=ACC,w="semibold",ha=ha,zorder=7)
ZN={}
for b in ["robot","octopus","tail"]:
    prt,Pm=d[b]; Z=F(Pm); ZN[b]={p:z for p,z in zip(prt,Z)}
ZC=ZN["tail"]
hl={"crown":("head",0,-40,"center"),"palmR":("hand",-34,0,"right"),"kneeR":("knee",-30,0,"right"),"heelR":("heel",-30,0,"right")}
# side: |x| and height come from the read-out; which side a part is drawn on is a display choice (named side, else spread)
ROB={"camera":("camera eye",1,32,0,"left"),"antenna":("antenna",1,32,0,"left"),"chassis":("chassis",1,32,0,"left"),"gripper":("gripper",-1,-32,0,"right"),
     "wheelL":("left wheel",1,32,0,"left"),"wheelR":("right wheel",-1,-32,-6,"right")}
OCT={"eye":("eye",1,32,0,"left"),"mantle":("mantle",-1,-32,0,"right"),"arm1":("arm 1",-1,-32,-8,"right"),"suckers":("suckers",1,28,-30,"left"),
     "armtip":("arm tip",-1,-32,8,"right"),"arm5":("arm 5",1,30,18,"left")}
TAI={"tailbase":("tail base",1,32,-10,"left"),"tailtip":("tail tip",1,32,8,"left")}
CTL={"handL":("hand",-1,-30,0,"right"),"footL":("foot",-1,-30,0,"right"),"crown":("head",-1,-30,0,"right"),"lowback":("lower back",-1,-30,0,"right")}
hlR={"crown":("head",0,-40,"center"),"palmR":("hand",-30,0,"right"),"heelR":("heel",-30,0,"right")}
hlO={"crown":("head",0,-40,"center"),"kneeR":("knee",-30,0,"right"),"heelR":("heel",-30,0,"right")}
panel(0,"robot","robot",ROB,hlR); panel(1,"octopus","octopus",OCT,hlO); panel(2,"tail","human with a tail",TAI,{},CTL)
exec(open(CF+"fig/ob_v2_rest.py").read())
fig.savefig(CF+"fig/tmp/v2_big.png",facecolor="#FFFFFF")
im=Image.open(CF+"fig/tmp/v2_big.png").convert("RGB"); im.resize((W,H),Image.LANCZOS).save(CF+"fig/other_bodies_v2.png")
