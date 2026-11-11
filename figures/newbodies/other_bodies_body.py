import os
exec(open(os.path.expandvars("$BODY_REPO/figures/out/ob_head.py")).read())
T(L,92,"Imagining a new body, Qwen3 maps each new part onto a human part",size=52,w="bold",fam="Inter Display")
# ---- method line
T(L,190,"Qwen3-32B is told \u201cimagine your body is different\u201d, then reads touches such as \u201can ice cube touches your left wheel\u201d.",size=31)
T(L,240,"Each coral line joins a new part to the human place (of 54) whose layer-52 activation is closest.",size=31)

def human(hx,top,s=1.0,c=BODY,z=1):
    P=lambda x,y:(hx+x*s,top+y*s)
    seg([P(0,170),P(0,230)],60*s,c,z)
    seg([P(-150,270),P(150,270)],70*s,c,z)
    ax.add_patch(FancyBboxPatch(P(-120,250),240*s,380*s,boxstyle=f"round,pad=0,rounding_size={60*s}",fc=c,ec="none",zorder=z))
    for sg in (-1,1):
        seg([P(sg*160,280),P(sg*195,480),P(sg*215,660)],52*s,c,z)
        ax.add_patch(Ellipse(P(sg*222,720),62*s,100*s,fc=c,ec="none",zorder=z))
        seg([P(sg*62,600),P(sg*72,880),P(sg*78,1130)],74*s,c,z)
        seg([P(sg*78,1160),P(sg*120,1170)],46*s,c,z)
    ax.add_patch(Circle(P(0,95),88*s,fc=c,ec="none",zorder=z))
    return P
def places(P): return {
 "heelL":P(80,1165),"heelR":P(-80,1165),"palmR":P(-222,730),"palmL":P(222,730),"lips":P(0,145),
 "earR":P(-86,100),"earL":P(86,100),"chest":P(0,360),"forehead":P(0,35),"forearmR":P(-205,560),"forearmL":P(205,560),
 "kneeR":P(-72,880),"kneeL":P(72,880),"neckB":P(0,215),"littleL":P(238,770),"lowback":P(0,560)}
HN={"heelL":"heel","heelR":"heel","palmR":"palm","lips":"lips","earR":"ear","earL":"ear","chest":"chest",
    "forehead":"forehead","forearmR":"forearm","kneeR":"knee","neckB":"back of neck","littleL":"little finger"}
FILL="#E6E1D8"
def arc(p,q,rad):
    ax.add_patch(FancyArrowPatch(p,q,connectionstyle=f"arc3,rad={rad}",arrowstyle="-",lw=4.5,color=CORAL,zorder=7,shrinkA=0,shrinkB=0))
    dot(*p,r=12,z=8); dot(*q,r=14,z=8)

top=420; S=0.78; hs=1180*S
cols=[(L+20,"robot"),(L+830,"octopus"),(L+1620,"human with a tail")]
for x0,name in cols: T(x0,330,name,size=40,w="semibold")

def link(body,D,P,part,rad=0.0):
    h=near[body]["nearest_human"][part][0]; HP=places(P); arc(D[part],HP[h],rad); return h
used={}
def hlabels(P,keys,dx=40):
    HP=places(P)
    for k,(ddx,ddy,ha) in keys.items(): T(HP[k][0]+ddx,HP[k][1]+ddy,HN[k],size=29,c=INK,ha=ha)

# --- robot panel
x0=cols[0][0]; rx=x0+165
P=human(x0+630,top,S)
wy=top+1165*S-20; cb=wy-60; ct=cb-330      # chassis bottom/top
ax.add_patch(FancyBboxPatch((rx-130,ct),260,330,boxstyle="round,pad=0,rounding_size=30",fc=FILL,ec=INK,lw=4,zorder=3))
seg([(rx,ct),(rx,ct-90)],12,INK,3)
ax.add_patch(FancyBboxPatch((rx-75,ct-190),150,100,boxstyle="round,pad=0,rounding_size=24",fc=FILL,ec=INK,lw=4,zorder=3))
cam=(rx,ct-140); ax.add_patch(Circle(cam,30,fc=INK,zorder=4))
ant=(rx-30,ct-310); seg([(rx-40,ct-190),ant],6,INK,3); ax.add_patch(Circle(ant,12,fc=INK,zorder=4))
gy=ct+150; seg([(rx+130,gy),(rx+190,gy),(rx+225,gy-35)],12,INK,3); seg([(rx+190,gy),(rx+225,gy+35)],12,INK,3)
for sg in (-1,1):
    ax.add_patch(Circle((rx+sg*85,wy),55,fc=INK,zorder=4)); ax.add_patch(Circle((rx+sg*85,wy),15,fc=FILL,zorder=5))
RB=dict(camera=cam,antenna=ant,chassis=(rx,ct+170),gripper=(rx+225,gy),wheelR=(rx-85,wy),wheelL=(rx+85,wy))
for part,rad in [("camera",.0),("antenna",-.2),("chassis",.0),("gripper",.1),("wheelL",-.12),("wheelR",.14)]:
    used[("robot",part)]=link("robot",RB,P,part,rad)
T(rx-95,ct-140,"camera\neye",size=31,w="medium",ha="right",linespacing=1.1); T(ant[0]-30,ant[1],"antenna",size=31,w="medium",ha="right")
T(rx,ct+280,"chassis",size=31,w="medium",ha="center"); T(rx+225,gy+85,"gripper",size=31,w="medium",ha="center")
T(rx,wy+95,"wheels",size=31,w="medium",ha="center")
hlabels(P,{"lips":(80,5,"left"),"earR":(-30,-45,"right"),"chest":(0,-50,"center"),"palmR":(0,85,"center"),"heelR":(-10,70,"center"),"heelL":(10,70,"center")})
# --- octopus panel
x0=cols[1][0]; ox=x0+230; oy=top+170
P=human(x0+590,top,S)
angs=np.linspace(-1.0,1.0,8); OB={}
for k,a in enumerate(angs):
    t=np.linspace(0,1,60); r=90+(hs-280)*t
    xx=ox+np.sin(a)*r*0.3+16*np.sin(6*t+k)*t; yy=oy+60+np.cos(a)*r
    ax.plot(xx,yy,color=INK,lw=15*PT,solid_capstyle="round",zorder=3)
    if k==7: OB["arm1"]=(xx[36],yy[36])
    if k==5: OB["arm5"]=(xx[50],yy[50])
ax.add_patch(Ellipse((ox,oy),200,260,fc=FILL,ec=INK,lw=4,zorder=4))
for sg in (-1,1): ax.add_patch(Circle((ox+sg*45,oy+60),13,fc=INK,zorder=5))
OB.update(mantle=(ox,oy-70),eye=(ox+45,oy+60))
for part,rad in [("mantle",-.2),("eye",-.05),("arm1",-.1),("arm5",.1)]:
    used[("octopus",part)]=link("octopus",OB,P,part,rad)
T(ox,oy-170,"mantle",size=31,w="medium",ha="center"); T(ox-115,oy+60,"eye",size=31,w="medium",ha="right")
T(ox-150,top+1165*S-40,"arms",size=31,w="medium",ha="right")
hlabels(P,{"forehead":(0,-55,"center"),"earL":(20,0,"left"),"forearmR":(-20,-55,"center"),"kneeR":(0,-60,"center")})
# --- tail panel
x0=cols[2][0]
Pt=human(x0+180,top,S,c=FILL,z=2)
tb=Pt(60,600); tt=(x0+420,top+900*S)
t=np.linspace(0,1,60); xx=tb[0]+t*(tt[0]-tb[0]); yy=tb[1]+t*(tt[1]-tb[1])+70*np.sin(np.pi*t)-60*np.sin(2*np.pi*t)
ax.plot(xx,yy,color=INK,lw=20*PT,solid_capstyle="round",zorder=3)
P=human(x0+590,top,S)
TB=dict(tailbase=tb,tailtip=tt)
for part,rad in [("tailbase",-.3),("tailtip",.25)]:
    used[("tail",part)]=link("tail",TB,P,part,rad)
T(tt[0]-45,tt[1]+100,"tail tip",size=31,w="medium",ha="center"); T(tb[0]-60,tb[1]+80,"tail base",size=31,w="medium",ha="center")
hlabels(P,{"neckB":(0,-170,"center"),"littleL":(50,-80,"right")})
print(used)
exec(open(os.path.expandvars('$BODY_REPO/figures/out/ob_bottom.py')).read())
plt.savefig(os.path.expandvars("$BODY_REPO/figures/out/other_bodies.png"),dpi=100,facecolor=BG)
