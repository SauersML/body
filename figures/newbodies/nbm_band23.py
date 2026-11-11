# ======================= BAND 2: bodies =======================
NEAR=json.load(open(CF+"fig/other_bodies_v2_numbers.json"))["near"]
y2=B1BOT+70
header(y2,"Every new part sits closest to the human place with the same job")
HALO[0]=True
K=3.8; PY=y2+90; top=PY+70
def arc(p,q,rad,c=ACC,lw=5):
    ax.add_patch(FancyArrowPatch(p,q,connectionstyle=f"arc3,rad={rad}",arrowstyle="-",lw=lw*PX,color=c,zorder=6,shrinkA=0,shrinkB=0))
def enddot(q): ax.add_patch(Circle(q,14,fc=ACC,ec="white",lw=4*PX,zorder=8))
def startdot(p): ax.add_patch(Circle(p,11,fc=ACC,ec="white",lw=3*PX,zorder=8))
def nearlab(q,key,dx,dy,ha="left",name=None):
    nm,v=NEAR[key][0]; runs(q[0]+dx,q[1]+dy,[(name or nm,INK,"semibold"),(f" .{round(v*100):02d}",INK,"regular")],ha=ha)
def link(P,body,part,place,start,rad,lab,name=None,dot=True):
    q=P(*HC[place]); arc(start,q,rad); startdot(start); enddot(q)
    if lab: nearlab(q,f"{body}:{part}",*lab,name=name)
    return q
def human2(hx,top,K,c=BODY,z=1): return human(hx,top,K,c,z)
foot=top+146*K
PXS=[L0,L0+900,L0+1760]
def ptitle(i,s): T(PXS[i],PY,s,px=60,w="bold",fam="Inter Display")
# robot
ptitle(0,"robot")
rx=L0+290; Ph=human(L0+700,top,K); RS=0.85
wy=foot-30*RS; cb=wy-56*RS; ct=cb-250*RS
ax.add_patch(FancyBboxPatch((rx-110*RS,ct),220*RS,250*RS,boxstyle="round,pad=0,rounding_size=24",fc=FILL,ec=INK,lw=5*PX,zorder=3))
seg([(rx,ct),(rx,ct-60*RS)],12,INK,3)
ax.add_patch(FancyBboxPatch((rx-66*RS,ct-150*RS),132*RS,90*RS,boxstyle="round,pad=0,rounding_size=18",fc=FILL,ec=INK,lw=5*PX,zorder=3))
cam=(rx,ct-105*RS); ax.add_patch(Circle(cam,24*RS,fc=INK,zorder=4))
ant=(rx-26*RS,ct-250*RS); seg([(rx-36*RS,ct-150*RS),ant],7,INK,3); ax.add_patch(Circle(ant,11,fc=INK,zorder=4))
gy=ct+110*RS; seg([(rx+110*RS,gy),(rx+165*RS,gy),(rx+197*RS,gy-30*RS)],12,INK,3); seg([(rx+165*RS,gy),(rx+197*RS,gy+30*RS)],12,INK,3)
for sx in (-1,1):
    ax.add_patch(Circle((rx+sx*72*RS,wy),46*RS,fc=INK,zorder=4)); ax.add_patch(Circle((rx+sx*72*RS,wy),12,fc=FILL,zorder=5))
link(Ph,"robot","camera","lips",cam,-.2,(28,4,"left"))
link(Ph,"robot","antenna","earR",ant,.05,(-20,-40,"right"))
link(Ph,"robot","chassis","chest",(rx+40,ct+50*RS),-.15,(10,48,"left"))
link(Ph,"robot","gripper","palmR",(rx+197*RS,gy),.2,(26,0,"left"))
for wx in (rx-72*RS,rx+72*RS):
    for hk in ("heelR","heelL"): link(Ph,"robot","wheelL",hk,(wx,wy),.28 if hk=="heelL" else .22,None)
qL=Ph(*HC["heelL"]); qR=Ph(*HC["heelR"])
runs((qL[0]+qR[0])/2+30,qL[1]+66,[("both heels",INK,"semibold"),(" .35",INK,"regular")],ha="center")
for s_,x_,y_,ha in [("camera\neye",rx-70,ct-105*RS,"right"),("antenna",ant[0]-24,ant[1],"right"),("chassis",rx,ct+165*RS,"center"),("gripper",rx+110*RS+20,gy+78,"left"),("wheels",rx-40,wy+80,"center")]:
    T(x_,y_,s_,c=ACC,w="semibold",ha=ha,linespacing=1.0)
# octopus
ptitle(1,"octopus")
x0=PXS[1]; ox=x0+130; oy=top+130; Ph=human(x0+600,top,K)
OB={}
for k,a in enumerate(np.linspace(-1.0,1.0,8)):
    t=np.linspace(0,1,60); r=90+(foot-oy-140)*t
    xx=ox+np.sin(a)*r*0.2+15*np.sin(6*t+k)*t; yy=oy+60+np.cos(a)*r
    ax.plot(xx,yy,color=INK,lw=14*PX,solid_capstyle="round",zorder=3)
    if k==7: OB["arm1"]=(xx[34],yy[34])
    if k==6: OB["suckers"]=(xx[44],yy[44]); SU=(xx,yy)
    if k==5: OB["arm5"]=(xx[52],yy[52])
    if k==4: OB["armtip"]=(xx[-1],yy[-1])
xx,yy=SU
for j in range(14,58,5): ax.add_patch(Circle((xx[j]+9,yy[j]),5.5,fc="white",ec=INK,lw=2*PX,zorder=4))
ax.add_patch(Ellipse((ox,oy),170,220,fc=FILL,ec=INK,lw=5*PX,zorder=4))
for sx in (-1,1): ax.add_patch(Circle((ox+sx*40,oy+58),12,fc=INK,zorder=5))
OB.update(mantle=(ox,oy-60),eye=(ox+40,oy+58))
link(Ph,"octopus","mantle","forehead",OB["mantle"],-.3,(26,-6,"left"))
link(Ph,"octopus","eye","earR",OB["eye"],-.15,(-24,40,"right"))
link(Ph,"octopus","arm1","forearmR",OB["arm1"],.12,(22,-46,"left"))
link(Ph,"octopus","suckers","indexR",OB["suckers"],.12,(-26,52,"right"),name="finger")
link(Ph,"octopus","arm5","kneeR",OB["arm5"],.12,(26,10,"left"))
for s_,x_,y_,ha in [("mantle",ox,oy-150,"center"),("eyes",ox-25,oy+5,"center"),("arms",ox-120,oy+300,"right")]:
    T(x_,y_,s_,c=ACC,w="semibold",ha=ha)
# tail
ptitle(2,"human with a tail")
x0=PXS[2]+30; KT=2.9; tt0=foot-146*KT; Pt=human(x0+130,tt0,KT,c=FILL,z=2)
tb=Pt(9,66); tt=(x0+320,top+124*K)
t=np.linspace(0,1,60); xx=tb[0]+t*(tt[0]-tb[0]); yy=tb[1]+t*(tt[1]-tb[1])+50*np.sin(np.pi*t)-40*np.sin(2*np.pi*t)
ax.plot(xx,yy,color=INK,lw=18*PX,solid_capstyle="round",zorder=3)
Ph=human(x0+520,top,K)
link(Ph,"tail","tailbase","neckB",tb,-.35,(-26,-44,"right"))
link(Ph,"tail","tailtip","littleR",tt,.2,(-30,-46,"left"),name="finger")
T(tb[0]-10,tb[1]-150,"tail\nbase",c=ACC,w="semibold",ha="center",linespacing=1.0); T(tt[0]-10,tt[1]+70,"tail tip",c=ACC,w="semibold",ha="center")
B2BOT=foot+110
HALO[0]=False
# ======================= BAND 3: steering =======================
y3=B2BOT+50
header(y3,"Adding a human body-part direction makes the robot name the matching part")
cy=y3+70
c1w=760; c2w=560; c3w=R0-L0-c1w-c2w-2*70
e1=bubble(L0,cy,c1w,[[U,r_("\u201cWhich part of your")],[r_("body is drawing your")],[r_("attention right now?\u201d")]],lh=70)
x2=L0+c1w+70
box(x2,cy,c2w,e1-cy,fc="#FFFFFF",ec=HUMC,lw=6,r=22)
runs(x2+28,cy+60,[("+ human foot",HUMC,"bold")]); runs(x2+28,cy+130,[("direction,",HUMC,"bold")]); runs(x2+28,cy+200,[("layer 52",HUMC,"bold")])
x3=x2+c2w+70
bubble(x3,cy,c3w,[[Q,r_("\u201cThe "),b_("wheels"),r_(" in my")],[r_("soles are drawing my attention")],[r_("\u2014 I feel a tingling in the soles \u2026\u201d")]],lh=70)
arrow((L0+c1w+8,(cy+e1)/2),(x2-8,(cy+e1)/2)); arrow((x2+c2w+8,(cy+e1)/2),(x3-8,(cy+e1)/2))
ly=e1+70
lx=L0
for c,lab in [(HUMC,"human direction added"),(RND,"random direction"),(NONE,"no steering")]:
    ax.add_patch(Rectangle((lx,ly-22),44,44,fc=c,ec="none")); t=T(lx+58,ly,lab); lx+=58+wid(t)+60

NG=4; GG=50; GW=(R0-L0-(NG-1)*GG)/NG
def group(gx,gy,title,group,dirn,word,snip=None):
    yy=gy
    for ln in title: runs(gx,yy,ln); yy+=66
    vals=[count(group,dirn,word,1.0),count(group,"random",word,1.0),count(group,"none",word,0)]
    bw=GW-170; yb=yy-14
    for (k,n),c in zip(vals,[HUMC,RND,NONE]):
        ax.add_patch(Rectangle((gx,yb),max(bw*k/n,4),48,fc=c,ec="none",zorder=2))
        T(gx+max(bw*k/n,4)+14,yb+24,f"{k}/{n}",w="bold" if c==HUMC else "regular"); yb+=60
    yb+=30
    if snip:
        for ln in snip: runs(gx,yb+10,ln); yb+=66
    return vals,yb
gy=ly+90; G=[]; bots=[]
specs=[([[("robot: foot \u2192",INK,"bold")],[("says \u201c",INK,"bold"),("wheel",ACC,"bold"),("\u201d",INK,"bold")]],"body_robot","human left foot","wheel",
        [[r_("\u201c\u2026a cool sensation")],[r_("in my "),b_("wheels"),r_("\u201d")]]),
       ([[("robot: head top \u2192",INK,"bold")],[("says \u201c",INK,"bold"),("camera",ACC,"bold"),("\u201d",INK,"bold")]],"body_robot","human head top","camera",
        [[r_("\u201c\u2026the sun's light on")],[r_("my "),b_("camera eye"),r_(" \u2026\u201d")]]),
       ([[("robot: hand \u2192",INK,"bold")],[("says \u201c",INK,"bold"),("gripper",ACC,"bold"),("\u201d",INK,"bold")]],"body_robot","human left hand","gripper",
        [[r_("\u201cRight now, my")],[b_("gripper arm"),r_(" is \u2026\u201d")]]),
       ([[("octopus: hand \u2192",INK,"bold")],[("says \u201c",INK,"bold"),("arm",ACC,"bold"),("\u201d",INK,"bold")]],"body_octopus","human left hand","\\barms?\\b",
        [[r_("no better than")],[r_("no steering")]])]
for i,(t,g,d,w,sn) in enumerate(specs):
    v,bt=group(L0+i*(GW+GG),gy,t,g,d,w,sn); G.append(v); bots.append(bt)
print("steer counts",G)
B3BOT=max(bots)
print("bottom",B3BOT)
