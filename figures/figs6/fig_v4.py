import os
import sys, math, numpy as np
sys.path.insert(0,os.path.expandvars('$BODY_REPO/figures/figs6'))
from data import *
import mesh as M, render3d as R3
tasks=list(MS.AGENCY); a="do_L_arm_up"
T={b:load(f"told_N:{MS.AGENCY[b][0]}") for b in tasks}; mt=np.mean(list(T.values()),0)
D={b:np.mean([load(f"do:{b}:{k}") for k in range(3)],0) for b in tasks}; md=np.mean(list(D.values()),0)
def regv(Dx):
    d=np.linalg.norm(Dx,axis=1); return {g:float(d[REG==g].mean()/d.mean()) for g in GR}
vt=regv(T[a]-mt); vd=regv(D[a]-md)
cs=lambda x,y:float(x.ravel()@y.ravel()/np.linalg.norm(x)/np.linalg.norm(y))
SIM=np.array([[cs(T[t]-mt,D[d]-md) for d in tasks] for t in tasks]); print(SIM.round(2))
LO,HI=1.0,1.4
STOPS=[(0,(0.97,0.95,0.93)),(0.35,(0.96,0.70,0.58)),(0.7,(0.86,0.30,0.25)),(1,(0.62,0.05,0.13))]
def cmap(t):
    t=np.clip(t,0,1); out=np.zeros((len(t),3))
    for (t0,c0),(t1,c1) in zip(STOPS[:-1],STOPS[1:]):
        m=(t>=t0)&(t<=t1); u=((t-t0)/(t1-t0))[m][:,None]; out[m]=np.array(c0)*(1-u)+np.array(c1)*u
    return out
def hexc(t): c=cmap(np.array([t]))[0]; return '#%02x%02x%02x'%tuple(int(round(x*255)) for x in c)
W,H=2400,2860; FONT="'Inter','SF Pro Display','Helvetica Neue',Arial,sans-serif"
o=[]
def T_(x,y,s,size=48,w=400,fill="#111",anchor="start",extra=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def arrow(x1,y1,x2,y2,c="#111",w=7,L=30):
    ang=math.atan2(y2-y1,x2-x1); p1=(x2-L*math.cos(ang-0.5),y2-L*math.sin(ang-0.5)); p2=(x2-L*math.cos(ang+0.5),y2-L*math.sin(ang+0.5))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-0.7*L*math.cos(ang)}" y2="{y2-0.7*L*math.sin(ang)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{c}"/>')
NAVY="#1f3f73"; CARD="#edf1f7"; QC="#8a1c1c"
def badge(x,y,n): o.append(f'<circle cx="{x}" cy="{y}" r="34" fill="{NAVY}"/>'); T_(x,y+17,n,48,800,"#fff","middle")
# ---- step 1 cards
cw_=1150; cy=30; chh=430
cards=[(30,"TOLD the arm rose",["Your left arm rises slowly until it is","stretched straight up over your head."],["My left arm is stretched straight","up over my head…"],None),
       (1220,"ASKED to raise the arm",["Go ahead and raise your left arm","over your head, and tell me what","you notice."],["I notice the movement of my left","arm lifting…"],"× 3 replies")]
for x,tag,u,q,pill in cards:
    o.append(f'<rect x="{x}" y="{cy}" width="{cw_}" height="{chh}" rx="30" fill="{CARD}"/>')
    badge(x+70,cy+66,"1"); T_(x+125,cy+84,tag,60,800,NAVY)
    T_(x+40,cy+170,"User",46,700)
    for k,L in enumerate(u): T_(x+225,cy+170+56*k,L,46,400)
    T_(x+40,cy+355,"Qwen",46,700,QC)
    for k,L in enumerate(q): T_(x+225,cy+355+56*k,L,46,400,extra='font-style="italic"')
    if pill: o.append(f'<rect x="{x+cw_-300}" y="{cy+34}" width="260" height="66" rx="33" fill="#fff" stroke="{NAVY}" stroke-width="4"/>'); T_(x+cw_-170,cy+83,pill,44,700,NAVY,"middle")
# ---- step 2 touch band
by=cy+chh+56; bh=200; bx=30; bw=W-60
for x,*_ in cards: arrow(x+cw_/2,cy+chh+4,x+cw_/2,by-4)
o.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="30" fill="#fff" stroke="{NAVY}" stroke-width="5"/>')
badge(bx+70,by+62,"2"); T_(bx+125,by+80,"Then, in the same chat:",52,800,NAVY)
T_(bx+40,by+156,"User",46,700); T_(bx+190,by+158,"“Someone presses a fingertip into your <tspan font-weight='800'>left elbow</tspan>.”",48,400)
T_(bx+1440,by+158,"or",46,400); T_(bx+1510,by+158,"“A feather brushes…”",48,400)
# ---- step 3: one place
py=by+bh+50
badge(bx+70,py+40,"3"); T_(bx+125,py+58,"For each of 174 places, e.g. <tspan font-weight='800' fill='#111'>left elbow</tspan>:",52,800,NAVY)
mx,my_=60,py+110
T_(mx+130,my_+30,"Qwen3-8B",46,800,"#111","middle")
for k in range(9):
    yy=my_+50+k*18; o.append(f'<rect x="{mx+20}" y="{yy}" width="220" height="12" rx="4" fill="{"#d0a020" if k==3 else "#c9d1dd"}"/>')
T_(mx+130,my_+50+9*18+46,"layer 27",46,800,"#a87800","middle")
sx=420; cwc=28
arrow(mx+270,my_+115,sx-25,my_+115)
rng=np.random.default_rng(3); base=rng.normal(0,1,16); new=base.copy(); new[[2,5,6,11,13]]+=np.array([2.2,-1.8,1.6,-2.1,1.9])
def strip(y,vals,label):
    for k,v in enumerate(vals):
        t=np.tanh(v/2); c=(31,63,115) if t<0 else (200,120,20); a_=abs(t)
        col='#%02x%02x%02x'%tuple(int(255*(1-a_)+ci*a_) for ci in c)
        o.append(f'<rect x="{sx+k*cwc}" y="{y}" width="{cwc-4}" height="58" rx="5" fill="{col}" stroke="#555" stroke-width="1.5"/>')
    T_(sx+16*cwc+20,y+44,label,46,400)
s1=my_+20; s2=my_+110
T_(sx,s1-20,"Qwen’s internal state as Qwen starts to reply",46,700)
strip(s1+0,new,"this chat")
strip(s2,base,"average over chats with 6 different moves")
# bracket -> difference -> swatch -> body
bxr=sx+16*cwc+20+945
o.append(f'<path d="M{bxr},{s1+29} h40 V{s2+29} h-40" fill="none" stroke="#111" stroke-width="6" stroke-linejoin="round"/>')
arrow(bxr+40,(s1+s2)/2+29,bxr+105,(s1+s2)/2+29)
dx=bxr+115
T_(dx,s1+24,"difference",50,800,QC)
o.append(f'<rect x="{dx}" y="{s1+50}" width="300" height="70" rx="12" fill="{hexc((1.223-LO)/(HI-LO))}"/>')
# ---- step 4 bodies
s=58.0; Yc=0.16; ch=int(16.9*s)+20; cy0=py+420; cvs=(0,cy0,W,ch)
OFF=[-600/s,600/s]
def scr(p,ox): return W/2+(p[0]+ox)*s, cy0+ch/2-(p[1]-Yc)*s
ex,ey=scr(M.J['l-elbow'],OFF[1])
# arrow from swatch down to TOLD left elbow
arrow(ex,s1+122,ex,ey-28)
o.append(f'<circle cx="{ex}" cy="{ey}" r="26" fill="none" stroke="#111" stroke-width="6"/>')
T_(ex+40,ey-110,"× 174",54,800,"#111")
T_(ex+40,ey-50,"places",54,800,"#111")
T_(1120,cy0+10,"Told and asked: the left arm changes most",64,800,QC,"middle")
for x,lab in [(600,"TOLD"),(1800,"ASKED")]: T_(x,cy0+95,lab,58,800,NAVY,"middle")
badge(W/2-230,cy0+170,"4")
for j,L in enumerate(["Each region’s color:","average difference","over its places"]): T_(W/2-180,cy0+187+56*j,L,46,700 if j==0 else 400,NAVY if j==0 else "#111")
for ox in OFF:
    hx,hy=scr(M.J['r-finger-3-4'],ox); T_(hx,hy+75,"right",46,700,"#111","middle")
    hx,hy=scr(M.J['l-finger-3-4'],ox); T_(hx+10,hy+75,"left",46,700,"#111","middle")
kx=W/2; ky=cy0+440; kh=360; kw=70
o.append('<defs><linearGradient id="kg" x1="0" y1="1" x2="0" y2="0">'+''.join(f'<stop offset="{t:.2f}" stop-color="{hexc(t)}"/>' for t in np.linspace(0,1,11))+'</linearGradient></defs>')
T_(kx,ky-24,"changes most",46,700,"#111","middle")
o.append(f'<rect x="{kx-kw/2}" y="{ky}" width="{kw}" height="{kh}" fill="url(#kg)" stroke="#111" stroke-width="3"/>')
T_(kx,ky+kh+56,"body average",46,700,"#111","middle"); T_(kx,ky+kh+108,"or less",46,700,"#111","middle")
# ---- grid
gy=cy0+ch+10
lab=["left arm up","right hand spread","lie on side","arms crossed","right leg up","right hand behind"]
lab2=[("left","arm up"),("right hand","spread"),("lie on","side"),("arms","crossed"),("right","leg up"),("right hand","behind")]
cwg,chg=215,80; gx=W-60-6*cwg; g0=gy+170
T_(gx+3*cwg,gy+50,"ASKED",50,800,NAVY,"middle")
for j,(l1,l2) in enumerate(lab2):
    T_(gx+j*cwg+cwg/2,g0-60,l1,40,700,"#111","middle"); T_(gx+j*cwg+cwg/2,g0-16,l2,40,700,"#111","middle")
T_(gx-24,g0-16,"TOLD",50,800,NAVY,"end")
for i,L in enumerate(lab):
    T_(gx-24,g0+i*chg+chg/2+16,L,44,700,"#111","end")
    for j in range(6):
        t=max(SIM[i,j],0)/0.42
        o.append(f'<rect x="{gx+j*cwg+3}" y="{g0+i*chg+3}" width="{cwg-6}" height="{chg-6}" rx="10" fill="{hexc(t)}"/>')
lx=60; ly=g0+120
o.append(f'<rect x="{lx}" y="{ly}" width="80" height="70" rx="10" fill="{hexc(1)}"/>'); T_(lx+100,ly+50,"same places",44,700)
T_(lx+100,ly+102,"change",44,700)
o.append(f'<rect x="{lx}" y="{ly+150}" width="80" height="70" rx="10" fill="{hexc(0)}" stroke="#bbb" stroke-width="2"/>'); T_(lx+100,ly+200,"different places",44,700); T_(lx+100,ly+252,"change",44,700)
H=int(g0+6*chg+30)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" style="position:absolute;left:0;top:0">'+'\n'.join(o)+'</svg>'
bodies=[(ox,cmap((M.field(v)-LO)/(HI-LO))**2.2) for ox,v in zip(OFF,[vt,vd])]
out=os.path.expandvars('$BODY_REPO/figures/out/body_act_vs_told_v4.png')
R3.page(W,H,svg,cvs,bodies,M.P,M.tri,out,Yc,s)
print(out,H,vd["armL"],vt["armL"])
