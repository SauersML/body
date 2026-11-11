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
W,H=2400,3100; FONT="'Inter','SF Pro Display','Helvetica Neue',Arial,sans-serif"
o=[]
def T_(x,y,s,size=48,w=400,fill="#111",anchor="start",extra=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def arrow(x1,y1,x2,y2,c="#111",w=7,L=30):
    ang=math.atan2(y2-y1,x2-x1); p1=(x2-L*math.cos(ang-0.5),y2-L*math.sin(ang-0.5)); p2=(x2-L*math.cos(ang+0.5),y2-L*math.sin(ang+0.5))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-0.7*L*math.cos(ang)}" y2="{y2-0.7*L*math.sin(ang)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{c}"/>')
NAVY="#1f3f73"; CARD="#edf1f7"; QC="#8a1c1c"
def badge(x,y,n): o.append(f'<circle cx="{x}" cy="{y}" r="34" fill="{NAVY}"/>'); T_(x,y+17,n,48,800,"#fff","middle")
# ---- header
T_(W/2,92,"Told the left arm rose, or asked to raise the left arm:",66,800,"#111","middle")
T_(W/2,174,"Qwen’s internal map of the body changes at the left arm in both",66,800,"#111","middle")
# ---- step 1 cards
cw_=1150; cy=230; chh=440
cards=[(30,"TOLD the arm rose",["Your left arm rises slowly until it is","stretched straight up over your head."],["My left arm is stretched straight","up over my head…"],None),
       (1220,"ASKED to raise the arm",["Go ahead and raise your left arm","over your head, and tell me what","you notice."],["I notice the movement of my left","arm lifting…"],"× 3 replies")]
for x,tag,u,q,pill in cards:
    o.append(f'<rect x="{x}" y="{cy}" width="{cw_}" height="{chh}" rx="30" fill="{CARD}"/>')
    badge(x+70,cy+66,"1"); T_(x+125,cy+84,tag,60,800,NAVY)
    T_(x+40,cy+170,"User",46,700)
    for k,L in enumerate(u): T_(x+225,cy+170+56*k,L,46,400)
    T_(x+40,cy+360,"Qwen",46,700,QC)
    for k,L in enumerate(q): T_(x+225,cy+360+56*k,L,46,400,extra='font-style="italic"')
    if pill: o.append(f'<rect x="{x+cw_-300}" y="{cy+34}" width="260" height="66" rx="33" fill="#fff" stroke="{NAVY}" stroke-width="4"/>'); T_(x+cw_-170,cy+83,pill,44,700,NAVY,"middle")
# ---- step 2 touch band
by=cy+chh+60; bh=250; bx=30; bw=W-60
for x,*_ in cards: arrow(x+cw_/2,cy+chh+4,x+cw_/2,by-4)
o.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="30" fill="#fff" stroke="{NAVY}" stroke-width="5"/>')
badge(bx+70,by+62,"2"); T_(bx+125,by+80,"Then, in the same chat:",52,800,NAVY)
T_(bx+40,by+148,"User",46,700); T_(bx+190,by+150,"“Someone presses a fingertip into your <tspan font-weight='800'>left elbow</tspan>.”",48,400)
T_(bx+190,by+214,"or",46,400); T_(bx+260,by+214,"“A feather brushes across your <tspan font-weight='800'>left elbow</tspan>.”",48,400)
o.append(f'<rect x="{bx+bw-640}" y="{by+28}" width="600" height="72" rx="36" fill="{NAVY}"/>'); T_(bx+bw-340,by+79,"repeat for 174 places",46,800,"#fff","middle")
# ---- step 3 read + compare
py=by+bh+50; ph=300
badge(bx+70,py+40,"3")
# model stack
mx,my_=150,py+20
T_(mx+130,my_+40,"Qwen3-8B",46,800,"#111","middle")
nl=9
for k in range(nl):
    yy=my_+60+k*18; hl=(k==3)
    o.append(f'<rect x="{mx+20}" y="{yy}" width="220" height="12" rx="4" fill="{"#d0a020" if hl else "#c9d1dd"}"/>')
T_(mx+130,my_+60+nl*18+48,"layer 27",46,800,"#a87800","middle")
arrow(mx+270,my_+120,mx+380,my_+120)
# two state strips
sx=560; rng=np.random.default_rng(3); base=rng.normal(0,1,16); new=base.copy(); new[[2,5,6,11,13]]+=np.array([2.2,-1.8,1.6,-2.1,1.9])
def strip(y,vals,label):
    for k,v in enumerate(vals):
        t=np.tanh(v/2); c=(31,63,115) if t<0 else (200,120,20); a=abs(t)
        col='#%02x%02x%02x'%tuple(int(255*(1-a)+ci*a) for ci in c)
        o.append(f'<rect x="{sx+k*34}" y="{y}" width="30" height="60" rx="5" fill="{col}" stroke="#555" stroke-width="1.5"/>')
    T_(sx+16*34+30,y+46,label,46,400)
T_(sx,py+22,"Qwen’s internal state as Qwen starts to reply to the “left elbow” touch:",46,700)
strip(py+60,new,"in this chat")
strip(py+150,base,"same place, averaged over chats with 6 different moves")
# difference bracket -> swatch
T_(sx,py+280,"Big difference  →  that place is dark red on the body",46,700,QC)
# ---- step 4 bodies
s=58.0; Yc=0.16; ch=int(16.9*s)+20; cy0=py+ph+40; cvs=(0,cy0,W,ch)
OFF=[-600/s,600/s]
badge(W/2-200,cy0+24,"4"); T_(W/2-145,cy0+41,"Color each place",52,800,NAVY)
for x,lab in [(600,"TOLD"),(1800,"ASKED")]: T_(x,cy0+40,lab,58,800,NAVY,"middle")
def scr(p,ox): return W/2+(p[0]+ox)*s, cy0+ch/2-(p[1]-Yc)*s
for ox in OFF:
    hx,hy=scr(M.J['l-finger-3-4'],ox); T_(hx,hy+75,"left",46,700,"#111","middle")
    hx,hy=scr(M.J['r-finger-3-4'],ox); T_(hx,hy+75,"right",46,700,"#111","middle")
kx=W/2; ky=cy0+150; kh=400; kw=70
o.append('<defs><linearGradient id="kg" x1="0" y1="1" x2="0" y2="0">'+''.join(f'<stop offset="{t:.2f}" stop-color="{hexc(t)}"/>' for t in np.linspace(0,1,11))+'</linearGradient></defs>')
T_(kx,ky-20,"changed most",46,700,"#111","middle")
o.append(f'<rect x="{kx-kw/2}" y="{ky}" width="{kw}" height="{kh}" fill="url(#kg)" stroke="#111" stroke-width="3"/>')
T_(kx,ky+kh+56,"no more than",46,700,"#111","middle"); T_(kx,ky+kh+108,"after other moves",46,700,"#111","middle")
for j,L in enumerate(["Color shows Qwen’s","internal state,","not a report","of feeling"]): T_(kx,ky+kh+210+54*j,L,46,400 if j<2 else 700,"#111","middle")
# ---- grid
gy=cy0+ch+10
T_(W/2,gy+50,"Same for all 6 moves: told and asked light up the same places",56,800,"#111","middle")
lab=["left arm up","right hand spread","lie on side","arms crossed","right leg up","right hand behind"]
lab2=[("left","arm up"),("right hand","spread"),("lie on","side"),("arms","crossed"),("right","leg up"),("right hand","behind")]
cwg,chg=215,80; gx=W-60-6*cwg; g0=gy+222
T_(gx+3*cwg,gy+112,"ASKED",46,800,NAVY,"middle")
for j,(l1,l2) in enumerate(lab2):
    T_(gx+j*cwg+cwg/2,g0-60,l1,40,700,"#111","middle"); T_(gx+j*cwg+cwg/2,g0-16,l2,40,700,"#111","middle")
T_(gx-24,g0-16,"TOLD",46,800,NAVY,"end")
for i,L in enumerate(lab):
    T_(gx-24,g0+i*chg+chg/2+16,L,44,700,"#111","end")
    for j in range(6):
        t=max(SIM[i,j],0)/0.42
        o.append(f'<rect x="{gx+j*cwg+3}" y="{g0+i*chg+3}" width="{cwg-6}" height="{chg-6}" rx="10" fill="{hexc(t)}"/>')
lx=60; ly=g0+250
o.append(f'<rect x="{lx}" y="{ly}" width="80" height="70" rx="10" fill="{hexc(1)}"/>'); T_(lx+100,ly+50,"same places",44,700)
o.append(f'<rect x="{lx}" y="{ly+100}" width="80" height="70" rx="10" fill="{hexc(0)}" stroke="#bbb" stroke-width="2"/>'); T_(lx+100,ly+150,"different places",44,700)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" style="position:absolute;left:0;top:0">'+'\n'.join(o)+'</svg>'
bodies=[(ox,cmap((M.field(v)-LO)/(HI-LO))**2.2) for ox,v in zip(OFF,[vt,vd])]
out=os.path.expandvars('$BODY_REPO/figures/out/body_act_vs_told_v3.png')
R3.page(W,H,svg,cvs,bodies,M.P,M.tri,out,Yc,s)
print(out, g0+6*chg)
