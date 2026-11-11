import os
import sys, math, numpy as np
sys.path.insert(0,os.path.expandvars('$BODY_REPO/figures/figs5'))
from data import *
import mesh as M, render3d as R3
tasks=list(MS.AGENCY); a="do_L_arm_up"
T={b:load(f"told_N:{MS.AGENCY[b][0]}") for b in tasks}; mt=np.mean(list(T.values()),0)
D={b:np.mean([load(f"do:{b}:{k}") for k in range(3)],0) for b in tasks}; md=np.mean(list(D.values()),0)
def regv(Dx):
    d=np.linalg.norm(Dx,axis=1); return {g:float(d[REG==g].mean()/d.mean()) for g in GR}
vt=regv(T[a]-mt); vd=regv(D[a]-md); print(vt); print(vd)
LO,HI=1.0,1.4
STOPS=[(0,(0.93,0.88,0.84)),(0.35,(0.96,0.70,0.58)),(0.7,(0.86,0.30,0.25)),(1,(0.62,0.05,0.13))]
def cmap(t):
    t=np.clip(t,0,1); out=np.zeros((len(t),3))
    for (t0,c0),(t1,c1) in zip(STOPS[:-1],STOPS[1:]):
        m=(t>=t0)&(t<=t1); u=((t-t0)/(t1-t0))[m][:,None]; out[m]=np.array(c0)*(1-u)+np.array(c1)*u
    return out
def hexc(t): c=cmap(np.array([t]))[0]; return '#%02x%02x%02x'%tuple(int(round(x*255)) for x in c)
# ---- layout
W,H=2400,2140; FONT="'Inter','SF Pro Display','Helvetica Neue',Arial,sans-serif"
o=[]
def T_(x,y,s,size=50,w=400,fill="#111",anchor="start",extra=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')
def arrow(x1,y1,x2,y2,c="#111",w=7,L=30):
    ang=math.atan2(y2-y1,x2-x1); p1=(x2-L*math.cos(ang-0.5),y2-L*math.sin(ang-0.5)); p2=(x2-L*math.cos(ang+0.5),y2-L*math.sin(ang+0.5))
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2-0.7*L*math.cos(ang)}" y2="{y2-0.7*L*math.sin(ang)}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/><path d="M{x2},{y2} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{c}"/>')
NAVY="#1f3f73"; CARD="#edf1f7"; QC="#8a1c1c"
cw_=1150; cards=[(30,"TOLD the arm rose",["…Your left arm rises slowly until it is","stretched straight up over your head."],["My left arm is stretched straight","up over my head…"]),
                 (1220,"ASKED to raise the arm",["Go ahead and raise your left arm","over your head, and tell me what","you notice."],["I notice the movement of my left","arm lifting…"])]
cy=30; chh=470
for x,tag,u,q in cards:
    o.append(f'<rect x="{x}" y="{cy}" width="{cw_}" height="{chh}" rx="30" fill="{CARD}"/>')
    T_(x+40,cy+82,tag,62,800,NAVY)
    T_(x+40,cy+170,"User",46,700)
    for k,L in enumerate(u): T_(x+265,cy+170+58*k,L,46,400)
    T_(x+40,cy+360,"Qwen",46,700,QC)
    for k,L in enumerate(q): T_(x+265,cy+360+58*k,L,46,400,extra='font-style="italic"')
# touch band
by=chh+cy+60; bh=110; bx=110; bw=W-2*bx
for x,_,_,_ in cards: arrow(x+cw_/2,cy+chh+4,x+cw_/2,by-4)
o.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="30" fill="#fff" stroke="{NAVY}" stroke-width="5"/>')
T_(bx+40,by+65,"User",46,700); T_(bx+190,by+72,"“Someone presses a fingertip firmly into your <tspan font-weight='800'>left elbow</tspan>.”",48,400)
o.append(f'<rect x="{bx+bw-400}" y="{by+18}" width="370" height="74" rx="37" fill="{NAVY}"/>'); T_(bx+bw-215,by+70,"× 174 places",48,800,"#fff","middle")
# bodies
s=78.0; Yc=0.16; ch=int(16.9*s)+20; cy0=by+bh+40; cvs=(0,cy0,W,ch)
OFF=[-7.7,7.7]  # world units -> centres at 1200 -/+ 600
for k,x in enumerate([cards[0][0]+cw_/2,cards[1][0]+cw_/2]):
    arrow(x,by+bh+4,x,cy0+30)
def scr(p,ox): return W/2+(p[0]+ox)*s, cy0+ch/2-(p[1]-Yc)*s
for ox in OFF:
    hx,hy=scr(M.J['l-finger-3-4'],ox); T_(hx,hy+80,"left",48,700,"#111","middle")
    hx,hy=scr(M.J['r-finger-3-4'],ox); T_(hx,hy+80,"right",48,700,"#111","middle")
# key (centre column)
kx=W/2; ky=cy0+250; kh=520; kw=70
for j,(L,w) in enumerate([("Qwen3-8B",800),("layer-27 state",800),("change at each",400),("place, vs. average",400),("of 6 actions",400)]): T_(kx,ky-215+54*j,L,46,w,"#111","middle")
o.append('<defs><linearGradient id="kg" x1="0" y1="1" x2="0" y2="0">'+''.join(f'<stop offset="{t:.2f}" stop-color="{hexc(t)}"/>' for t in np.linspace(0,1,11))+'</linearGradient></defs>')
o.append(f'<rect x="{kx-kw/2}" y="{ky+20}" width="{kw}" height="{kh}" fill="url(#kg)" stroke="#111" stroke-width="3"/>')
T_(kx-kw/2-16,ky+60,"1.4×",46,700,"#111","end"); T_(kx-kw/2-16,ky+20+kh,"1×",46,700,"#111","end")
T_(kx+kw/2+16,ky+20+kh,"typical",46,400,"#111","start")
my=ky+kh+175
T_(kx,my,"0.46",120,800,NAVY,"middle")
T_(kx,my+62,"told vs. asked",46,700,"#111","middle"); T_(kx,my+116,"pattern match",46,700,"#111","middle")
T_(kx,my+180,"other actions:",46,400,"#111","middle"); T_(kx,my+234,"−0.10",46,700,"#111","middle")
T_(kx,my+320,"6 of 6 actions:",46,700,"#111","middle"); T_(kx,my+374,"best match is",46,400,"#111","middle"); T_(kx,my+428,"the same action",46,400,"#111","middle")
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}" style="position:absolute;left:0;top:0">'+'\n'.join(o)+'</svg>'
bodies=[(ox,cmap((M.field(v)-LO)/(HI-LO))**2.2) for ox,v in zip(OFF,[vt,vd])]
out=os.path.expandvars('$BODY_REPO/figures/out/body_act_vs_told_v2.png')
R3.page(W,H,svg,cvs,bodies,M.P,M.tri,out,Yc,s)
print(out)
