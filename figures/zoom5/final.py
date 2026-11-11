import os
exec(open('drive.py').read()); exec(open('cams.py').read())
import sys
W,H=2600,2330
V=[]; html=[]
def T(x,y,s,size=48,w=400,anchor="start",col="#111",lh=1.1):
    tr={"start":"","middle":"transform:translateX(-50%);","end":"transform:translateX(-100%);"}[anchor]
    html.append(f'<div class="t" style="left:{x}px;top:{y}px;font-size:{size}px;font-weight:{w};color:{col};line-height:{lh};{tr}">{s}</div>')
def box(x,y,w,h,st): html.append(f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;{st}"></div>')
def view(reg,src,vp,k,dots):
    c=dict(CAM[reg][k]); V.append(dict(reg=reg,src=src,vp=vp,dots=dots,**c))
BLUE,SAND="#23305f","#c9ab84"
def bars(reg,x,y):
    for k,(nm,v,col) in enumerate([("Qwen3-32B",S[reg]["r32"],BLUE),("words alone",S[reg]["rw"],SAND)]):
        yy=y+k*122; full=330; w=full*max(v,0)
        T(x,yy,nm,48,700 if k==0 else 400)
        box(x,yy+60,full,44,"background:#eeeeee;border-radius:3px")
        box(x,yy+60,w,44,f"background:{col};border-radius:3px")
        T(x+w+12,yy+56,f"{v:.2f}",48,700)
# ---- title
T(40,24,"Up close, Qwen3-32B maps face and foot more like the skin than the sentence words do,<br>and hand and forearm less",56,700,lh=1.13)
# ---- method strip
MY=190
html.append(f'<div class="t" style="left:40px;top:{MY}px;font-size:48px;padding:16px 26px;background:#f1f1f1;border-radius:26px">A mosquito bites <b>the tip of your left index finger</b>.</div>')
T(1222,MY+14,"→",60,400)
box(1290,MY+2,510,92,f"background:{BLUE};border-radius:14px")
T(1545,MY+18,"Qwen3-32B, layer 53",48,700,"middle","#fff")
T(1812,MY+14,"→",60,400)
html.append(f'<div style="position:absolute;left:1888px;top:{MY+14}px;width:68px;height:68px;border-radius:50%;background:conic-gradient(from 90deg,#f19a9a,#e7b46d,#a9c66d,#5ccaa5,#5bb7e3,#9aa3f3,#d79be0,#f19a9a)"></div>')
T(1972,MY-2,"same colour on the skin<br>= similar internal state",48,400,lh=1.1)
T(40,MY+118,"Bars: how well closeness inside the model matches closeness on the skin (0 = none, 1 = perfect)",48,400)
TOP=400
def head(nm,x,y): T(x+40,y,nm,64,700)
# ---- face (left top)
cx,cy=0,TOP
head("face",cx,cy)
view("face","m",[cx+30,cy+70,820,840],0,0.16)
view("face","w",[cx+870,cy+70,400,410],0,0.16)
T(cx+885,cy+490,"Words of the<br>sentences alone",48,400,lh=1.08)
bars("face",cx+885,cy+640)
# ---- foot (right top)
cx,cy=1300,TOP
head("foot",cx,cy)
view("foot","m",[cx+20,cy+70,840,420],0,0.22); view("foot","m",[cx+20,cy+490,840,420],1,0.22)
T(cx+840,cy+92,"inner side",48,400,"end"); T(cx+60,cy+512,"outer side",48)
view("foot","w",[cx+870,cy+70,400,205],0,0.22); view("foot","w",[cx+870,cy+280,400,205],1,0.22)
T(cx+885,cy+490,"Words of the<br>sentences alone",48,400,lh=1.08)
bars("foot",cx+885,cy+640)
# ---- hand (left bottom)
B=TOP+960
cx,cy=0,B
head("hand",cx,cy)
view("hand","m",[cx+10,cy+70,425,820],0,0.14); view("hand","m",[cx+440,cy+70,425,820],1,0.14)
T(cx+60,cy+82,"palm",48); T(cx+470,cy+82,"back",48)
view("hand","w",[cx+870,cy+70,200,390],0,0.14); view("hand","w",[cx+1075,cy+70,200,390],1,0.14)
T(cx+885,cy+490,"Words of the<br>sentences alone",48,400,lh=1.08)
bars("hand",cx+885,cy+640)
# ---- forearm (right bottom)
cx,cy=1300,B
head("forearm",cx,cy)
view("forearm","m",[cx+20,cy+130,840,330],0,0.2); view("forearm","m",[cx+20,cy+540,840,330],1,0.2)
T(cx+60,cy+80,"palm side",48); T(cx+60,cy+470,"back",48)
T(cx+40,cy+870,"elbow",48); T(cx+850,cy+870,"wrist",48,400,"end")
view("forearm","w",[cx+870,cy+110,400,157],0,0.2); view("forearm","w",[cx+870,cy+290,400,157],1,0.2)
T(cx+885,cy+490,"Words of the<br>sentences alone",48,400,lh=1.08)
bars("forearm",cx+885,cy+640)
out=sys.argv[1] if len(sys.argv)>1 else os.path.expandvars("$BODY_REPO/figures/out/body_zoomins_v3.png")
shoot(dict(key=2.4,hemi=0.22,env=0.3,W=W,H=H,dpr=2,views=V,html="".join(html)),out); print("ok",out)
