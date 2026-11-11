import os
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Ellipse, Polygon
FD=os.path.expandvars("$BODY_WORK/fonts/inter/extras/ttf/")
for f in ["Inter-Regular","Inter-Medium","Inter-SemiBold","Inter-Bold","InterDisplay-SemiBold","InterDisplay-Bold","InterDisplay-Medium"]:
    fm.fontManager.addfont(FD+f+".ttf")
plt.rcParams.update({"font.family":"Inter","font.size":30})
INK="#1E2A3E"; MUTE="#6B7280"; BG="#FBF9F5"; CARD="#F1ECE4"; BODY="#DDD5C8"
CORAL="#D2603F"; CORAL2="#EDAA92"; GREY="#C8C3BA"; RULE="#D9D3C8"
SO=os.path.expandvars("$BODY_REPO/map/somato/")
near=json.load(open(SO+"out3_32b/v3_summary.json"))["bodies"]
sc=json.load(open(SO+"out3_32b/steer4_L52_scores.json"))
W,H=2600,2520
fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off"); ax.set_aspect("equal")
L,R=90,W-90
PT=0.72  # points per px
def T(x,y,s,size=30,c=INK,w="regular",ha="left",va="center",fam="Inter",**k):
    return ax.text(x,y,s,fontsize=size,color=c,weight=w,ha=ha,va=va,family=fam,**k)
def seg(pts,wpx,c=BODY,z=1):
    xs,ys=zip(*pts); ax.plot(xs,ys,color=c,lw=wpx*PT,solid_capstyle="round",solid_joinstyle="round",zorder=z)
def dot(x,y,r=13,c=CORAL,z=6): ax.add_patch(Circle((x,y),r,fc=c,ec=BG,lw=3,zorder=z))

