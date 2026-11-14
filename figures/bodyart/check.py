import numpy as np
from paint import *; from render import render
cm={'head':(.9,.9,.9),'neck':(.6,.6,.6),'torso':(.8,.7,.3),'armL':(.9,.2,.2),'armR':(.2,.4,.9),'handL':(1,.6,.6),'handR':(.6,.8,1),'legL':(.2,.8,.3),'legR':(.6,.3,.8),'footL':(.5,1,.5),'footR':(.9,.6,1)}
V=np.array([cm[r] for r in SR]); zf=np.where(SP[:,2]>0,1.0,0.35)[:,None]; V=V*zf
yb=((SP[:,1]//10)%2)[:,None]*0.0
col=paint(V,smooth=4)
W=2400;H=1600
render([dict(col=col,ang=0,vp=(0,0,800,H)),dict(col=col,ang=np.pi,vp=(800,0,800,H)),dict(col=col,ang=np.pi/2,vp=(1600,0,800,H))],'check.png',W,H)
