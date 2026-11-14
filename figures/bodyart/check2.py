import numpy as np
from paint import *; from render import render
V=np.zeros((len(SP),3)); V[:]=.5
hz=np.isin(SR,['handL','armL']); V[hz&(SP[:,2]>0.5)]=(1,.85,.2); V[hz&(SP[:,2]<-0.5)]=(.2,.4,1)
V[hz&(SP[:,1]>143.5)&(np.abs(SP[:,2])<=0.5)]=(1,.2,.2)   # thumb/up side
col=paint(V,smooth=3)
render([dict(col=col,ang=a,vp=(i*800,0,800,800),hh=2.2,yc=1.6,cx=c) for i,(a,c) in enumerate([(0,4.0),(np.pi,-4.0),(-np.pi/2,0)])],'check2.png',2400,800)
