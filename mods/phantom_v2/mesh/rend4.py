import os
import numpy as np, sys; sys.path.insert(0,'.')
from render3d import *
OUT=os.path.expandvars('$BODY_REPO/mods/phantom_v2/mesh/')
th=float(sys.argv[1]); cs,sn=np.cos(th),np.sin(th); xp=lambda x,z: x*cs+z*sn; zp=lambda x,z:-x*sn+z*cs
hx,hz=-48,24; print('hand zp',zp(hx,hz),'torso zp approx',zp(-15,0))
W=H=800; y0,y1=94,128; h=y1-y0; cx=xp(-45,20)
for nm in ['cg','cc']:
    render(W,H,dict(views=[dict(col=nm+'.bin',x=0,ry=th)],cam=dict(l=cx-h/2,r=cx+h/2,t=y1,b=y0,near=-500,far=100-float(sys.argv[2]))),OUT+f'{nm}_hand.png')
# full bodies
W2,H2=1000,1500
for nm in ['cg','cc']:
    render(W2,H2,dict(views=[dict(col=nm+'.bin',x=0,ry=0.3)],cam=dict(l=-62,r=62,t=184,b=-2)),OUT+f'{nm}_body.png')
