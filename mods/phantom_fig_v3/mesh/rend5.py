import os
import numpy as np, sys; sys.path.insert(0,'.')
from render3d import *
OUT=os.path.expandvars('$BODY_REPO/mods/phantom_fig_v3/mesh/')
th=1.45; cs,sn=np.cos(th),np.sin(th); xp=lambda x,z: x*cs+z*sn
y0,y1=95,127; h=y1-y0; cx=xp(-46,22)
for nm in ['cg','cc']:
    render(700,700,dict(views=[dict(col=nm+'.bin',x=0,ry=th)],cam=dict(l=cx-h/2,r=cx+h/2,t=y1,b=y0,near=-500,far=80)),OUT+f'{nm}_hand.png')
    pr=render(760,1180,dict(views=[dict(col=nm+'.bin',x=0,ry=0.3)],cam=dict(l=-58,r=58,t=182,b=0)),OUT+f'{nm}_body.png',pts=[(-46.3,112,22)])
print(pr)
