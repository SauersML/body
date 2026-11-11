import numpy as np, sys; sys.path.insert(0,'.')
from render3d import *
for th in [0.6,1.2,1.57]:
    c,s=np.cos(th),np.sin(th); x,z=-46,22; xp=x*c+z*s
    render(900,900,dict(views=[dict(col='cg.bin',x=0,ry=th)],cam=dict(l=xp-22,r=xp+22,t=135,b=91)),f'/tmp/hz_{th}.png')
