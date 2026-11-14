import numpy as np
from colors import *; from paint import paint,SP; from render import render
Dd=data(); Y,_=pcs(Dd['q32']); Y[:,2]*=np.sign(np.corrcoef(Y[:,2],SP[:,1])[0,1])
th=np.radians([0,60,120,180,240,300])
pan=[dict(col=paint(to_rgb2(Y,t),smooth=10),ang=0,vp=(i*700,0,700,1150),hh=8.7) for i,t in enumerate(th)]
render(pan,'pal.png',4200,1150,bg='#ffffff')
