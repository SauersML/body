import os
import numpy as np, sys; sys.path.insert(0,'.')
from paint import *; from render3d import *
from matplotlib.colors import LinearSegmentedColormap
Z=np.load(os.path.expandvars('$BODY_REPO/mods/phantom_v2/v2_disp_L27.npz')); g=Z['gone'].mean(0); c=Z['ctrl'].mean(0)
CM=LinearSegmentedColormap.from_list('m',['#F3EEE9','#F6C9A8','#EC8A5E','#C9302C','#7A0B24'])
LO,HI=0.6,2.2
def col(v): return CM(np.clip((v-LO)/(HI-LO),0,1))[:,:3]
vg=smooth(paint(g),10); vc=smooth(paint(c),10)
write_mesh(); write_col('cg.bin',col(vg)); write_col('cc.bin',col(vc))
th=float(sys.argv[1]) if len(sys.argv)>1 else 0.9
cs,sn=np.cos(th),np.sin(th); xp=lambda x,z: x*cs+z*sn
W,H=1220,820; y0,y1=96,142; h=y1-y0; w=h*W/H; cx=xp(-40,14)+4
cam=dict(l=cx-w/2,r=cx+w/2,t=y1,b=y0)
pts=[(-46.3,114.1,18.9)]
