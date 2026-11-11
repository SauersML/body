import numpy as np, sys; sys.path.insert(0,'.')
from paint import *; from render3d import *
exec(open('../a1.py').read().split("GN=")[0])
X,_=load('told_N:R_hand_gone'); Sh,_=load('sham:R_hand_gone'); d=np.linalg.norm(X-Sh,axis=1)/unit
v=paint(d); print(v.min(),v.max())
import matplotlib; cm=matplotlib.colormaps['magma_r']
rgb=cm(np.clip((v-0.3)/2.0,0,1))[:,:3]
write_mesh(); write_col('c0.bin',rgb)
render(1400,1600,dict(views=[dict(col='c0.bin',x=0)],cam=dict(l=-60,r=60,t=185,b=-5)),'/tmp/t0.png')
