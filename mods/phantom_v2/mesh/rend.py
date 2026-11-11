import numpy as np, sys; sys.path.insert(0,'.')
from paint import *; from render3d import *
from matplotlib.colors import LinearSegmentedColormap
Z=np.load('../v2_disp_L27.npz'); g=Z['gone'].mean(0); c=Z['ctrl'].mean(0)
CM=LinearSegmentedColormap.from_list('m',['#F2EDE8','#F7C6A3','#E8795A','#C0282D','#6B0A22'])
LO,HI=0.4,2.6
def col(v): return CM(np.clip((v-LO)/(HI-LO),0,1))[:,:3]
vg=smooth(paint(g),15); vc=smooth(paint(c),15)
write_mesh(); write_col('cg.bin',col(vg)); write_col('cc.bin',col(vc))
np.save('cmap_samples.npy',CM(np.linspace(0,1,256))[:,:3])
W,H=1500,1500
render(W,H,dict(views=[dict(col='cc.bin',x=-58),dict(col='cg.bin',x=58)],cam=dict(l=-118,r=118,t=184,b=-52)),'/tmp/b2.png')
# zoom on right hand of gone body: right hand at x=-46 mesh -> +58 shift
render(900,900,dict(views=[dict(col='cg.bin',x=0,ry=0.0)],cam=dict(l=-60,r=-30,t=125,b=95)),'/tmp/hz.png')
