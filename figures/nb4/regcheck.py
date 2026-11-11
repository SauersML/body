exec(open('drive.py').read())
import numpy as np, trimesh
from PIL import Image, ImageDraw
yc=89.5; hh=91; Wv,Hv=1000,1000
def V(col,front,vp):
    s=1 if front else -1
    return dict(col=col,vp=vp,halfh=hh,eye=[0,yc,400*s],tgt=[0,yc,0],light=[-120*s,300,300*s])
shoot(dict(W=2*Wv,H=Hv,dpr=2,views=[V("plain",1,[0,0,Wv,Hv]),V("plain",0,[Wv,0,Wv,Hv])]),"regcheck.png")
M=np.load("neutral_mesh.npz"); body=trimesh.Trimesh(M['v'],M['f'],process=False)
pos=np.load("spot_pos_neutral.npy"); _,_,tid=trimesh.proximity.closest_point(body,pos); nrm=body.face_normals[tid]
im=Image.open("regcheck.png"); dr=ImageDraw.Draw(im); s=Hv/(2*hh)
for p,n in zip(pos,nrm):
    for k,sg in [(0,1),(1,-1)]:
        if n[2]*sg>0.05:
            x=k*Wv+Wv/2+sg*p[0]*s; y=Hv/2-(p[1]-yc)*s; dr.ellipse([x-4,y-4,x+4,y+4],fill=(200,30,30))
im.save("regcheck.png")
