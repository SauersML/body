import os
import sys, numpy as np
from rend import *
g=np.load('geo.npz'); Y=g['Y']; cat=g['cat']; side=g['side']; sd=g['sd']; err=g['err']; part=g['part']
cols=colours(cat,side); n=len(Y)
u0=np.median(Y[side==0,0]); u=Y[:,0]-u0; v=Y[:,1]; w=Y[:,2]
T=34.0; N=int(T*FPS)
twins=np.array([(i,part[i]) for i in range(n) if side[i]>0])
def fade(t,a,b,r=0.7): return float(np.clip(min((t-a)/r,(b-t)/r),0,1))
def opening(t): return smooth((t-8)/8)
def crisp(t): return smooth((t-20)/3.5)
def pos(t):
    o=opening(t); phi=np.where(sd<0, np.pi/2-np.pi*o, np.pi/2)
    x=np.where(side==0,u,u*np.sin(phi)); z=np.where(side==0,w,w+u*np.cos(phi))
    return np.c_[x,v,z]
def frame(k):
    t=k/FPS; P=pos(t); c=(P.min(0)+P.max(0))/2; c[2]=P[:,2].mean()
    yaw=np.radians(-6-34*fade(t,4.5,18,4.0)+5*np.sin(2*np.pi*t/T*2)); pitch=np.radians(10)
    x,y,z,f=project(P,c,yaw,pitch,scale=4.2,off=(0,60))
    S=P.copy(); S[:,1]=P[:,1].min()-6; sx,sy,_,_=project(S,c,yaw,pitch,scale=4.2,off=(0,60))
    cr=crisp(t)
    blur=cr*np.clip((err-4)*0.9,0,15); rad=6.5*(1+cr*np.clip((err-4)/10,0,1.4))
    a=np.full(n,min(1,t/1.0)*min(1,(T-t)/0.8))*(1-cr*np.clip((err-4)/16,0,0.65))
    ea=0.6*fade(t,6,18.5,1.5)
    tx=[("left and right lie on top of each other",W/2,128,74,fade(t,0.8,7.6)),
        ("sharpest: hands and face",W/2,128,78,fade(t,22.5,T-0.6))]
    return draw_frame(x,y,z,f,cols,rad,a,1.0,blur=blur,edges=twins,ealpha=ea,texts=tx,shadow=(sx,sy))
def control(): return np.array([opening(k/FPS)+0.6*crisp(k/FPS) for k in range(N)])
if __name__=="__main__":
    if sys.argv[1]=='still':
        import PIL.Image as I
        for s in map(float,sys.argv[2:]): I.fromarray(frame(int(s*FPS))[:,:,:3]).save(f'v2_{s:04.1f}.png')
    elif sys.argv[1]=='seg': seg(frame,int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    else:
        np.save('v2_ctrl.npy',control()); encode(__file__,N,os.path.expandvars('$BODY_REPO/figures/out/body_video_unfold'))
