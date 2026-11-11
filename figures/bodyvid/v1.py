import os
import sys, numpy as np
from rend import *
g=np.load('geo.npz'); B=g['body']; Y=g['Y']; E=g['E']; cat=g['cat']; side=g['side']
cols=colours(cat,side); n=len(B)
T=34.0; N=int(T*FPS)
order={'head':0,'arm':1,'hand':2,'torso':1.5,'leg':2.5,'foot':3}
lag=np.array([order[c] for c in cat])/3*1.2 + (B[:,1].max()-B[:,1])/177*0.3
def morph_amt(t):
    if t<4.5: return np.zeros(n)
    if t<20: return smooth((t-4.5-lag)/5.5)
    return 1-smooth((t-20-lag)/5.5)
def fade(t,a,b,r=0.7): return float(np.clip(min((t-a)/r,(b-t)/r),0,1))
def pos(t):
    m=morph_amt(t)[:,None]; return B*(1-m)+Y*m, m.mean()
cB=(B.min(0)+B.max(0))/2; cY=(Y.min(0)+Y.max(0))/2
def frame(k):
    t=k/FPS; P,mm=pos(t); c=cB*(1-mm)+cY*mm
    yaw=np.radians(32*np.sin(2*np.pi*t/T*1.0)); pitch=np.radians(9)
    x,y,z,f=project(P,c,yaw,pitch,scale=4.4,off=(0,75))
    S=P.copy(); S[:,1]=P[:,1].min()-6; sx,sy,_,_=project(S,c,yaw,pitch,scale=4.4,off=(0,75))
    a=np.full(n,min(1,t/1.0)*min(1,(T-t)/0.8))
    ea=0.55*fade(t,13,T-0.8,2.5)
    tx=[("407 places on the body",W/2,128,78,fade(t,0.6,4.6)),
        ("the same places inside Qwen3-32B",W/2,128,78,fade(t,12.6,19.6))]
    return draw_frame(x,y,z,f,cols,np.full(n,6.5),a,1.0,edges=E,ealpha=ea,texts=tx,shadow=(sx,sy))
def control():  # real displacement from the body pose, for the sound
    return np.array([np.linalg.norm(pos(k/FPS)[0]-B,axis=1).mean() for k in range(N)])
if __name__=="__main__":
    if sys.argv[1]=='still':
        import PIL.Image as I
        for s in map(float,sys.argv[2:]): I.fromarray(frame(int(s*FPS))[:,:,:3]).save(f'v1_{s:04.1f}.png')
    elif sys.argv[1]=='seg': seg(frame,int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    else:
        np.save('v1_ctrl.npy',control()); encode(__file__,N,os.path.expandvars('$BODY_REPO/figures/out/body_video_morph'))
