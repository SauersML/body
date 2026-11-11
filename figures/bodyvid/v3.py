import os; os.environ.setdefault('OMP_NUM_THREADS','2')
import sys, numpy as np, torch, skia
torch.set_num_threads(2)
from rend import *
s=np.load('surf_dense.npz'); S=s['S'].astype(np.float64); N0=s['N'].astype(np.float64)
w=np.load('warp.npz'); Dv=w['Dv'].astype(np.float64); J=w['J'].astype(np.float64); COL=w['col'].astype(np.float64)
SV1=np.linalg.svd(np.eye(3)[None]+J,compute_uv=False)[:,0]
g=np.load('geo.npz'); X=g['body']; Yu=w['Yu']; scol=colours(g['cat'],g['side'])
T=32.0; NF=int(T*FPS); SS=2; WW,HH=W*SS,H*SS
def amt(t):
    if t<5: return 0.0
    if t<20: return smooth((t-5)/7)
    return 1-smooth((t-20)/7)
def fade(t,a,b,r=0.7): return float(np.clip(min((t-a)/r,(b-t)/r),0,1))
OFFS=np.array([[dx,dy] for dx in range(-4,5) for dy in range(-4,5) if dx*dx+dy*dy<=18.5])
OFFB=np.array([[dx,dy] for dx in range(-10,11) for dy in range(-10,11) if dx*dx+dy*dy<=100])
OFF=np.array([[dx,dy] for dx in range(-7,8) for dy in range(-7,8) if dx*dx+dy*dy<=50])
def splat(x,y,z,col,r_px):
    xi=np.round(x).astype(np.int64); yi=np.round(y).astype(np.int64)
    zb=torch.full((HH*WW,),-1e9,dtype=torch.float64); img=np.ones((HH*WW,3))
    big=r_px>4.2; chunks=[]
    for msk,O,cs in ((~big,OFFS,150000),(big,OFFB,15000)):
        ids=np.where(msk)[0]; dist=np.hypot(O[:,0],O[:,1])
        for i in range(0,len(ids),cs): chunks.append((ids[i:i+cs],O,dist))
    def expand(ids,O,dist):
        px=xi[ids,None]+O[None,:,0]; py=yi[ids,None]+O[None,:,1]
        ok=(dist[None,:]<=r_px[ids,None])&(px>=0)&(px<WW)&(py>=0)&(py<HH)
        return np.broadcast_to(ids[:,None],ok.shape)[ok], (py*WW+px)[ok]
    for c in chunks:
        pid,lin=expand(*c); zb.scatter_reduce_(0,torch.from_numpy(lin),torch.from_numpy(z[pid]),reduce='amax')
    zbn=zb.numpy()
    for c in chunks:
        pid,lin=expand(*c); win=z[pid]>=zbn[lin]-1e-9; img[lin[win]]=col[pid[win]]
    return img.reshape(HH,WW,3), zbn.reshape(HH,WW)
def splat_old(x,y,z,col,r_px):
    # z-buffer splat; r_px per point (in supersampled pixels)
    xi=np.round(x).astype(np.int64); yi=np.round(y).astype(np.int64)
    zb=torch.full((HH*WW,),-1e9,dtype=torch.float64); idx_all=[]; 
    keep=np.hypot(OFF[:,0],OFF[:,1])
    px=(xi[:,None]+OFF[None,:,0]); py=(yi[:,None]+OFF[None,:,1])
    ok=(keep[None,:]<=r_px[:,None])&(px>=0)&(px<WW)&(py>=0)&(py<HH)
    pid=np.broadcast_to(np.arange(len(x))[:,None],ok.shape)[ok]; lin=(py*WW+px)[ok]
    zz=torch.tensor(z[pid]); L=torch.tensor(lin)
    zb.scatter_reduce_(0,L,zz,reduce='amax')
    win=(zz>=zb[L]-1e-9)
    img=np.ones((HH*WW,3)); img[lin[win.numpy()]]=col[pid[win.numpy()]]
    return img.reshape(HH,WW,3), zb.numpy().reshape(HH,WW)
LD=np.array([-0.45,0.65,0.62]); LD/=np.linalg.norm(LD); RIM=np.array([0.7,0.2,-0.7]); RIM/=np.linalg.norm(RIM)
def frame(k):
    t=k/FPS; m=amt(t)
    P=S+m*Dv; Jm=np.eye(3)[None]+m*J
    Nn=np.linalg.solve(np.transpose(Jm,(0,2,1)),N0[...,None])[...,0]; Nn/=np.linalg.norm(Nn,axis=1,keepdims=True)+1e-12
    st=np.abs(np.linalg.det(Jm))**(1/3)   # local stretch -> bigger splats
    Q=X+m*(Yu-X); c=(P.min(0)+P.max(0))/2
    yaw=np.radians(-28*np.sin(2*np.pi*t/T)+0); pitch=np.radians(8)
    R=rot(yaw,pitch); scale=4.3*SS
    x,y,z,f=project(P,c,yaw,pitch,scale=scale,off=(0,75*SS)); x=(x-W/2)+WW/2; y=(y-H/2)+HH/2
    x=x*1; Nc=Nn@R.T
    face=Nc[:,2]>-0.05
    diff=np.clip(Nc@LD,0,1); rim=np.clip(Nc@RIM,0,1)**2; amb=0.5+0.5*Nc[:,1]
    half=(LD+np.array([0,0,1])); half/=np.linalg.norm(half); spec=np.clip(Nc@half,0,1)**40
    col=COL*(0.34+0.5*diff[:,None]+0.16*amb[:,None])+0.35*spec[:,None]+0.18*rim[:,None]
    col=np.clip(col,0,1)
    r=np.clip(1.75*(1+m*(SV1-1))*f*SS*0.9,2.0,10.0)
    img,zb=splat(x[face],y[face],z[face],col[face],r[face])
    # the 407 real spots, riding the same warp at their own model positions
    qx,qy,qz,qf=project(Q,c,yaw,pitch,scale=scale,off=(0,75*SS)); qx=qx-W/2+WW/2; qy=qy-H/2+HH/2
    im=(np.clip(img,0,1)*255).astype(np.uint8)
    a=np.full((HH,WW,1),255,np.uint8); im=np.concatenate([im,a],2)
    surf=skia.Surface(np.ascontiguousarray(im)); cv=surf.getCanvas()
    # soft floor shadow
    sh=skia.Paint(AntiAlias=True,MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle,40*SS)); sh.setColor4f(skia.Color4f(0.15,0.17,0.22,0.10))
    Sx=P[::400].copy(); Sx[:,1]=P[:,1].min()-4; sx,sy,_,_=project(Sx,c,yaw,pitch,scale=scale,off=(0,75*SS))
    cv.drawOval(skia.Rect(float(sx.min()-WW/2+WW/2-W/2*0), float(sy.mean()-HH/2+HH/2-25*SS-H/2*0), float(sx.max()), float(sy.mean()+25*SS)),sh) if False else None
    img2=surf.makeImageSnapshot().resize(W,H,skia.SamplingOptions(skia.CubicResampler.Mitchell()))
    s2=skia.Surface(W,H); c2=s2.getCanvas(); c2.clear(skia.ColorWHITE); c2.drawImage(img2,0,0)
    tx=[("a smooth body surface",W/2,128,78,fade(t,0.6,4.8)),("the same surface inside Qwen3-32B",W/2,128,78,fade(t,12.6,19.4))]
    for (txt,tx_,ty,size,al) in tx:
        if al<=0.003: continue
        font=skia.Font(FONT,size); wd=font.measureText(txt); pt=skia.Paint(AntiAlias=True); pt.setColor4f(skia.Color4f(0.07,0.07,0.09,float(al)))
        c2.drawString(txt,float(tx_-wd/2),float(ty),font,pt)
    out=s2.makeImageSnapshot().toarray()
    fo=min(1,t/1.0,(T-t)/0.8); out[:,:,:3]=(out[:,:,:3]*fo+255*(1-fo)).astype(np.uint8)
    return out
def control(): return np.array([amt(k/FPS) for k in range(NF)])
if __name__=="__main__":
    if sys.argv[1]=='still':
        import PIL.Image as I
        for s_ in map(float,sys.argv[2:]): I.fromarray(frame(int(s_*FPS))[:,:,:3]).save(f'v3_{s_:04.1f}.png')
    elif sys.argv[1]=='seg': seg(frame,int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    else:
        np.save('v3_ctrl.npy',control()); encode(__file__,NF,os.path.expandvars('$BODY_REPO/figures/out/body_video_smooth'),workers=9)
