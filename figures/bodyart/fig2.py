import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb as c2r
from chains import *; from colors import pcs
from scipy.linalg import orthogonal_procrustes
fm.fontManager.addfont('/System/Library/Fonts/Avenir Next.ttc'); plt.rcParams['font.family']='Avenir Next'
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
Fd=np.c_[abs(SP[:,0]),SP[:,1],SP[:,2]]; mu=Fd.mean(0); Fd=Fd-mu
def view(Z,k=3):
    Y,_=pcs(Z,k); Y=Y-Y.mean(0); Rm,_=orthogonal_procrustes(Y,Fd); V=Y@Rm; sc=np.sqrt((Fd**2).sum()/(V**2).sum()); return V*sc+mu
INK='#1D1D1F'; TR='#33407A'; ARM=('#F08A4B','#B8325A'); LEG=('#3FB8A3','#1F5F7A')
reg_col={'head':TR,'neck':TR,'torso':TR}
def lerp(a,b,t): a,b=np.array(c2r(a)),np.array(c2r(b)); return a+(b-a)*t
def spot_col(i):
    r=SR[i]
    if r in reg_col: return c2r(TR)
    if r.startswith('arm') or r.startswith('hand'): return lerp(*ARM,np.clip((abs(SP[i,0])-16)/84,0,1))
    t=np.clip((98-SP[i,1])/98,0,1); t=0.85*t+0.15*(SP[i,2]>12)
    return lerp(*LEG,t)
SC=np.array([spot_col(i) for i in range(len(SP))])
def cr(P,n=12):
    P=np.vstack([P[0],P,P[-1]]); out=[]
    for i in range(1,len(P)-2):
        p0,p1,p2,p3=P[i-1],P[i],P[i+1],P[i+2]
        for t in np.linspace(0,1,n,endpoint=False):
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    out.append(P[-2]); return np.array(out)
def draw(ax,V,lw=9,dot=70,full=False):
    ax.scatter(V[:,0],V[:,1],s=dot,c=SC,alpha=0.28,lw=0,zorder=1)
    segs=[]
    for side in ('left','right'):
        for nm,ch in chains(side).items():
            P=np.array([V[g].mean(0) for g in ch])[:,:2]; Q=cr(P); t=np.linspace(0,1,len(Q))
            cols=[lerp(*(ARM if nm=='arm' else LEG),tt) for tt in t]
            lc=LineCollection(np.stack([Q[:-1],Q[1:]],1),colors=cols,lw=lw,capstyle='round',joinstyle='round',alpha=0.9,zorder=3); ax.add_collection(lc)
    P=np.array([V[g].mean(0) for g in spine()])[:,:2]; Q=cr(P); ax.plot(Q[:,0],Q[:,1],c=TR,lw=lw,solid_capstyle='round',zorder=4,alpha=.95)
    ax.scatter(P[0,0],P[0,1],s=lw*lw*9,c=TR,zorder=5)
    ax.set_aspect('equal'); ax.axis('off')
if __name__=='__main__':
    real=np.c_[SP[:,0],SP[:,1],SP[:,2]]
    V53=view(R['reply|53|all|A'].astype(np.float32))
    fig=plt.figure(figsize=(30,24),dpi=100)
    ax0=fig.add_axes([0.01,0.40,0.38,0.52]); ax1=fig.add_axes([0.42,0.37,0.56,0.56])
    # real: full body, but chains drawn with true x
    VV=real.copy(); draw(ax0,VV,lw=9)
    ax0.set_xlim(-105,105); ax0.set_ylim(-8,185)
    draw(ax1,V53,lw=12,dot=110); x0,x1=ax1.get_xlim()
    fig.text(0.20,0.955,'a body',fontsize=64,color=INK,ha='center'); fig.text(0.70,0.955,'the same spots inside Qwen3-32B',fontsize=64,color=INK,ha='center')
    def end(nm,side='left',k=-1):
        ch=chains(side)[nm]; return np.mean([V53[ch[k]].mean(0) for side in ('left','right') for ch in [chains(side)[nm]]],0)
    hp=end('arm'); fp=end('leg'); hd=V53[spine()[0]].mean(0)
    hy=V53[np.isin(SR,['handL','handR']),1]; ax1.text(hp[0]-4,np.percentile(hy,97)+5,'left and right hand',fontsize=60,color=ARM[1],va='bottom',ha='center')
    ax1.text(fp[0]+9,fp[1]+8,'left and right foot',fontsize=60,color=LEG[1],va='center',ha='left')
    ax1.text(hd[0]-5,hd[1],'head',fontsize=60,color=TR,va='center',ha='right')
    ax1.set_xlim(x0,x1+42)
    Ls=[28,41,48,50,53]; w=0.96/len(Ls)
    for k,L in enumerate(Ls):
        a=fig.add_axes([0.02+k*w,0.01,w*0.95,0.26]); draw(a,view(R[f'reply|{L}|all|A'].astype(np.float32)),lw=6,dot=30)
        fig.text(0.02+k*w+w*0.475,0.30,f'layer {L}',fontsize=60,color=INK,ha='center')
    fig.savefig('f2_raw.png',facecolor='white'); 
