import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb as c2r
from chains import *
fm.fontManager.addfont('/System/Library/Fonts/Avenir Next.ttc'); plt.rcParams['font.family']='Avenir Next'
INK='#1D1D1F'; TR='#33407A'; ARM=('#F08A4B','#B8325A'); LEG=('#3FB8A3','#1F5F7A')
def lerp(a,b,t): a,b=np.array(c2r(a)),np.array(c2r(b)); return a+(b-a)*t
def spot_col(i):
    r=SR[i]
    if r in ('head','neck','torso'): return c2r(TR)
    if r.startswith('arm') or r.startswith('hand'): return lerp(*ARM,np.clip((abs(SP[i,0])-16)/84,0,1))
    return lerp(*LEG,np.clip((98-SP[i,1])/98,0,1))
SC=np.array([spot_col(i) for i in range(len(SP))])
CH=[(nm,ch) for s in ('left','right') for nm,ch in chains(s).items()]; SPN=spine()
def cr(P,n=10):
    P=np.vstack([P[0],P,P[-1]]); out=[]
    for i in range(1,len(P)-2):
        p0,p1,p2,p3=P[i-1],P[i],P[i+1],P[i+2]
        for t in np.linspace(0,1,n,endpoint=False):
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    out.append(P[-2]); return np.array(out)
def draw(ax,V,lw=10,dot=90):
    ax.scatter(V[:,0],V[:,1],s=dot,c=SC,alpha=0.30,lw=0,zorder=1)
    for nm,ch in CH:
        P=np.array([V[g].mean(0) for g in ch]); Q=cr(P); t=np.linspace(0,1,len(Q))
        cols=[lerp(*(ARM if nm=='arm' else LEG),tt) for tt in t]
        ax.add_collection(LineCollection(np.stack([Q[:-1],Q[1:]],1),colors=cols,lw=lw,capstyle='round',joinstyle='round',alpha=0.92,zorder=3))
    P=np.array([V[g].mean(0) for g in SPN]); Q=cr(P); ax.plot(Q[:,0],Q[:,1],c=TR,lw=lw,solid_capstyle='round',zorder=4)
    ax.scatter(P[0,0],P[0,1],s=lw*lw*9,c=TR,zorder=5)
    ax.set_aspect('equal'); ax.axis('off')
