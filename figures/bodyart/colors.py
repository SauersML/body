import numpy as np
D='$BODY_WORK/body/hires2/'
def pcs(Z,k=3):
    Z=Z-Z.mean(0); u,s,vt=np.linalg.svd(Z,full_matrices=False); Y=u[:,:k]*s[:k]; return Y/Y.std(0), (s**2/(s**2).sum())[:k]
def oklab_to_srgb(L,a,b):
    l=(L+0.3963377774*a+0.2158037573*b)**3; m=(L-0.1055613458*a-0.0638541728*b)**3; s=(L-0.0894841775*a-1.2914855480*b)**3
    r=4.0767416621*l-3.3077115913*m+0.2309699292*s; g=-1.2684380046*l+2.6097574011*m-0.3413193965*s; bb=-0.0041960863*l-0.7034186147*m+1.7076147010*s
    lin=np.stack([r,g,bb],-1)
    return np.where(lin<=0.0031308,12.92*lin,1.055*np.clip(lin,0,None)**(1/2.4)-0.055)
def to_rgb(Y, Lc=0.70, Ls=0.10, C=0.14):
    t=np.tanh(Y/1.6)
    L=Lc+Ls*t[:,2]; a=C*t[:,0]; b=C*t[:,1]
    # gamut: shrink chroma until in range
    out=oklab_to_srgb(L,a,b)
    for _ in range(20):
        bad=((out<0)|(out>1)).any(-1)
        if not bad.any(): break
        a[bad]*=0.93; b[bad]*=0.93; out=oklab_to_srgb(L,a,b)
    return np.clip(out,0,1)
def data():
    z32=np.load(D+'emb3_b53.npz')['ZR']; z8=np.load(D+'emb3_s27.npz')['ZR']
    rnd=np.random.default_rng(0).standard_normal(z32.shape)
    return dict(q32=z32,q8=z8,rnd=rnd)
from scipy.stats import norm, rankdata
def gauss(Y): return norm.ppf((rankdata(Y,axis=0)-0.5)/len(Y))
def to_rgb2(Y, theta=0., Lc=0.66, Ls=0.16, C=0.15, sq=1.3, mode='rank'):
    G=gauss(Y) if mode=='rank' else Y
    t=np.tanh(G/sq); a0=C*t[:,0]; b0=C*t[:,1]
    a=np.cos(theta)*a0-np.sin(theta)*b0; b=np.sin(theta)*a0+np.cos(theta)*b0
    L=Lc+Ls*t[:,2]; out=oklab_to_srgb(L,a,b)
    for _ in range(30):
        bad=((out<0)|(out>1)).any(-1)
        if not bad.any(): break
        a[bad]*=0.93; b[bad]*=0.93; out=oklab_to_srgb(L,a,b)
    return np.clip(out,0,1)
