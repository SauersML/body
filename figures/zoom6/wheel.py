# color wheel = the exact PCA->color map used for the skin (colour.py oklab), as an image over PC1/PC2 in [-R,R]
import numpy as np, json
from PIL import Image
exec(open('colour.py').read().split('SKIN=')[0].split('Z=states()')[0].split('import numpy')[0]) if False else None
LF=0.78
def oklab(c):
    r=np.linalg.norm(c,axis=1); th=np.arctan2(c[:,1],c[:,0]); rs=np.sqrt(1-np.exp(-r**2/2))
    return np.stack([LF+0*rs,0.165*rs*np.cos(th),0.165*rs*np.sin(th)],1)
def lab2lin(c):
    l_=c[:,0]+0.3963377774*c[:,1]+0.2158037573*c[:,2]; m_=c[:,0]-0.1055613458*c[:,1]-0.0638541728*c[:,2]; s_=c[:,0]-0.0894841775*c[:,1]-1.2914855480*c[:,2]
    l,m,s=l_**3,m_**3,s_**3
    return np.clip(np.stack([4.0767416621*l-3.3077115913*m+0.2309699292*s,-1.2684380046*l+2.6097574011*m-0.3413193965*s,-0.0041960863*l-0.7034186147*m+1.7076147010*s],1),0,1)
R=3.0; N=1000
g=np.linspace(-R,R,N); X,Y=np.meshgrid(g,-g)
c=np.c_[X.ravel(),Y.ravel()]; rgb=(lab2lin(oklab(c))**(1/2.2)*255).round().astype(np.uint8).reshape(N,N,3)
a=np.full((N,N),255,np.uint8); rr=np.hypot(X,Y); a[rr>R]=0
Image.fromarray(np.dstack([rgb,a])).save('wheel.png')
