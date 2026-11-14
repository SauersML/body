import json, numpy as np
from PIL import Image
d=json.load(open('mapdata.json'))
X0,X1,Y0,Y1=-72,58,-58,82; NX=1800; NY=int(NX*(Y1-Y0)/(X1-X0))
C={'sitting':(29,127,140),'standing':(84,96,214),'lying':(181,81,122)}
xs=np.linspace(X0,X1,NX); ys=np.linspace(Y1,Y0,NY)
def kde(Z,bw):
    gx=np.exp(-(xs[None,:]-Z[:,0:1])**2/(2*bw*bw)); gy=np.exp(-(ys[None,:]-Z[:,1:2])**2/(2*bw*bw))
    return gy.T@gx
px_per=NX/(X1-X0)
for t in d:
    A=[];Cs=[]
    for p,c in C.items():
        Z=np.array(d[t]['clouds'][p]); D=kde(Z,7.0)
        # density level enclosing 50% / 90% of the sentences
        dz=np.array([D[min(max(int((Y1-z[1])/(Y1-Y0)*(NY-1)),0),NY-1), min(max(int((z[0]-X0)/(X1-X0)*(NX-1)),0),NX-1)] for z in Z])
        l50,l90=np.percentile(dz,50),np.percentile(dz,10)
        sm=lambda v,l,w: 1/(1+np.exp(-(v-l)/(w*l)))
        a=0.22*sm(D,l90,0.06)+0.38*sm(D,l50,0.04)
        # outline of the 50% region
        e=np.exp(-((D-l50)/(0.035*l50))**2); a=np.maximum(a,0.95*e*(D>0))
        A.append(a); Cs.append(np.array(c,float))
    A=np.array(A); tot=1-np.prod(1-A,0); col=np.einsum('pij,pc->ijc',A,np.array(Cs))/np.maximum(A.sum(0),1e-9)[...,None]
    Image.fromarray((255*(1-tot[...,None])+col*tot[...,None]).clip(0,255).astype(np.uint8)).save(f'field_{t}.png')
json.dump(dict(X0=X0,X1=X1,Y0=Y0,Y1=Y1,NX=NX,NY=NY),open('field.json','w')); print(NX,NY)
