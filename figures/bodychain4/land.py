import numpy as np, scipy.sparse as sp, json
from scipy.sparse.csgraph import dijkstra
P=np.load('meshP.npy'); A=sp.load_npz('meshA.npz')
def snap(x,y,z=None,face=None,r=0.25):
    if face is None: return int(np.argmin(((P-np.array([x,y,z]))**2).sum(1)))
    ax,sg={'front':(2,1),'back':(2,-1),'out':(0,1),'top':(1,1),'bottom':(1,-1)}[face]
    other=[i for i in range(3) if i!=ax]; c={0:x,1:y,2:z}
    m=np.ones(len(P),bool)
    for o in other:
        if c[o] is not None: m&=np.abs(P[:,o]-c[o])<r
    i=np.where(m)[0]; return int(i[np.argmax(sg*P[i,ax])])
L=dict(thumb=9433,index=9633,middle=8900,ring=9711,little=9197)
# hand normals for palm/knuckles
F=np.load('meshF.npy'); 
def normals():
    N=np.zeros_like(P)
    for a,b,c in [(0,1,2),(0,2,3)]:
        n=np.cross(P[F[:,b]]-P[F[:,a]],P[F[:,c]]-P[F[:,a]])
        for k in (a,b,c): np.add.at(N,F[:,k],n)
    return N/np.linalg.norm(N,axis=1,keepdims=True)
N=normals()
hc=np.array([4.42,2.15,2.2]); near=np.where(np.linalg.norm(P-hc,axis=1)<0.35)[0]
L['palm']=int(near[np.argmin(N[near,1])]); L['backhand']=int(near[np.argmax(N[near,1])])
L.update(wrist=snap(4.1,2.6,1.7), forearm=snap(3.65,3.0,0.85), elbow=snap(3.1,3.55,-0.1), upperarm=snap(2.5,4.4,0.1,'out' ) if False else snap(2.6,4.35,0.15),
         shoulder=snap(1.8,None,0.1,'top'))
L.update(hip=snap(None,-0.4,0.0,'out'), thigh=snap(1.2,-2.4,None,'front'), knee=snap(1.45,-4.6,None,'front'), shin=snap(1.75,-6.0,None,'front'),
         ankle=snap(None,-7.55,0.0,'out'), heel=snap(2.0,None,-0.42,'bottom') , sole=snap(2.0,None,0.6,'bottom'), toes=snap(2.0,-8.0,None,'front',r=0.15))
L.update(crown=snap(0,None,0.3,'top'), forehead=snap(0,7.7,None,'front'), cheek=snap(0.5,6.95,None,'front',r=0.08), ear=snap(None,7.3,0.4,'out'),
         lips=snap(0,6.72,None,'front',r=0.06), neckF=snap(0,5.95,None,'front'), neckB=snap(0,6.0,None,'back'), chest=snap(0,4.2,None,'front'),
         upback=snap(0,4.2,None,'back'), belly=snap(0,1.3,None,'front'), lowback=snap(0,1.3,None,'back'))
L['tongue']=L['lips']
for k,i in L.items(): print(f"{k:9s}", P[i].round(2), N[i].round(1))
json.dump(L,open('landmarks.json','w'))
