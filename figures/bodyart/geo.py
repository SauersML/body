import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from mesh import load
P,T=load(); C=np.load('C.npy')
E=np.vstack([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]); w=np.linalg.norm(C[E[:,0]]-C[E[:,1]],axis=1)
G=sp.coo_matrix((w,(E[:,0],E[:,1])),shape=(len(C),)*2).tocsr(); G=G.maximum(G.T)
x,y,z=C.T
arm=((np.abs(x)>=17)&(y>115))|((np.abs(x)>=19)&(y>92))
seed=np.where((np.abs(x)<15)&(y>110)&(y<150))[0]
d=dijkstra(G,indices=seed,min_only=True)
np.savez('geo.npz',arm=arm,d=d)
