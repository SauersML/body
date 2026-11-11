import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
P=np.load('meshP.npy');F=np.load('meshF.npy')
E=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,3]],F[:,[3,0]]]); E=np.unique(np.sort(E,1),axis=0)
w=np.linalg.norm(P[E[:,0]]-P[E[:,1]],axis=1); n=len(P)
A=sp.coo_matrix((w,(E[:,0],E[:,1])),shape=(n,n)).tocsr(); A=A+A.T; sp.save_npz('meshA.npz',A)
def near(p): return int(np.argmin(((P-np.array(p))**2).sum(1)))
wr=near((4.15,2.35,0.5)); print('wrist',P[wr])
d=dijkstra(A,indices=wr)
hand=np.where((P[:,0]>3.6)&(P[:,1]<2.3))[0]
# local maxima of d among neighbours
nb=[A[i].indices for i in range(n)]
tips=[i for i in hand if all(d[i]>=d[j] for j in nb[i]) and d[i]>0.4]
for i in sorted(tips,key=lambda i:-d[i]): print(i,P[i].round(3),round(d[i],3))
