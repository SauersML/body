import numpy as np, scipy.sparse as sp, json
from scipy.sparse.csgraph import dijkstra
P=np.load('meshP.npy'); A=sp.load_npz('meshA.npz'); L=json.load(open('landmarks.json')); tk=json.load(open('tk.json'))
ids=[L[t] for t in tk]; D=dijkstra(A,indices=ids)[:,ids]*10   # dm -> cm
ti=tk.index('tongue'); li=tk.index('lips'); D[ti]+=3; D[:,ti]+=3; D[ti,ti]=0
X=P[ids]*10; E=np.sqrt(((X[:,None]-X[None])**2).sum(-1)); E[ti]+=3; E[:,ti]+=3; E[ti,ti]=0
np.save('skin_cm.npy',D); np.save('straight_cm.npy',E)
for a,b in [('thumb','shoulder'),('toes','hip'),('lips','tongue'),('thumb','index'),('chest','upback'),('hip','belly')]:
    print(a,b,round(D[tk.index(a),tk.index(b)],1),round(E[tk.index(a),tk.index(b)],1))
