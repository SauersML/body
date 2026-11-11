import os
# Per-vertex color field on the neutral mesh: geodesic Gaussian blend of per-spot values, fading to neutral skin far from spots.
import numpy as np, trimesh, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
M=np.load("neutral_mesh.npz"); body=trimesh.Trimesh(M['v'],M['f'],process=False)
R=np.load("reg_all.npz"); d=np.load(os.path.expandvars('$BODY_REPO/figures/src/nn2.npz'))
V=body.vertices; F=body.faces
pos=(V[F[R['tid']]]*R['bary'][:,:,None]).sum(1)          # spot positions carried onto the neutral mesh
# the neutral head was re-laid (vertices moved), so head spots are carried by a ray from the head centre instead
RI=trimesh.ray.ray_triangle.RayMeshIntersector(body); C0=np.array([0,164.0,3.0])
for i in np.where(d['reg']=='head')[0]:
    dv=R['pos'][i]-C0; dv/=np.linalg.norm(dv); loc,_,_=RI.intersects_location([C0],[dv]); t=(loc-C0)@dv; pos[i]=loc[np.argmin(np.where(t>0,t,1e9))]
print("head spots moved by the re-laid head: median %.1f cm"%np.median(np.linalg.norm(pos-R['pos'],axis=1)[d['reg']=='head']))
vi=np.argmin(((V[None,:,:]-pos[:,None,:])**2).sum(-1),1) if False else trimesh.proximity.ProximityQuery(body).vertex(pos)[1]
e=body.edges_unique; w=body.edges_unique_length
G=coo_matrix((np.r_[w,w],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(V),)*2).tocsr()
LIM=30.0
D=dijkstra(G,indices=vi,limit=LIM).astype(np.float32)        # (407, nV) geodesic cm, inf beyond LIM
Ds=D[:,vi].copy(); np.fill_diagonal(Ds,np.inf); nn=Ds.min(1)
sig=np.clip(0.9*nn,1.5,6.0); print("nn spacing median %.1f cm, sigma range %.1f-%.1f"%(np.median(nn),sig.min(),sig.max()))
Dm=np.where(np.isinf(D),1e4,D)
Wt=np.exp(-Dm**2/(2*sig[:,None]**2)); ws=Wt.sum(0)
# opacity of the color field: depends on distance to the nearest spot (in units of that spot's sigma)
z=np.min(Dm/sig[:,None],0); al=np.clip((3.5-z)/1.5,0,1); al=al*al*(3-2*al)
STOPS=[(0,(8,29,88)),(8,(34,94,168)),(16,(29,145,192)),(24,(65,182,150)),(32,(161,211,90)),(40,(255,214,0))]
def cmap(v):
    v=np.clip(v,0,40); xs=[s[0] for s in STOPS]; C=np.array([s[1] for s in STOPS],float)
    return np.stack([np.interp(v,xs,C[:,k]) for k in range(3)],1)/255
SKIN=np.array([0.90,0.885,0.87])
def lin(c): return c**2.2
from scipy.spatial.distance import pdist,squareform
P=d['P'].copy(); P[:,0]=abs(P[:,0]); DF=squareform(pdist(P)); tw=d['tw']; DD=DF.copy(); np.fill_diagonal(DD,np.inf); ok=tw>=0; DD[np.where(ok)[0],tw[ok]]=np.inf
ratio=d['d_32B']/np.sort(DD,1)[:,:5].mean(1); np.save(os.path.expandvars('$BODY_REPO/figures/bodyvid3/ratio_nn2order.npy'),ratio)
val=(Wt.T@((ratio-1)/2.5*40))/np.maximum(ws,1e-12)
c=lin(cmap(val))*al[:,None]+lin(SKIN)*(1-al[:,None])
(np.clip(c,0,1)**(1/2.2)*255).round().astype(np.uint8).tofile(os.path.expandvars("$BODY_REPO/figures/bodyvid3/web/colratio.bin"))
print('ok')
