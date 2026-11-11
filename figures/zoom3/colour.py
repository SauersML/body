import os
import numpy as np, trimesh, pickle, json, sys
sys.path.insert(0,os.path.expandvars("$BODY_REPO/figures/zoom2"))
exec(open(os.path.expandvars('$BODY_REPO/figures/src/reg.py')).read())
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
Z=states()
def rho(key, ix):
    iu=np.triu_indices(len(ix),1); return float(spearmanr(squareform(pdist(Z[key][ix]))[iu], squareform(pdist(P[ix]))[iu])[0])
def pca2(X):
    X=X-X.mean(0); u,s,vt=np.linalg.svd(X,full_matrices=False); c=u[:,:2]*s[:2]; return c/c.std(0)
def procrustes(A,B): u,_,vt=np.linalg.svd(A.T@B); return A@(u@vt)
LF=0.78
def oklab(c):
    r=np.linalg.norm(c,axis=1); th=np.arctan2(c[:,1],c[:,0]); rs=np.sqrt(1-np.exp(-r**2/2))
    return np.stack([LF+0*rs,0.165*rs*np.cos(th),0.165*rs*np.sin(th)],1)
def lab2lin(c):
    l_=c[:,0]+0.3963377774*c[:,1]+0.2158037573*c[:,2]; m_=c[:,0]-0.1055613458*c[:,1]-0.0638541728*c[:,2]; s_=c[:,0]-0.0894841775*c[:,1]-1.2914855480*c[:,2]
    l,m,s=l_**3,m_**3,s_**3
    return np.clip(np.stack([4.0767416621*l-3.3077115913*m+0.2309699292*s,-1.2684380046*l+2.6097574011*m-0.3413193965*s,-0.0041960863*l-0.7034186147*m+1.7076147010*s],1),0,1)
SKIN=np.array([0.86,0.030,0.040])
M=np.load("mesh.npz"); body=trimesh.Trimesh(M['v'],M['f'],process=False); R=pickle.load(open("reg.pkl","rb")); res=R['res']
V=body.vertices; e=body.edges_unique; w=body.edges_unique_length
G=coo_matrix((np.r_[w,w],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(V),)*2).tocsr()
out={}; cols={}
from scipy.spatial import cKDTree
kd=cKDTree(V)
for reg in ["face","hand","foot","forearm"]:
    ix=REG[reg]; pos=res[reg]['pos']; c32=pca2(Z["32B"][ix]); cw=procrustes(pca2(Z["words"][ix]),c32)
    _,vi=kd.query(pos)
    D=dijkstra(G,indices=vi,limit=12.0)          # geodesic (mesh-edge) distance from each spot, cm
    Ds=D[:,vi]; np.fill_diagonal(Ds,np.inf); nn=Ds.min(1); sig=np.clip(0.9*nn,0.8*np.median(nn),1.6*np.median(nn))
    Wt=np.exp(-np.where(np.isinf(D),1e9,D)**2/(2*sig[:,None]**2))   # (spots, verts)
    ws=Wt.sum(0); al=1-np.exp(-ws/0.35)
    for src,c in [("m",c32),("w",cw)]:
        lab=(Wt.T@oklab(c))/np.maximum(ws,1e-9)[:,None]; L=SKIN*(1-al[:,None])+lab*al[:,None]
        cols[f"{reg}_{src}"]=(np.clip(lab2lin(L),0,1)**(1/2.2)*255).round().astype(np.uint8)  # store gamma-encoded, decode in shader
    out[reg]=dict(pos=pos.round(3).tolist(),nrm=res[reg]['nrm'].round(3).tolist(),r32=rho("32B",ix),rw=rho("words",ix),snap=res[reg]['dist'].round(2).tolist(),
                  phr=res[reg]['phr'],sigma=float(np.median(nn)))
    print(reg,"geodesic nn spacing median %.2f cm"%np.median(nn),"rho %.2f %.2f"%(out[reg]['r32'],out[reg]['rw']))
# write binary
V32=V.astype(np.float32); N32=body.vertex_normals.astype(np.float32); F32=body.faces.astype(np.uint32)
V32.tofile("geo_pos.bin"); N32.tofile("geo_nrm.bin"); F32.tofile("geo_idx.bin")
for k,c in cols.items(): c.tofile(f"col_{k}.bin")
# eyes: sclera + iris
J=pickle.load(open("joints.pkl","rb")); eyes=[]
for s,k in [("l",0),("r",1)]:
    ev=M[f'ev{k}']; ef=M[f'ef{k}']; c=ev.mean(0); tgt=nz=J[f'{s}-eye-target']-J[f'{s}-eye']; tgt/=np.linalg.norm(tgt)
    cosang=((ev-c)/np.linalg.norm(ev-c,axis=1,keepdims=True))@tgt
    col=np.where(cosang[:,None]>0.93,[0.10,0.07,0.05],np.where(cosang[:,None]>0.80,[0.30,0.22,0.15],[0.93,0.91,0.89]))
    eyes.append(dict(v=ev.round(3).tolist(),f=ef.tolist(),c=col.round(3).tolist()))
out["_eyes"]=eyes; out["_nv"]=len(V); out["_nf"]=len(F32)
out["_frames"]={k:np.asarray(v).round(4).tolist() for k,v in R['frames'].items()}
json.dump(out,open("spots.json","w"))
